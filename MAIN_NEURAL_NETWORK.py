import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Nadam
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import MaxNorm

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# todo: draw big picture
traj_length = 20  # how many "snapshots" in time we take
nof_samples_distr = 20  # how many points to sample from distribution
checking_step = 1
beginning_m = 1

epochz = 10000
patienc = 500
batchsize = 512
d = 4  # beginning dim

samples = 20000
half_sam = samples // 2


def custom_loss(y_true, y_pred):  # MY CUSTOM LOSS FUNCTION - same as in Talitha's approach
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))  # turn vectors into matrices
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])  # connect matrices into complex matrices

    transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)  # complex conjugate of matrices
    result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)  # M.dag() * M
    final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))  # previous / trace - normalisation

    finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
                                  tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)  # turning
    # it back into the vector

    return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)  # MSE calculation


def init_net(num_points_px, num_points_time):  # creating and compiling the network
    net = tf.keras.models.Sequential()
    net.add(Dense(512, input_shape=(num_points_px * num_points_time,), activation='relu', kernel_constraint=MaxNorm(3)))
    net.add(Dropout(0.2))

    net.add(Dense(256, activation='relu', kernel_constraint=MaxNorm(3)))
    net.add(Dropout(0.2))

    net.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))

    net.add(Dense(2 * d ** 2, activation='tanh'))

    # Using the Adam optimizer with Nesterov momentum and the learning rate schedule
    optimizer = Nadam(learning_rate=0.001)

    net.compile(loss=custom_loss, optimizer=optimizer)
    return net


def give_back_matrix(vectr):
    global d

    # Reshape the vector into a 2D array with real and imaginary parts
    vec = vectr.reshape(2, d**2)

    # Combine the real and imaginary parts to create a complex matrix
    matrix = vec[0, :] + 1j * vec[1, :]

    # Reshape the matrix to have dimensions (d, d)
    matrix = matrix.reshape(d, d)

    # Create a Qobj using the reshaped matrix
    return qt.Qobj(matrix)


def my_fidelity(vec1, vec2):
    # Convert input vectors to Qobj matrices
    vec1 = give_back_matrix(vec1)
    vec2 = give_back_matrix(vec2)

    # Check if vec1 is Hermitian
    if vec1.isherm:
        # Normalize vec2
        vec2_normalized = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()

        # Calculate and return the fidelity between vec1 and the normalized vec2
        return qt.fidelity(vec1, vec2_normalized)
    else:
        raise ValueError('X is not Hermitian!')


def sample_dataset(dataset, num_points_px, num_points_time):
    time_indices = np.linspace(0, dataset.shape[0] - 1, num_points_time, dtype=int)
    px_indices = np.linspace(0, dataset.shape[1] - 1, num_points_px, dtype=int)

    sampled_dataset_2d = dataset[np.ix_(time_indices, px_indices)]

    sampled_dataset = sampled_dataset_2d.flatten()

    return sampled_dataset


def create_points_samples(num_samples):
    samplez = []

    # Iterate over the number of samples
    for i in range(num_samples):
        # Load trajectory data from the file
        trajectory = np.load(f"new_data/drawn_points/one_sample{i}.npy")

        # Convert the list of lists to a 2D NumPy array
        data_array = np.array(trajectory)

        # Flatten the trajectory and append it to the samples list
        samplez.append(data_array)

    return samplez


y_states1 = np.load('new_data/states.npy')
y_states = y_states1[:half_sam, :]
y_states_valid = y_states1[half_sam:, :]

y_bins1 = create_points_samples(samples)

y_bins = y_bins1[:half_sam]
y_bins_valid = y_bins1[half_sam:]

y_bins = np.float32(y_bins)
y_bins_valid = np.float32(y_bins_valid)

y_states = np.float32(y_states)
y_states_valid = np.float32(y_states_valid)

in_fidelities = []
for num_points_px in np.arange(beginning_m, nof_samples_distr + checking_step, checking_step, dtype=int):
    in_fidelities_1 = []
    for num_points_time in np.arange(beginning_m, traj_length + checking_step, checking_step, dtype=int):
        print(f"DOING {num_points_px} POINTS IN P(x), {num_points_time} POINTS IN TIME ... ")
        beginning = time.time()

        y_bins_final = [sample_dataset(x, num_points_px, num_points_time) for x in y_bins]
        y_bins_valid_final = [sample_dataset(x, num_points_px, num_points_time) for x in y_bins_valid]

        y_bins_final = np.array(y_bins_final)
        y_bins_valid_final = np.array(y_bins_valid_final)

        model = init_net(num_points_px, num_points_time)  # creating the network

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min')

        callbackz = [early_stop,
                     tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/best_model_draw.h5', monitor='val_loss',
                                                        save_best_only=True, mode='min')]

        # early stopping and saving the best model, to use in validation set

        history = model.fit(y_bins_final, y_states, batch_size=batchsize, epochs=epochz,
                            validation_data=(y_bins_valid_final, y_states_valid),
                            callbacks=callbackz)  # training the network

        # get the number of epochs at which training stopped
        num_epochs = early_stop.stopped_epoch + 1

        model1 = tf.keras.models.load_model(f'Models/best_model_draw.h5',
                                            custom_objects={
                                                'custom_loss': custom_loss})  # loading the best model for the
        # validation set

        avg_infidelities = []
        validation_predict = model1.predict(y_bins_valid_final)  # use the best model on valid_data
        fidelities = [my_fidelity(y_states_valid[i, :], validation_predict[i, :]) for i in range(half_sam)]
        print(f"For {num_points_px} and {num_points_time} score: {1 - np.average(fidelities)}")
        # average INfidelity in validation set
        in_fidelities_1.append(1 - np.average(fidelities))

        print(f"--- {time.time() - beginning} seconds ---")  # how much time did it take

        # create a directory for saved data if it doesn't exist
        if not os.path.exists('training_data/run_1_20'):
            os.makedirs('training_data/run_1_20')

        # save the data to a file in the training_data directory
        with open(f'training_data/run_1_20/data_{num_points_px}_px_{num_points_time}_t.txt', 'w') as f:
            f.write(f"Num_epochs: {num_epochs}\n")
            f.write(f"Time of training: {time.time() - beginning} s \n")
            f.write(f"Avg infidelity: {1 - np.average(fidelities)} \n")
            f.write(f"Median infidelity: {1 - np.median(fidelities)} \n")
            f.write(f"Std dev. infidelity: {np.std(fidelities)} \n")
            f.write(f"Min. infidelity: {1 - np.max(fidelities)} \n")
            f.write(f"Max. infidelity: {1 - np.min(fidelities)} \n")

        # Clear the Tensorflow session and release the memory
        K.clear_session()

    in_fidelities.append(in_fidelities_1)

# Convert in_fidelities to a NumPy array
in_fidelities_array = np.array(in_fidelities)

# Create the x and y tick labels
x_tick_labels = np.arange(beginning_m, nof_samples_distr+checking_step, checking_step, dtype=int)
y_tick_labels = np.arange(beginning_m, traj_length+checking_step, checking_step, dtype=int)

# Create the 2D plot
plt.figure(figsize=(8, 6))
plt.imshow(in_fidelities_array, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Infidelity')

# todo: make colors logarithmic maybe

# Set the x and y ticks and labels
plt.xticks(np.arange(len(x_tick_labels)), x_tick_labels)
plt.yticks(np.arange(len(y_tick_labels)), y_tick_labels)
plt.xlabel('Number of points in P(x)')
plt.ylabel('Number of points in time')
plt.savefig("punkty_1-20.png")

# # Show the plot
# plt.show()
