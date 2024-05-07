import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import time
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

traj_length = 200  # how many "snapshots" in time we take
nof_samples_distr = 200 # how many points to sample from distribution
checking_step = 10
beginning_m = 1

checking_arr = np.arange(beginning_m, traj_length+1, checking_step, dtype=int)
checking_arr = np.append(checking_arr, traj_length)

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
    net.add(Dense(512, input_shape=(num_points_px * num_points_time,), activation='relu'))
    net.add(Dropout(0.2))

    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.2))

    net.add(Dense(512, activation='relu'))

    net.add(Dense(2 * (d ** 2), activation='tanh'))

    # Using the Adam optimizer
    optimizer = Adam()

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
        trajectory = np.load(f"/net/ascratch/people/plgredhoodies/data_harmonic/one_sample_{i}.npy")

        # Convert the list of lists to a 2D NumPy array
        data_array = np.array(trajectory)

        # Flatten the trajectory and append it to the samples list
        samplez.append(data_array)

    return samplez


y_states1 = np.load('/net/ascratch/people/plgredhoodies/data_harmonic/states.npy')
y_states = y_states1[:half_sam, :]
y_states_valid = y_states1[half_sam:, :]

y_bins1 = create_points_samples(samples)   

y_bins = y_bins1[:half_sam]
y_bins_valid = y_bins1[half_sam:]

in_fidelities = []
num_epochs_array = np.empty((nof_samples_distr,traj_length))
time_array = np.empty((nof_samples_distr,traj_length))
infidelity_array = np.empty((nof_samples_distr,traj_length))
median_infid_array = np.empty((nof_samples_distr,traj_length))
std_array = np.empty((nof_samples_distr,traj_length))
min_array = np.empty((nof_samples_distr,traj_length))
max_array = np.empty((nof_samples_distr,traj_length))

for num_points_px in checking_arr:
    in_fidelities_1 = []
    for num_points_time in checking_arr:

        K.clear_session()

        print(f"DOING {num_points_px} POINTS IN P(x), {num_points_time} POINTS IN TIME ... ")
        beginning = time.time()

        y_bins_final = [sample_dataset(x, num_points_px, num_points_time) for x in y_bins]
        y_bins_valid_final = [sample_dataset(x, num_points_px, num_points_time) for x in y_bins_valid]

        y_bins_final = np.array(y_bins_final)
        y_bins_valid_final = np.array(y_bins_valid_final)

        print(y_bins_final[:][:20])

        model = init_net(num_points_px, num_points_time)  # creating the network

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min')

        filepath = '/net/people/plgrid/plgredhoodies/machine_scripts/models_harmonic_200_10/best_model_draw.h5'

        callbackz = [early_stop,
                     tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                                        save_best_only=True, mode='min')]

        # early stopping and saving the best model, to use in validation set

        history = model.fit(y_bins_final, y_states, batch_size=batchsize, epochs=epochz,
                            validation_data=(y_bins_valid_final, y_states_valid),
                            callbacks=callbackz)  # training the network

        # get the number of epochs at which training stopped
        num_epochs = early_stop.stopped_epoch + 1

        model1 = tf.keras.models.load_model(filepath,custom_objects={'custom_loss': custom_loss})  # loading the best model for the
        # validation set

        avg_infidelities = []
        validation_predict = model1.predict(y_bins_valid_final)  # use the best model on valid_data
        fidelities = [my_fidelity(y_states_valid[i, :], validation_predict[i, :]) for i in range(half_sam)]
        print(f"For {num_points_px} and {num_points_time} score: {1 - np.average(fidelities)}")
        # average INfidelity in validation set
        in_fidelities_1.append(1 - np.average(fidelities))

        print(f"--- {time.time() - beginning} seconds ---")  # how much time did it take

        # save the data to a file in the training_data directory
        num_epochs_array[num_points_time-1, num_points_px-1] = num_epochs
        time_array[num_points_time-1, num_points_px-1] = time.time() - beginning
        infidelity_array[num_points_time-1, num_points_px-1] = 1 - np.average(fidelities)
        median_infid_array[num_points_time-1, num_points_px-1] =  1 - np.median(fidelities)
        std_array[num_points_time-1, num_points_px-1] = np.std(fidelities)
        min_array[num_points_time-1, num_points_px-1] = 1 - np.max(fidelities)
        max_array[num_points_time-1, num_points_px-1] = 1 - np.min(fidelities)



    in_fidelities.append(in_fidelities_1)


np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/num_epochs.npy", num_epochs_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/training_time.npy", time_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/infidelities.npy", infidelity_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/medians.npy", median_infid_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/std.npy", std_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/min.npy", min_array)
np.save(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/max.npy", max_array)


# Convert in_fidelities to a NumPy array
in_fidelities_array = np.array(in_fidelities)

# Create the x and y tick labels
x_tick_labels = checking_arr
y_tick_labels = checking_arr

# Create the 2D plot
plt.figure(figsize=(8, 6))
plt.imshow(in_fidelities_array, origin='lower', cmap='viridis', aspect='auto', norm=colors.LogNorm())
plt.colorbar(label='Infidelity')

# Set the x and y ticks and labels
plt.xticks(np.arange(len(x_tick_labels)), x_tick_labels)
plt.yticks(np.arange(len(y_tick_labels)), y_tick_labels)
plt.xlabel('Number of points in P(x)')
plt.ylabel('Number of points in time')
plt.savefig(f"/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/run_{beginning_m}_{nof_samples_distr}_{checking_step}/punkty_{beginning_m}_{nof_samples_distr}_{checking_step}.png")
