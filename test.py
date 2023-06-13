import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Nadam
from tensorflow.keras.constraints import MaxNorm

def custom_loss(y_true, y_pred):  # MY CUSTOM LOSS FUNCTION - same as in Talitha's approach
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))  # turn vectors into matrices
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])  # connect matrices into complex matrices
    print()
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


samples = 20000
half_sam = samples // 2
d = 4

y_states1 = np.load('new_data/states.npy')
y_states = y_states1[:half_sam, :]
y_states_valid = y_states1[half_sam:, :]

y_bins1 = create_points_samples(samples)

y_bins = y_bins1[:half_sam]
y_bins_valid = y_bins1[half_sam:]

y_bins_final = [sample_dataset(x, 10, 10) for x in y_bins]
y_bins_valid_final = [sample_dataset(x, 10, 10) for x in y_bins_valid]

y_bins_final = np.array(y_bins_final)
y_bins_valid_final = np.array(y_bins_valid_final)

y_bins_final = np.array(y_bins_final, dtype=np.float32)
y_bins_valid_final = np.array(y_bins_valid_final, dtype=np.float32)
y_states = np.array(y_states, dtype=np.float32)
y_states_valid = np.array(y_states_valid, dtype=np.float32)

print(y_bins_final.dtype)
print(y_states.dtype)

model = init_net(10, 10)  # creating the network

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='min')

callbackz = [early_stop, tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/best_model_draw.h5', monitor='val_loss',
                                        save_best_only=True, mode='min')] 

# early stopping and saving the best model, to use in validation set

history = model.fit(y_bins_final, y_states, batch_size=512, epochs=10000,
                    validation_data=(y_bins_valid_final, y_states_valid),
                    callbacks=callbackz)  # training the network

model1 = tf.keras.models.load_model(f'Models/best_model_draw.h5',
                                    custom_objects={'custom_loss': custom_loss})  # loading the best model for the
        # validation set

validation_predict = model1.predict(y_bins_valid_final)  # use the best model on valid_data
fidelities = [my_fidelity(y_states_valid[i, :], validation_predict[i, :]) for i in range(half_sam)]
print(1 - np.average(fidelities))  # average INfidelity in validation set

# DRAWING WIGNER STUFF
fidelity_worst = 1
fidelity_best = 0
infidel = []
previous = 0

min_index = fidelities.index(min(fidelities))
max_index = fidelities.index(max(fidelities))
print(f"best fidelity: {max(fidelities)}")
print(f"worst fidelity: {min(fidelities)}")

mini_norm = give_back_matrix(validation_predict[min_index, :])
pred_drawing_worst = (mini_norm.dag() * mini_norm) / (mini_norm.dag() * mini_norm).tr()

maxi_norm = give_back_matrix(validation_predict[max_index, :])
pred_drawing_best = (maxi_norm.dag() * maxi_norm) / (maxi_norm.dag() * maxi_norm).tr()

true_drawing_worst = give_back_matrix(y_states_valid[min_index, :])
true_drawing_best = give_back_matrix(y_states_valid[max_index, :])

# DRAWING THE BEST AND WORST FIT IN VALID. SET - looks cool
xvec = np.linspace(-5, 5, 200)
W1_good = qt.wigner(true_drawing_best, xvec, xvec)
W2_good = qt.wigner(pred_drawing_best, xvec, xvec)
W1_bad = qt.wigner(true_drawing_worst, xvec, xvec)
W2_bad = qt.wigner(pred_drawing_worst, xvec, xvec)

wmap = qt.wigner_cmap(W1_good)  # can edit colormap, put it in cmap
fig, axs = plt.subplots(2, 2)
plott = axs[0, 0].contourf(xvec, xvec, W1_good, 100, cmap='RdBu_r')
axs[0, 0].set_title("True state - best fidelity")

axs[0, 1].contourf(xvec, xvec, W2_good, 100, cmap='RdBu_r')
axs[0, 1].set_title("Predicted state - best fidelity")

axs[1, 0].contourf(xvec, xvec, W1_bad, 100, cmap='RdBu_r')
axs[1, 0].set_title("True state - worst fidelity")

axs[1, 1].contourf(xvec, xvec, W2_bad, 100, cmap='RdBu_r')
axs[1, 1].set_title("Predicted state - worst fidelity")

fig.suptitle('True vs predicted Wigner function')
fig.tight_layout()
fig.colorbar(plott, ax=axs[:, :], location='right')
plt.savefig('11111_wigner_check.png', bbox_inches='tight')

plt.clf()

x_ax = np.arange(0, len(history.history['loss']), 50)
plt.plot(history.history['loss'])
plt.plot(x_ax, history.history['val_loss'][::50])  # plot both losses during training
plt.title('Model loss')
plt.ylabel('loss functions')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['training set', 'validation set'], loc='upper left')
plt.savefig('11111_plot_loss.png', bbox_inches='tight')
plt.show()