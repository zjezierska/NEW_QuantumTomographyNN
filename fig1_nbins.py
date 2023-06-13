import time
import numpy as np
import tensorflow as tf
import qutip as qt
import matplotlib.pyplot as plt

start_time = time.time()

d = 4
traj_length = 20
patienc = 500
batchsize = 512
epochz = 10000
nof_samples = 9000
bins_to_check = np.arange(2, 42, 2)

y_states = np.load('new_data/states.npy')
y_states_valid = y_states[nof_samples // 2:, :]
y_states = y_states[:nof_samples // 2, :]


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


def init_net(num_bin):  # creating and compiling the network
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(800, input_shape=(num_bin * traj_length,), activation='sigmoid'))
    net.add(tf.keras.layers.Dense(800, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(400, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(200, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(2 * d ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam')
    return net


def create_bins(num_samples, num_bin):
    samplez = []
    for i in range(num_samples):
        bin_heights = []
        trajectory = np.load(f"new_data/drawn_points/one_sample{i+1}.npy")
        for j in range(traj_length):
            heights, bins = np.histogram(trajectory[j], num_bin,
                                         density=True)  # getting heights from the histogram of samples
            bin_heights.append(heights)

        samplez.append([element for sublist in bin_heights for element in sublist])

    return samplez


def my_fidelity(vec1, vec2):  # normalizing vec2 and calculating fidelity between two states in vector form
    vec1 = give_back_matrix(vec1)
    vec2 = give_back_matrix(vec2)
    if vec1.isherm:
        vec2 = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()
        return qt.fidelity(vec1, vec2)
    else:
        raise ValueError('X is not Hermitian!')


def give_back_matrix(vectr):  # turn the 2d**2 vector back into Qobj matrix
    vec = vectr.reshape(2, d ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(d, d)
    return qt.Qobj(matrix)


def get_infidelity(num_bin):
    print(f"BEGINNING {num_bin} BINS")
    y_bins = create_bins(nof_samples, num_bin)

    n = nof_samples // 2
    y_bins_valid = y_bins[n:]
    y_bins = y_bins[:n]

    y_bins = np.array(y_bins)
    y_bins_valid = np.array(y_bins_valid)

    model = init_net(num_bin)

    callbackz = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min'),
                 tf.keras.callbacks.ModelCheckpoint(filepath=f'new_models/best_model_draw.h5', monitor='val_loss',
                                                    save_best_only=True,
                                                    mode='min')]

    model.fit(y_bins, y_states, batch_size=batchsize, epochs=epochz,
              validation_data=(y_bins_valid, y_states_valid),
              callbacks=callbackz)  # training the network

    model1 = tf.keras.models.load_model(f'new_models/best_model_draw.h5',
                                        custom_objects={'custom_loss': custom_loss})

    valid_predict = model1.predict_on_batch(y_bins_valid)
    fidelities = [my_fidelity(y_states_valid[i, :], valid_predict[i, :]) for i in range(n)]

    return 1 - np.average(fidelities)


plot_points = [get_infidelity(i) for i in bins_to_check]

with open(f'plot_points_{nof_samples//2}.txt', 'w') as f:
    for item in plot_points:
        f.write(f"{item} \n")

fig, ax = plt.subplots()
ax.plot(bins_to_check, plot_points, '-o')
ax.set_xlabel(r'Number of bins $N$')
ax.set_ylabel(r'Average infidelity $1 - F$')
ax.set_yscale('log')
plt.savefig(f"nbins_{nof_samples//2}_betterbatchsize.png", bbox_inches='tight')
plt.show()
