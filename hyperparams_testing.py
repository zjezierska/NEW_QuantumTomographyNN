
import qutip as qt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


samples = 20000
batchsize = 512
d = 4  # beginning dim
traj_length = 10  # how many "snapshots" in time we take
nof_samples_distr = 100  # how many points to sample from distribution

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


def data():
    samples = 20000
    dataa = [np.load(f"new_data/drawn_points/one_sample{i + 1}.npy") for i in range(samples)]
    data_in = np.array(dataa[:samples // 2])
    data_in_valid = np.array(dataa[samples // 2:])

    targets1=[]
    with h5py.File('/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/drawn_states/states.h5', 'r') as f:
        for i in range(samples):
            dataset_name = f'states_{i}'
            if dataset_name in f:
                data = f[dataset_name][()]
                targets1.append(np.concatenate((data.real.flatten(), data.imag.flatten())))

    data_out_all = np.array(targets1)
    data_out_valid = data_out_all[samples // 2:, :]
    data_out = data_out_all[:samples // 2, :]

    return data_in, data_out, data_in_valid, data_out_valid


def model(data_in, data_out, data_in_valid, data_out_valid):
    def custom_loss(y_true, y_pred):  # MY CUSTOM LOSS FUNCTION - same as in Talitha's approach
        input_shape = tf.shape(y_pred)

        trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

        trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
        matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))  # turn vectors into matrices
        matrix_com = tf.complex(matrix_form[:, 0, :, :],
                                matrix_form[:, 1, :, :])  # connect matrices into complex matrices
        transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)  # complex conjugate of matrices
        result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)  # M.dag() * M
        final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))  # previous / trace - normalisation

        finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
                                      tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)  # turning
        # it back into the vector

        return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)  # MSE calculation

    d = 4  # beginning dim
    traj_length = 10  # how many "snapshots" in time we take
    nof_samples_distr = 100  # how many points to sample from distribution

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}, input_shape=(nof_samples_distr * traj_length,),
                                    activation={{choice(['relu', 'sigmoid'])}}))
    model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}},
                                    activation={{choice(['relu', 'sigmoid'])}}))

    if {{choice(['two', 'three'])}} == 'three':
        model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}},
                                        activation={{choice(['relu', 'sigmoid'])}}))

    model.add(tf.keras.layers.Dense(2 * d ** 2, activation='tanh'))

    adam = tf.keras.optimizers.Adam(learning_rate={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd = tf.keras.optimizers.SGD(learning_rate={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})

    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model.compile(loss=custom_loss, optimizer=optim)

    model.fit(data_in, data_out,
              batch_size={{choice([128, 256, 512])}},
              epochs=1000,
              verbose=1,
              validation_data=(data_in_valid, data_out_valid))

    print(model.metrics_names)
    score = model.evaluate(data_in_valid, data_out_valid, verbose=0)
    return {'loss': score, 'status': STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=50,
                                      trials=Trials())

data_in, data_out, data_in_valid, data_out_valid = data()
print(best_run)

print("Evaluation of best performing model:")
print(best_model.evaluate(data_in_valid, data_out_valid))
