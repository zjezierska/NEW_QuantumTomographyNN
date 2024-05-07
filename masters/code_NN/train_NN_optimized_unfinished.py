import numpy as np
import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
TRAJ_LENGTH = 200
NOF_SAMPLES_DISTR = 200
CHECKING_STEP = 10
EPOCHS = 10000
PATIENCE = 500
BATCH_SIZE = 512
D = 4  # Dimension
SAMPLES = 20000
HALF_SAMPLES = SAMPLES // 2

# Paths
DATA_PATH = "/net/ascratch/people/plgredhoodies/data_harmonic/"
MODEL_PATH = "/net/people/plgrid/plgredhoodies/machine_scripts/models_harmonic_200_10/best_model_draw.h5"
TRAINING_DATA_PATH = "/net/people/plgrid/plgredhoodies/machine_scripts/training_data_harmonic/"

# def custom_loss(y_true, y_pred):
#     input_shape = tf.shape(y_pred)

#     trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

#     trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
#     matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))  # turn vectors into matrices
#     matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])  # connect matrices into complex matrices

#     transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)  # complex conjugate of matrices
#     result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)  # M.dag() * M
#     final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))  # previous / trace - normalisation

#     finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
#                                   tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)  # turning
#     # it back into the vector

#     return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)  # MSE calculation


def custom_loss(y_true, y_pred):
    input_shape = tf.shape(y_pred)
    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)
    trace_reshaped = tf.reshape(trace, [input_shape[0], 1, 1])

    # Reshape y_pred for complex matrix operations
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])

    # Compute the conjugate transpose and the matrix multiplication
    transpose_matrix = tf.linalg.adjoint(matrix_com)
    product_matrix = tf.matmul(transpose_matrix, matrix_com)

    # Normalize and flatten the result
    normalized_matrix = product_matrix / tf.cast(trace_reshaped, tf.complex64)
    flattened_real_imag = tf.concat([tf.math.real(normalized_matrix), tf.math.imag(normalized_matrix)], axis=-1)
    flattened_real_imag = tf.reshape(flattened_real_imag, (input_shape[0], -1))

    return tf.reduce_mean(tf.square(flattened_real_imag - y_true), axis=-1)


def initialize_network(input_shape):
    net = Sequential([
        Dense(512, input_shape=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dense(2 * (D ** 2), activation='tanh')
    ])
    net.compile(loss=custom_loss, optimizer=Adam())
    return net


def create_points_samples(num_samples):
    samplez = []
    # Iterate over the number of samples
    for i in range(num_samples):
        # Load trajectory data from the file
        trajectory = os.path.join(DATA_PATH, f"one_sample_{i}.npy")

        # Convert the list of lists to a 2D NumPy array
        data_array = np.array(trajectory)

        # Flatten the trajectory and append it to the samples list
        samplez.append(data_array)

    return samplez


def train_network(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='min'),
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
    ]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
    return history

def evaluate_model(model, X_val, y_val):
    # Logic to evaluate the model
    pass

def plot_results(infidelities):
    plt.figure(figsize=(8, 6))
    plt.imshow(infidelities, origin='lower', cmap='viridis', aspect='auto', norm=colors.LogNorm())
    plt.colorbar(label='Infidelity')
    plt.xlabel('Number of points in P(x)')
    plt.ylabel('Number of points in time')
    plt.savefig(f"{TRAINING_DATA_PATH}punkty_plot.png")

def main():
    # Split data loading process
    # Load state data
    y_states1 = np.load(os.path.join(DATA_PATH, 'states.npy'))
    y_states = y_states1[:HALF_SAMPLES, :]
    y_states_valid = y_states1[HALF_SAMPLES:, :]

    # Create and split samples
    y_bins1 = np.load(os.path.join(DATA_PATH, 'trajectories.npy'))
    y_bins = y_bins1[:HALF_SAMPLES, :]
    y_bins_valid = y_bins1[HALF_SAMPLES:, :]

    # Initialize model
    model = initialize_network((NOF_SAMPLES_DISTR * TRAJ_LENGTH,))

    # Train model
    history = train_network(model, y_bins, y_states, y_bins_valid, y_states_valid, EPOCHS, BATCH_SIZE)

    # Evaluate model
    infidelities = evaluate_model(model, y_bins_valid, y_states_valid)

    # Plot results
    plot_results(infidelities)

if __name__ == "__main__":
    main()
