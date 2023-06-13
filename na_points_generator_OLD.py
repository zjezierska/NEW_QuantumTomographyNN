import tensorflow as tf
import numpy as np
import qutip as qt
# import gc
import time
import math
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import hermite

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# NUMBER PARAMETERS
D = 40  # possible evolution dims
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate

# QUANTUM DEFINITIONS
a = qt.destroy(D)  # annihilation operator
x = a.dag() + a
p = 1j * (a.dag() - a)
H_quartic = p * p / 4 + (x / alpha) * (x / alpha) * (x / alpha) * (x / alpha)
H_harmonic = a.dag() * a
# # A list of collapse operators - later
c_ops = [np.sqrt(gamma) * x]  # decoherence

beginning = time.time()  # testing the calculation time

# setting up the x space for P(x) and for wigner
xmax = 5
x_list = np.linspace(-xmax, xmax, 200)

traj_length = 20  # how many "snapshots" in time we take
nof_samples_distr = 1000  # how many points to sample from distribution
num_bin = 20  # number of histogram bins


# np.random.seed(63838)  # seed for reproductability


def draw_from_prob(xlist, prob_list, m):  # function to draw M points from P(x)
    norm_constant = simps(prob_list, xlist)  # normalization
    my_pdfs = prob_list / norm_constant  # generate PDF
    my_cdf = np.cumsum(my_pdfs)  # generate CDF
    my_cdf = my_cdf / my_cdf[-1]
    func_ppf = interp1d(my_cdf, xlist, fill_value='extrapolate')  # generate the inverse CDF
    draw_samples = func_ppf(np.random.uniform(size=m))  # generate M samples
    return draw_samples


def Psi_Psi_product(xlist, dim):  # function that returns products psi_m*psi_n for a given list of x
    product_list = []

    exp_list = np.exp(-xlist ** 2 / 2.0)
    norm_list = [np.pi ** (-0.25) / math.sqrt(2.0 ** m * math.factorial(m)) for m in range(dim)]
    herm_list = [np.polyval(hermite(m), xlist) for m in range(dim)]

    for m in range(dim):
        psi_m = exp_list * norm_list[m] * herm_list[m]  # wave function psi_m of a harmonic oscillator
        for ni in range(dim):
            psi_n = exp_list * norm_list[ni] * herm_list[ni]  # wave function psi_n of a harmonic oscillator
            product_list.append(psi_m * psi_n)  # in general should be np.conjugate(psi_n)

    product_matrix = np.reshape(product_list,
                                (dim, dim, -1))  # reshape into matrix, such that [m,n] element is product psi_m*psi_n
    return product_matrix


psi_products = Psi_Psi_product(x_list, D)

tlist = np.linspace(0, 2 * np.pi, traj_length)  # we're taking traj_length points from one period of motion - 2 pi


def fancy_data_gen(nof_samples, h, d, big_d):
    # targets1 = [[] for _ in range(nof_samples)]

    for i in range(nof_samples):
        init_state = qt.rand_dm_ginibre(d)  # initial state is d-dimensional
        full_array = np.full([big_d, big_d], 0. + 0.j)
        t_new = full_array[0:d, 0:d] = init_state.full()  # this transforms the intial d-dimensional state into D-dim
        full_state = qt.Qobj(full_array)
        # targets1[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))
        np.save(f'/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/drawn_states/states{i}.npy',
                np.concatenate((t_new.real.flatten(), t_new.imag.flatten())))

        result = qt.mesolve(h, full_state, tlist, c_ops=c_ops)  # evolving the state

        samples = []
        for j in range(traj_length):
            rho = result.states[j]
            rho_matrix = rho.full()  # we're taking each state in snapshots of time

            before_sum = tf.math.multiply(rho_matrix[:, :, tf.newaxis], psi_products)
            first_sum = tf.math.reduce_sum(tf.math.real(before_sum), axis=0)
            prob_list = tf.math.reduce_sum(first_sum, axis=0)  # P(x) = sum ( rho * psi_products ) by definition
            sample = draw_from_prob(x_list, prob_list, nof_samples_distr)  # drawing points from P(x)

            # (heights, bins) = np.histogram(sample, num_bin, density=True)  # getting heights from the histogram of samples
            # if i % 100 == 0 and j == 0:
            print(
                f"GENERATING {j + 1} POINTS FOR {i + 1}. SAMPLE")  # saving only points from P(x), not the histogram ( I can create it in the machine code more flexibly)
            samples.append(list(sample))

            # del rho  # deleting variables from memory
            # del rho_matrix
            # del before_sum
            # del first_sum
            # del prob_list
            # del sample
            # gc.collect()

        np.save(
            f"/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/drawn_points/one_sample{i + 1}.npy",
            [element for sublist in samples for element in sublist])
        # del samples
        # del result
        # gc.collect()

    # targets = np.array(targets1) # saving all the states together, not by one sample
    # np.save('/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/states.npy', targets)


fancy_data_gen(20000, H_harmonic, 4, 40)  # calling the function
print(f"--- {time.time() - beginning} seconds ---")  # how much time did it take
