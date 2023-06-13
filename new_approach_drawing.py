import tensorflow as tf
import math
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import hermite
from parameters_talitha import *

d = 4


def Psi_Psi_product(x_list, dim):  # function that returns products psi_m*psi_n for a given list of x and a state matrix
    # (dim)x(dim)
    product_list = []

    exp_list = np.exp(-x_list ** 2 / 2.0)
    norm_list = [np.pi ** (-0.25) / math.sqrt(2.0 ** m * math.factorial(m)) for m in range(dim)]
    herm_list = [np.polyval(hermite(m), x_list) for m in range(dim)]

    for m in range(dim):
        psi_m = exp_list * norm_list[m] * herm_list[m]  # wave function psi_m of a harmonic oscillator
        for ni in range(dim):
            psi_n = exp_list * norm_list[ni] * herm_list[ni]  # wave function psi_n of a harmonic oscillator
            product_list.append(psi_m * psi_n)  # in general should be np.conjugate(psi_n)

    product_matrix = np.reshape(product_list,
                                (dim, dim, -1))  # reshape into matrix, such that [m,n] element is product psi_m*psi_n
    return product_matrix


def draw_from_prob(x_list, prob_list, m):  # function to draw M points from P(x)
    norm_constant = simps(prob_list, x_list)  # normalization
    my_pdfs = prob_list / norm_constant  # generate PDF
    my_cdf = np.cumsum(my_pdfs)  # generate CDF
    my_cdf = my_cdf / my_cdf[-1]
    func_ppf = interp1d(my_cdf, x_list, fill_value='extrapolate')  # generate the inverse CDF
    draw_samples = func_ppf(np.random.uniform(size=m))  # generate M samples
    return draw_samples


xvec = np.linspace(-5, 5, 200)  # from how many points we extrapolate the distribution
psi_products = Psi_Psi_product(xvec, d)  # creating the matrix psi_m * psi_n (x)

tlist = np.linspace(0, 2 * np.pi, 20)  # we're taking samples from 20 points in one period

init_state = qt.rand_dm_ginibre(d)  # initial state is d-dimensional

bin_heights = []
rho_matrix = init_state.full()

before_sum = tf.math.multiply(rho_matrix[:, :, tf.newaxis], psi_products)
first_sum = tf.math.reduce_sum(tf.math.real(before_sum), axis=0)
prob_list = tf.math.reduce_sum(first_sum, axis=0)  # P(x) = sum ( rho * psi_products ) by definition

sample = draw_from_prob(xvec, prob_list, 1000)  # create P(x) from xvec, sample M points from this

W1 = qt.wigner(init_state, xvec, xvec)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.contourf(xvec, xvec, W1, 100, cmap='RdBu_r')  # drawing the wigner function of the state
ax2.plot(xvec, prob_list)  # drawing P(x)
heights, bins, patches = ax2.hist(sample, bins=40, density=True)  # drawing the histogram of drawn points
plt.show()
