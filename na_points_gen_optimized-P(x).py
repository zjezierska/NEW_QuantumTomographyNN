import numpy as np
import qutip as qt
import time
from scipy.interpolate import interp1d
import math
from scipy.integrate import simps
from scipy.special import hermite

# NUMBER PARAMETERS
D = 5  # possible evolution dims
d = 4

traj_length = 200  # how many "snapshots" in time we take
nof_samples_distr = 200 # how many points to sample from distribution
tlist = np.linspace(0, 2 * np.pi, traj_length)  # we're taking traj_length points from one period of motion - 2 pi
xmax = 5
x_list = np.linspace(-xmax, xmax, nof_samples_distr)  # setting up the x space for P(x) and for wigner
# np.random.seed(63838)  # seed for reproductability

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


def draw_from_prob(xlist, prob_list, m):
    """
        Draw `m` samples from a probability distribution defined by `xlist` and `prob_list`.

        Parameters
        ----------
        xlist : ndarray
            A 1D array of x values.
        prob_list : ndarray
            A 1D array of probability values corresponding to the `xlist`.
        m : int
            The number of samples to draw.

        Returns
        -------
        draw_samples : ndarray
            A 1D array of `m` samples drawn from the probability distribution.
        """

    norm_constant = simps(prob_list, xlist)  # normalization
    my_pdfs = prob_list / norm_constant  # generate PDF
    my_cdf = np.cumsum(my_pdfs)  # generate CDF
    my_cdf = my_cdf / my_cdf[-1]
    func_ppf = interp1d(my_cdf, xlist, fill_value='extrapolate')  # generate the inverse CDF
    draw_samples = func_ppf(np.random.uniform(size=m))  # generate M samples

    return draw_samples


# TO ZAMIENIONE
def Psi_Psi_product(xlist, dim):
    """
    This function calculates the product of the wave functions psi_m and psi_n for a given list of x.

    Parameters:
    xlist (ndarray): List of x values
    dim (int): Dimension of the product matrix

    Returns:
    product_matrix (ndarray): Matrix with [m,n] element as product psi_m * psi_n"""
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


def fancy_data_gen(nof_samples, h, d, big_d):
    targets1 = [[] for _ in range(nof_samples)]

    for i in range(nof_samples):
        init_state = qt.rand_dm_ginibre(d)  # initial state is d-dimensional
        full_array = np.zeros((big_d, big_d), dtype=np.complex128)
        t_new = full_array[:d, :d] = init_state.full()  # this transforms the initial d-dimensional state into D-dim
        full_state = qt.Qobj(full_array)

        targets1[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))

        result = qt.mesolve(h, full_state, tlist, c_ops=c_ops)  # evolving the state

        trajectory = []
        for j in range(traj_length):
            rho_matrix = result.states[j].full()  # we're taking each state in snapshots of time
            last_column_sum = np.sum(rho_matrix[:, -1])
            last_row_sum = np.sum(rho_matrix[-1, :])
            if last_column_sum != 0 or last_row_sum != 0:
                raise ValueError("Error: The sum of the last column or last row is not zero!")

            before_sum = rho_matrix[:, :, np.newaxis] * psi_products
            first_sum = np.sum(before_sum.real, axis=0)
            prob_list = np.sum(first_sum, axis=0)  # P(x) = sum ( rho * psi_products ) by definition

            #pointz = draw_from_prob(x_list, prob_list, nof_samples_distr)  # drawing points from P(x)
            print(f"GENERATING {j + 1} POINTS FOR {i + 1}. SAMPLE")
            trajectory.append(list(prob_list))

        np.save(
            f"/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/drawn_points/one_sample_{i}.npy",
            trajectory)

    targets = np.array(targets1) # saving all the states together, not by one sample
    np.save('/Users/zuzannajezierska/Desktop/Machine Learning/New_approach/new_data/states.npy', targets)        


fancy_data_gen(20000, H_harmonic, d, D)  # calling the function
print(f"--- {time.time() - beginning} seconds ---")  # how much time did it take
