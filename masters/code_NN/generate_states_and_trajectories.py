import numpy as np
import qutip as qt
import time
from scipy.interpolate import interp1d
import math
from scipy.integrate import simps
from scipy.special import hermite

# Constants
D = 40  # Evolution dimensions
d = 4  # State dimension
TRAJ_LENGTH = 20  # Time snapshots
NOF_SAMPLES_DISTR = 20  # Points to sample from distribution
T_LIST = np.linspace(0, 2 * np.pi, TRAJ_LENGTH)
X_MAX = 5
X_LIST = np.linspace(-X_MAX, X_MAX, NOF_SAMPLES_DISTR)
np.random.seed(21370)  # For reproducibility
ALPHA = 5  # Inverse quarticity
GAMMA = 0.01  # Decoherence rate

# Quantum Definitions
a = qt.destroy(D)
x = a.dag() + a
p = 1j * (a.dag() - a)
H_QUARTIC = p**2 / 4 + (x / ALPHA)**4
H_HARMONIC = a.dag() * a
c_ops = [np.sqrt(GAMMA) * x]  # Decoherence operators

def draw_samples_from_distribution(xlist, prob_list, m):
    """Draw samples from a given probability distribution."""
    norm_constant = simps(prob_list, xlist)
    pdf = prob_list / norm_constant
    cdf = np.cumsum(pdf) / np.cumsum(pdf)[-1]
    inverse_cdf = interp1d(cdf, xlist, fill_value='extrapolate')
    return inverse_cdf(np.random.uniform(size=m))


def calculate_psi_products(xlist, dim):
    """Calculate the product of wave functions for a given x list and dimension."""
    exp_list = np.exp(-xlist**2 / 2.0)
    norm_list = [np.pi**(-0.25) / np.sqrt(2.0**m * math.factorial(m)) for m in range(dim)]
    herm_list = [np.polyval(hermite(m), xlist) for m in range(dim)]
    product_matrix = np.array([
        exp_list * norm_list[m] * herm_list[m] * exp_list * norm_list[n] * herm_list[n]
        for m in range(dim) for n in range(dim)
    ]).reshape((dim, dim, -1))
    return product_matrix


def generate_quantum_state_trajectories(n_samples, h, dim, big_dim):
    """Generate quantum state trajectories and save the data."""
    psi_products = calculate_psi_products(X_LIST, D)
    targets = []
    trajectories = []

    for i in range(n_samples):
        initial_state = qt.rand_dm_ginibre(dim)
        full_array = np.zeros((big_dim, big_dim), dtype=np.complex128)
        full_array[:dim, :dim] = initial_state.full()
        full_state = qt.Qobj(full_array)
        targets.append(np.concatenate((initial_state.full().real.flatten(), initial_state.full().imag.flatten())))
        
        trajectory = evolve_state_and_extract_trajectory(h, full_state, psi_products)

        # append it to the samples list
        trajectories.append(np.array(trajectory))

    np.save('new_data/trajectories.npy', np.array(trajectories))

    np.save('new_data/states.npy', np.array(targets))


def evolve_state_and_extract_trajectory(h, initial_state, psi_products):
    """Evolve a given initial state and extract its trajectory."""
    result = qt.mesolve(h, initial_state, T_LIST, c_ops=c_ops)
    trajectory = []

    for j, state in enumerate(result.states):
        rho_matrix = state.full()
        validate_state(rho_matrix)
        prob_list = calculate_probabilities(rho_matrix, psi_products)
        trajectory.append(prob_list)
    
    return trajectory


def validate_state(rho_matrix):
    """Ensure the state's last column and row sums are within acceptable limits."""
    if rho_matrix[:, -1].sum() > 1e-5 or rho_matrix[-1, :].sum() > 1e-5:
        raise ValueError("Sum of the last column or row is not negligible.")


def calculate_probabilities(rho_matrix, psi_products):
    """Calculate probabilities based on the state and psi products."""
    interaction = rho_matrix[:, :, np.newaxis] * psi_products
    return list(np.sum(np.sum(interaction.real, axis=0), axis=0))


# Execution
start_time = time.time()
generate_quantum_state_trajectories(56, H_HARMONIC, d, D)  # Reduced the number for quicker testing
print(f"--- {time.time() - start_time} seconds ---")