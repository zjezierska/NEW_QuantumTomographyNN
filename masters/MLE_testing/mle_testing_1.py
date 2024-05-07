import numpy as np
import qutip as qt
import time
from scipy.interpolate import interp1d
import math
from scipy.special import hermite

a = np.load("/Users/zuzannajezierska/Desktop/Studia/masters/code_NN/new_data/states.npy")
b = np.load("/Users/zuzannajezierska/Desktop/Studia/masters/code_NN/new_data/trajectories.npy")

d = 4
iters = 10

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


def calc_rho(dims, num_iters, theta_data, quadrature_data, visualize_steps):
    
    likelihood_trend = [] # will be used to keep track of the likelihood of rho at each step
    
    n_list = np.linspace(0, dims - 1, dims, dtype = int) # creates an integer list of the values of n (Fock states) we are considering

    rho = np.zeros((dims, dims), complex) # initialize the correct dimension denisty matrix
    np.fill_diagonal(rho, 1) # fill the diagonals with ones
    rho = rho/sum(rho.diagonal()) # normalize the trace of the density matrix

    ''' The below matricies allow us to do the evaluation of the n wavefunctions at every data point (theta, x) in a vectorized fashion '''
    N = np.array(np.split(np.repeat(n_list, len(quadrature_data)), dims)) # A matrix that repeats the Fock state list in each column
    THETA = np.tile(theta_data, (dims, 1)) # A matrix that repeats the theta data list in each column
    X = np.tile(quadrature_data, (dims, 1)) # A matrix that repeats the quadrature data list in each row

    ################################################### Scipy Variable and Function Declarations ######################################################

    x = Symbol('x') # declare variable 'x' signifying the quadrature value
    theta = Symbol('\\theta') # declare variable 'theta' signifying the value of the phase
    n = Symbol('n', integer=True) # declare variable 'n' signifying the Fock states
    
    # Note: The coefs need to be done separately from the functions since it was giving me trouble at one point...
    coefs = (1/np.pi)**(1/4)/sqrt(2**n*factorial(n)) # the sympy function for the coefficients
    g = lambdify(n, coefs)(N) # evaluate the coefficents at the values of N

    func = exp(1j*n*(theta - np.pi/2))*exp(-x**2/2)*hermite(n, x) # the sympy function for the coefficients
    f = lambdify((n, theta, x), func)(N, THETA, X) # evaluate the coefficents at the values of N
    
    ''' Debug ''' 
    # print('The Wavefunction Equation:')
    # display(coefs*func)
    # print()
    
    ####################################################### Generate Useful Matricies ##########################################################
    
    psi_matrix = (f*g).transpose() # evaluation of the wavefunction at each data point (theta, x) for every consiered n - rows are (theta, x), columns are n 
    psi_squared_matrix = np.einsum('bi,bo->boi', psi_matrix, psi_matrix.conj()) # complex outer product
    likelihood = np.real(np.sum(np.log(np.sum(np.sum(psi_squared_matrix*np.real(rho), axis = 1), axis = 1)))) # calculate the likelihood of the current rho
    likelihood_trend.append(likelihood) # add likelihood to trend list
    
    ''' Debug '''
    # print('Psi Matrix is:\n', psi_matrix.transpose())
    # print('Psi Squared Matrix is:\n', psi_squared_matrix)
    
    # print('Initial Density Matrix:\n', rho, '\n')
    
    # plt.bar(n_list, np.diagonal(rho))
    # plt.show()

    # print('Likelihood For Current Rho:', likelihood)