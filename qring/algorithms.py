import numpy as np
import itertools
from scipy import sparse
from scipy import linalg

# Define matrix representations of the local ladder operators
a = sparse.csr_matrix(np.matrix([[0,1],[0,0]]))
a_dagger = sparse.csr_matrix(np.matrix([[0,0],[1,0]]))

def matrix_representation(matrix, n, L):
    """Function to create the whole space matrix representation for an operator on the nth site from the local matrix representation by performing the tensor product with identities on the other sites and the local matrix at the nth site."""
    if n!= 1:
        matrix_temp = sparse.kron(sparse.eye(2**(n-1)), matrix)
        return sparse.kron(matrix_temp, sparse.eye(2**(L-n)))
    else:
        return sparse.kron(matrix, sparse.eye(2**(L-1)))

def expectation(state, operator):
    """Function which returns expectation value of operator given a state"""
    return np.conj(state) @ operator @ state

def true_dispersion(ka, t, phi, L):
    E = -2*t*np.cos(ka + 2*np.pi*phi/L)
    return E

def low_k_dispersion(ka, t,phi,L):
    E = -2*t + t*(ka + 2*np.pi*phi/L)**2
    return E

def analytic_current(ka, t, phi, L):
    J = 2*t*np.sin(ka + 2*np.pi*phi/L)
    return J

def numerical_current(state, i, t, phi, L, particle_list, A_dagger_list, A_list):
    """Defining the current from site i to i+1 to be positive"""
    if i in range(0, L+1):
        particle_index_i = list(find_particle_indices(particle_list,L))
        particle_index_j = [[j] for j in particle_index_i]
        #current = -2 * t * np.imag(np.exp(-1j*2*np.pi * phi/L) * expectation(state, ((matrix_representation(a_dagger, np.mod(i+1,L), L)@ matrix_representation(a, i, L)).todense())[particle_index_i, particle_index_j]))
        current = -2 * t * np.imag(np.exp(-1j*2*np.pi * phi/L) * expectation(state, (A_dagger_list[np.mod(i+1,L)] @ A_list[i]).todense()[particle_index_i, particle_index_j]))
    else:
        print("Please enter a site number from 1 to L")
    return np.asscalar(current)

def kron_rec(c,n):
    """Recursive function which returns the kronecker product of itself n times"""
    if n ==1:
        return c
    else:
        return np.kron(c, kron_rec(c, n-1))

def find_particle_indices(particle_list, L):
    """Function to find the indices to keep for certain number of particles"""
    a = np.array([1,0]) # No occupation
    b = np.array([0,1]) # One occupation
    particle_indices = []
    for i in particle_list:
        if i == 0:
            particle_indices.append(0)
        else:
            index_combinations = itertools.combinations(range(L), i)
            for combinations in index_combinations:
                for j in range(i):
                    if combinations[j] != 0 and j == 0:
                        K = np.kron(kron_rec(a, combinations[j]), b)
                    elif j ==0:
                        K = b

                    if combinations[j] != combinations[-1]:
                        if combinations[j+1] - combinations[j] > 1:
                            K = np.kron(K,kron_rec(a, combinations[j+1] - combinations[j] -1))
                            K = np.kron(K,b)
                        else:
                            K = np.kron(K,b)
                    elif combinations[j] != range(L)[-1]:
                        K = np.kron(K,kron_rec(a,L - combinations[j] -1))
                particle_indices.append(np.squeeze(np.nonzero(K)))
    return particle_indices
        
