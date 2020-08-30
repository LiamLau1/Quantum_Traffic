from numba import complex128, float64, int32, jit, njit
from qring.algorithms import *
import numpy as np
from scipy import sparse
from scipy import linalg
from scipy.integrate import odeint
from scipy.integrate import complex_ode
from scipy.integrate import solve_ivp
from scipy import signal
import matplotlib.pyplot as plt

"""
Code to find the energy spectrum for a 1D Quantum ring with L sites and flux \Phi through the ring. We have restricted the on site occupation to either 0 or 1, so having the number of basis states grows as 2^L
Author: Liam L.H. Lau
"""

#@jit(nopython = True)
def Hamiltonian(t, phi, L, particle_list):
    A_list = [matrix_representation(a, n+1, L) for n in range(L)]
    A_dagger_list = [matrix_representation(a_dagger, n+1, L) for n in range(L)]
    particle_index_i = list(find_particle_indices(particle_list,L))
    particle_index_j = [[j] for j in particle_index_i]
    # Define Ahranov Bohm discrete phase changes for hopping
    phase1 = np.exp(-1j * 2 * np.pi * phi / L)
    phase2 = np.exp(1j * 2 * np.pi * phi / L)
    H = np.zeros((2**L, 2**L))
    for i in range(L):
        H += -t *(phase1 * A_dagger_list[np.mod(i+1, L)] @ A_list[i] + phase2 * A_dagger_list[i] @ A_list[np.mod(i+1,L)])

    return H[particle_index_i,particle_index_j]

def ground_state_current(i, t, phi, L, particle_list):
    """i runs from 0"""
    H = Hamiltonian(t, phi, L, particle_list)
    eigens = linalg.eig(H)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    return numerical_current(ground_state, i, t, phi, L, particle_list)

def ground_state_density(i, t, phi, L, particle_list):
    """i runs from 0"""
    H = Hamiltonian(t, phi, L, particle_list)
    eigens = linalg.eig(H)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    return numerical_density(ground_state, i, t, phi, L, particle_list)

def Hamiltonian_perturbed(k, t, phi, L, particle_list):
    """ Perturbed Hamiltonian with strong potential at site k """
    A_list = [matrix_representation(a, n+1, L) for n in range(L)]
    A_dagger_list = [matrix_representation(a_dagger, n+1, L) for n in range(L)]
    particle_index_i = list(find_particle_indices(particle_list,L))
    particle_index_j = [[j] for j in particle_index_i]
    # Define Ahranov Bohm discrete phase changes for hopping
    phase1 = np.exp(-1j * 2 * np.pi * phi / L)
    phase2 = np.exp(1j * 2 * np.pi * phi / L)
    H = np.zeros((2**L, 2**L))
    for i in range(L):
        H += -t *(phase1 * A_dagger_list[np.mod(i+1, L)] @ A_list[i] + phase2 * A_dagger_list[i] @ A_list[np.mod(i+1,L)]) 
        if i == k-1 :
           H+= 100000 * A_dagger_list[i]@ A_list[i]
    return H[particle_index_i,particle_index_j]

#@jit(complex128[:,:](float64, int32,float64, float64, int32, int32[:,]), nopython = False)
def gradual_Hamiltonian_perturbed(time, k, t, phi, L, particle_list):
    """ Perturbed Hamiltonian with strong potential at site k that is gradually turned on and off"""
    A_list = [matrix_representation(a, n+1, L) for n in range(L)]
    A_dagger_list = [matrix_representation(a_dagger, n+1, L) for n in range(L)]
    particle_index_i = list(find_particle_indices(particle_list,L))
    particle_index_j = [[j] for j in particle_index_i]
    # Define Ahranov Bohm discrete phase changes for hopping
    phase1 = np.exp(-1j * 2 * np.pi * phi / L)
    phase2 = np.exp(1j * 2 * np.pi * phi / L)
    H = np.zeros((2**L, 2**L))
    for i in range(L):
        H += -t *(phase1 * A_dagger_list[np.mod(i+1, L)] @ A_list[i] + phase2 * A_dagger_list[i] @ A_list[np.mod(i+1,L)]) 
        if i == k-1 :
           #H+= 100000* np.exp(-(time)**2) * A_dagger_list[i]@ A_list[i]
           H+= 10* np.exp(-(time)**2) * A_dagger_list[i]@ A_list[i]
    return H[particle_index_i,particle_index_j]

def current_quantum_quench(time, site, perturbed_site, t, phi, L, particle_list):
    # Act potential on site 1
    H_perturbed = Hamiltonian_perturbed(perturbed_site,t,phi,L,particle_list)
    # Pick the ground state
    eigens = linalg.eig(H_perturbed)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    # Change Hamiltonian
    H = Hamiltonian(t, phi, L, particle_list)
    # Define time evolution operator
    U = linalg.expm(-(1j)*time*H)
    ground_state_time =  U @ ground_state
    return numerical_current(ground_state_time, site, t, phi, L, particle_list)


#@jit(complex128[:,](float64, complex128[:,], float64, float64, int32, int32[:, ]), nopython = False)
def tdse(time,psi,  t, phi, L, particle_list):
    H = gradual_Hamiltonian_perturbed(time,1, t, phi, L,particle_list)
    dpsi_dt = -(1j) * H @ psi
    return np.reshape(np.array(dpsi_dt).T, (psi.shape[0],))
    #return np.array(dpsi_dt).T

def current_gradual_perturbation(time, site, perturbed_site, t, phi, L, particle_list):
    # Act potential on site 1
    H_unperturbed = gradual_Hamiltonian_perturbed(-4,perturbed_site,t,phi,L,particle_list) 
    H_perturbed = gradual_Hamiltonian_perturbed(time,perturbed_site,t,phi,L,particle_list)
    # Pick the ground state
    eigens = linalg.eig(H_unperturbed)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    # Hamiltonian is now time dependent, so need to solve set of linear DEs.
    ground_state_time = solve_ivp(tdse, (-4,time), ground_state,method = 'BDF',  args = (t, phi, L, particle_list)).y[:,-1]
    return numerical_current(ground_state_time, site, t, phi, L, particle_list)

def density_gradual_perturbation(time, initial_state_time, site, t, phi, L, particle_list):
    return numerical_density(initial_state_time, site, t, phi, L, particle_list)

def density_quantum_quench(time, site, perturbed_site, t, phi, L, particle_list):
    # Act potential on site 1
    H_perturbed = Hamiltonian_perturbed(perturbed_site,t,phi,L,particle_list)
    # Pick the ground state
    eigens = linalg.eig(H_perturbed)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    # Change Hamiltonian
    H = Hamiltonian(t, phi, L, particle_list)
    # Define time evolution operator
    U = linalg.expm(-(1j)*time*H)
    ground_state_time =  U @ ground_state
    return numerical_density(ground_state_time, site, t, phi, L, particle_list)


def occupation_current_plot(Lmax):
    #Function to plot the ground state probability current for different L and occupation
    L_list = range(2, Lmax+1)
    for L in L_list:
        particle_list = [[x] for x in range(1,L+1)]
        currents = [ground_state_current(0,t,phi,L,particle_number) for particle_number in particle_list]
        plt.plot(particle_list, currents, marker="+", label = "{} total sites".format(L))
    plt.legend()
    plt.xlabel('Particle Number')
    plt.xticks(ticks = range(1, Lmax+1))
    plt.ylabel('Maximum Probability Current $J$')
    plt.title('Plot of ground-state current varying number of bosons')
    plt.savefig('./plots/groundstate_current_occupation.pdf')


