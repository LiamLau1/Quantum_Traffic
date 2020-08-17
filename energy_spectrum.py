from numba import jit
from qring.algorithms import *
import numpy as np
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt

"""
Code to find the energy spectrum for a 1D Quantum ring with L sites and flux \Phi through the ring. We have restricted the on site occupation to either 0 or 1, so having the number of basis states grows as 2^L
Author: Liam L.H. Lau
"""

#@jit(nopython = True)
def Hamiltonian(A_list, A_dagger_list, t, phi, L, particle_list):
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
    H = Hamiltonian(A_list, A_dagger_list, t, phi, L, particle_list)
    eigens = linalg.eig(H)
    ground_state = eigens[1][:,np.argmin(eigens[0])]
    return numerical_current(ground_state, i, t, phi, L, particle_list, A_dagger_list, A_list)

L = 3
t = 1
phi = 0
# PARTICLE NUMBER
particle_list = [1]
global A_list
global A_dagger_list
A_list = [matrix_representation(a, n+1, L) for n in range(L)]
A_dagger_list = [matrix_representation(a_dagger, n+1, L) for n in range(L)]
particle_index_j = list(find_particle_indices(particle_list,L))
particle_index_i = [[i] for i in particle_index_j]

"""
phi_list = np.linspace(0,3,5000)
ground_current_list_1 = [ground_state_current(0,t,phi_value,L,particle_list) for phi_value in phi_list]
ground_current_list_2 = [ground_state_current(1,t,phi_value,L,particle_list) for phi_value in phi_list]
ground_current_list_3 = [ground_state_current(2,t,phi_value,L,particle_list) for phi_value in phi_list]
fig = plt.figure()
plt.plot(phi_list,ground_current_list_1, label=r"$1 \rightarrow 2$")
plt.plot(phi_list,ground_current_list_2, label = r"$2 \rightarrow 3$")
plt.plot(phi_list,ground_current_list_3, label = r"$3 \rightarrow 1$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$\phi = \frac{\Phi}{\Phi_0}$')
plt.ylabel('Probability Current $J$')
plt.legend(bbox_to_anchor=(0.82, 1))
fig.tight_layout()
plt.savefig('./plots/groundstate_current_3.pdf')


phi = 0.49999999
t_list = np.linspace(0,5,5000)
ground_current_list_1 = [ground_state_current(0,t_value,phi,L,particle_list) for t_value in t_list]
ground_current_list_2 = [ground_state_current(1,t_value,phi,L,particle_list) for t_value in t_list]
ground_current_list_3 = [ground_state_current(2,t_value,phi,L,particle_list) for t_value in t_list]
fig = plt.figure()
plt.plot(t_list,ground_current_list_1, label=r"$1 \rightarrow 2$")
plt.plot(t_list,ground_current_list_2, label = r"$2 \rightarrow 3$")
plt.plot(t_list,ground_current_list_3, label = r"$3 \rightarrow 1$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$t$')
plt.ylabel('Probability Current $J$')
#plt.legend(bbox_to_anchor=(0.82, 1))
plt.legend()
#fig.tight_layout()
plt.savefig('./plots/groundstate_current_hopping_3.pdf')
slope, intercept = np.polyfit((t_list), (ground_current_list_1), 1)
"""
H_single_particle = Hamiltonian(A_list, A_dagger_list, t, phi, L, particle_list)
#lattice_spacing = 2 * np.pi *R / L
# Chose R in such a way to make lattice spacing unity
lattice_spacing = 1
eigens = linalg.eig(H_single_particle, left=True, right=False)
energy_1 = eigens[0][0].real
vector_1 = eigens[1][:,0]
energy_2 = eigens[0][1].real
vector_2 = eigens[1][:,1]
energy_3 = eigens[0][2].real
vector_3 = eigens[1][:,2]
ground_state_energy = eigens[0][np.argmin(eigens[0])].real
ground_state = eigens[1][:,np.argmin(eigens[0])]

ka = np.linspace(-np.pi/lattice_spacing, np.pi/lattice_spacing, 1000) * lattice_spacing
E_true  = true_dispersion(ka ,t,phi,L)
E_low_k = low_k_dispersion(ka, t,phi,L)

J = analytic_current(ka, t, phi,L)
plt.plot(ka, E_true)
plt.plot(ka, E_low_k, ':')
plt.plot(ka, J, ':')
plt.plot(-2*np.pi/3,ground_state_energy, '+' , markersize = 13)
#plt.plot(-2*np.pi/3,energy_1, '+' , markersize = 13)
#plt.plot(2*np.pi/3,energy_3, '+' , markersize = 13)
#plt.title('Dispersion Relation for Quantum Ring with 1 Particle')
#plt.xlabel('ka')
#plt.ylabel('E')
plt.show()
#plt.savefig('./plots/single_dispersion_3.pdf')

