from numba import jit
from qring.algorithms import *
import numpy as np
from scipy import sparse
from scipy import linalg

"""
Code to find the energy spectrum for a 1D Quantum ring with L sites and flux \Phi through the ring. We have restricted the on site occupation to either 0 or 1, so having the number of basis states grows as 2^L
Author: Liam L.H. Lau
"""

#@jit(nopython = True)
def Hamiltonian(A_list, A_dagger_list, t, phi, L):
    # Define Ahranov Bohm discrete phase changes for hopping
    phase1 = np.exp(-1j * 2 * np.pi * phi / L)
    phase2 = np.exp(1j * 2 * np.pi * phi / L)
    H = np.zeros((2**L, 2**L))
    for i in range(L):
        H += -t *(phase1 * A_dagger_list[np.mod(i+1, L)] @ A_list[i] + phase2 * A_dagger_list[i] @ A_list[np.mod(i+1,L)])

    return H

L = 3
t = 1
phi = 0.2
A_list = [matrix_representation(a, n+1, L) for n in range(L)]
A_dagger_list = [matrix_representation(a_dagger, n+1, L) for n in range(L)]
H = Hamiltonian(A_list, A_dagger_list, t, phi, L)
single_particle_index_i = [1,2,4]
single_particle_index_j = [[j] for j in single_particle_index_i]
H_single_particle = H[single_particle_index_i, single_particle_index_j]
#lattice_spacing = 2 * np.pi *R / L
# Chose R in such a way to make lattice spacing unity
lattice_spacing = 1
eigens = linalg.eig(H_single_particle)
energy_1 = eigens[0][0].real
vector_1 = eigens[1][0]
energy_2 = eigens[0][1].real
vector_2 = eigens[1][1]
energy_3 = eigens[0][2].real
vector_3 = eigens[1][2]

ka = np.linspace(-np.pi/lattice_spacing, np.pi/lattice_spacing, 1000) * lattice_spacing
E_true  = true_dispersion(ka ,t,phi,L)
E_low_k = low_k_dispersion(ka, t,phi,L)
J = current(ka, t, phi,L)
plt.plot(ka, E_true)
plt.plot(ka, E_low_k, ':')
#plt.plot(ka, J, ':')
plt.plot(0,energy_2, '+' , markersize = 13)
plt.plot(-2*np.pi/3,energy_1, '+' , markersize = 13)
plt.plot(2*np.pi/3,energy_3, '+' , markersize = 13)
plt.title('Dispersion Relation for Quantum Ring with 1 Particle')
plt.xlabel('ka')
plt.ylabel('E')
plt.show()
#plt.savefig('./plots/single_dispersion_3.pdf')

find_particle_indices([1], 5)
