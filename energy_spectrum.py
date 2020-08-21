from hamiltonian import * 

L = 3
particle_list = [1]
phi = 0
t = 1
H_single_particle = Hamiltonian(t, phi, L, particle_list)
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
