from hamiltonian import *

L = 3
particle_list = [1]
phi = 0.2
t = 1

time_list = np.linspace(0,8,1000)
density_list_1 = [density_quantum_quench(time, 0, 1,t,phi,L,particle_list) for time in time_list]
density_list_2 = [density_quantum_quench(time, 1, 1, t,phi,L,particle_list) for time in time_list]
density_list_3 = [density_quantum_quench(time, 2, 1, t,phi,L,particle_list) for time in time_list]
plt.plot(time_list, density_list_1, label = 'density on site 1')
plt.plot(time_list, density_list_2, label = 'density on site 2')
plt.plot(time_list, density_list_3, label = 'density on site 3')
plt.legend()
plt.title('Plot of ground-state density')
plt.xlabel('time')
plt.ylabel('Site occupation')

f, Cxyf = signal.coherence(density_list_1, density_list_2,125)
g, Cxyg = signal.coherence(density_list_1, density_list_3,125)
h, Cxyh = signal.coherence(density_list_2, density_list_3,125)
plt.semilogy(f, Cxyf)
plt.semilogy(g, Cxyg)
plt.semilogy(h, Cxyh)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()
