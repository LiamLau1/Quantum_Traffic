from hamiltonian import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 6
t = 1
phi = 0.49999999
particle_list = [3]
perturbed_site = 1

def animate(frame, L, t, phi, particle_list, perturbed_site):
    current_time = frame/5 -4
    global initial_state_time
    if current_time == -4:
        # Act potential on site 1
        H_unperturbed = gradual_Hamiltonian_perturbed(-4,perturbed_site,t,phi,L,particle_list) 
        # Pick the ground state
        eigens = linalg.eig(H_unperturbed)
        initial_state_time_temp = eigens[1][:,np.argmin(eigens[0])]
    else:
        # Hamiltonian is now time dependent, so need to solve set of linear DEs
        H_perturbed = gradual_Hamiltonian_perturbed(current_time,perturbed_site,t,phi,L,particle_list)
        initial_state_time_temp = solve_ivp(tdse, (-4,current_time), initial_state_time,method = 'BDF',  args = (t, phi, L, particle_list)).y[:,-1]

    initial_state_time = initial_state_time_temp
    density_site_1 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 0, t, phi, L,particle_list).real
    density_site_2 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 1, t, phi, L,particle_list).real
    density_site_3 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 2, t, phi, L,particle_list).real
    density_site_4 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 3, t, phi, L,particle_list).real
    density_site_5 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 4, t, phi, L,particle_list).real
    density_site_6 = density_gradual_perturbation(frame/50 - 4, initial_state_time, 5, t, phi, L,particle_list).real
    #density_site_7 = density_quantum_quench(frame/100, 6, 1, t, phi, L,particle_list).real
    #density_site_8 = density_quantum_quench(frame/100, 7, 1, t, phi, L,particle_list).real
    #density_site_9 = density_quantum_quench(frame/100, 8, 1, t, phi, L,particle_list).real
    #density_site_10 = density_quantum_quench(frame/100, 9, 1, t, phi, L,particle_list).real
    #line.set_ydata([density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6, density_site_7, density_site_8, density_site_9, density_site_10])
    line.set_ydata([density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6])

"""
def animate(frame, L, t, phi, particle_list):
    density_site_1 = density_quantum_quench(frame/50, 0, 1, t, phi, L,particle_list).real
    density_site_2 = density_quantum_quench(frame/50, 1, 1, t, phi, L,particle_list).real
    density_site_3 = density_quantum_quench(frame/50, 2, 1, t, phi, L,particle_list).real
    density_site_4 = density_quantum_quench(frame/50, 3, 1, t, phi, L,particle_list).real
    density_site_5 = density_quantum_quench(frame/50, 4, 1, t, phi, L,particle_list).real
    density_site_6 = density_quantum_quench(frame/50, 5, 1, t, phi, L,particle_list).real
    #density_site_7 = density_quantum_quench(frame/100, 6, 1, t, phi, L,particle_list).real
    #density_site_8 = density_quantum_quench(frame/100, 7, 1, t, phi, L,particle_list).real
    #density_site_9 = density_quantum_quench(frame/100, 8, 1, t, phi, L,particle_list).real
    #density_site_10 = density_quantum_quench(frame/100, 9, 1, t, phi, L,particle_list).real
    #line.set_ydata([density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6, density_site_7, density_site_8, density_site_9, density_site_10])
    line.set_ydata([density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6])
"""
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(0.9, 6.1), ylim=(0, 1))
#sites = [1,2,3, 4,5, 6, 7, 8, 9, 10]
sites = [1,2,3, 4,5, 6]
plt.xticks(sites)
plt.yticks(np.arange(0,1.1, 0.1))
plt.xlabel('site')
plt.ylabel('occupation')
H_unperturbed = gradual_Hamiltonian_perturbed(-4,perturbed_site,t,phi,L,particle_list) 
# Pick the ground state
eigens = linalg.eig(H_unperturbed)
initial_state_time = eigens[1][:,np.argmin(eigens[0])]
density_site_1 = density_gradual_perturbation(-4, initial_state_time, 0, t, phi, L,particle_list).real
density_site_2 = density_gradual_perturbation(-4, initial_state_time, 1, t, phi, L,particle_list).real
density_site_3 = density_gradual_perturbation(-4, initial_state_time, 2, t, phi, L,particle_list).real
density_site_4 = density_gradual_perturbation(-4, initial_state_time, 3, t, phi, L,particle_list).real
density_site_5 = density_gradual_perturbation(-4, initial_state_time, 4, t, phi, L,particle_list).real
density_site_6 = density_gradual_perturbation(-4, initial_state_time, 5, t, phi, L,particle_list).real
#density_site_7 = density_quantum_quench(0, 6, 1, t, phi, L,particle_list).real
#density_site_8 = density_quantum_quench(0, 7, 1, t, phi, L,particle_list).real
#density_site_9 = density_quantum_quench(0, 8, 1, t, phi, L,particle_list).real
#density_site_10 = density_quantum_quench(0, 9, 1, t, phi, L,particle_list).real
#line, = ax.plot(sites, [density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6, density_site_7, density_site_8, density_site_9, density_site_10], marker = '+')
line, = ax.plot(sites, [density_site_1, density_site_2, density_site_3, density_site_4, density_site_5, density_site_6], marker = '+')

anim = animation.FuncAnimation(fig, animate, interval = 200, frames = 400, fargs = (L,t,phi,particle_list, perturbed_site))
anim.save('./plots/2gradualL63049999999.mp4')
#plt.draw()
#plt.show()
