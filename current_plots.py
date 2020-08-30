from hamiltonian import *

L = 3
particle_list = [1]
phi = 0
t = 1

# Plot of variation of current with flux
L = 3
phi_list = np.linspace(0,1.5,1000)
particle_list = [2]
ground_current_list_1 = [ground_state_current(0,t,phi,phi_value,particle_list) for phi_value in phi_list]
ground_current_list_2 = [ground_state_current(1,t,phi,phi_value,particle_list) for phi_value in phi_list]
ground_current_list_3 = [ground_state_current(2,t,phi,phi_value,particle_list) for phi_value in phi_list]
fig = plt.figure()
plt.plot(phi_list,ground_current_list_1, label=r"$1 \rightarrow 2$")
plt.plot(phi_list,ground_current_list_2, label = r"$2 \rightarrow 3$")
plt.plot(phi_list,ground_current_list_3, label = r"$3 \rightarrow 1$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$\phi = \frac{\Phi}{\Phi_0}$')
plt.ylabel('Probability Current $J$')
plt.legend(bbox_to_anchor=(0.82, 1))
fig.tight_layout()
#plt.savefig('./plots/groundstate_current_3.pdf')

##################################################################

# Plot for variation in maximum current with hopping strength t
phi = 0.49999999
t_list = np.linspace(0,5,1000)
ground_current_list_1 = [ground_state_current(0,t_value,phi,L,particle_list) for t_value in t_list]
ground_current_list_2 = [ground_state_current(1,t_value,phi,L,particle_list) for t_value in t_list]
ground_current_list_3 = [ground_state_current(2,t_value,phi,L,particle_list) for t_value in t_list]
fig = plt.figure()
plt.plot(t_list,ground_current_list_1, label=r"$1 \rightarrow 2$")
plt.plot(t_list,ground_current_list_2, label = r"$2 \rightarrow 3$")
plt.plot(t_list,ground_current_list_3, label = r"$3 \rightarrow 1$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$t$')
plt.ylabel('Maximum Probability Current $J$')
#plt.legend(bbox_to_anchor=(0.82, 1))
plt.legend()
#fig.tight_layout()
plt.savefig('./plots/groundstate_current_hopping_3_test.pdf')
slope, intercept = np.polyfit((t_list), (ground_current_list_1), 1)

##################################################################
# Ground state current for different number of sites and different particle number
phi = 0.49999999
L_list = np.arange(2,14,1)
ground_current_list_1 = [ground_state_current(0,t,phi,L_value,particle_list) for L_value in L_list]
ground_current_list_2 = [ground_state_current(1,t,phi,L_value,particle_list) for L_value in L_list]
#ground_current_list_3 = [ground_state_current(2,t,phi,L_value,particle_list) for L_value in L_list]
fig = plt.figure()
plt.scatter(L_list,ground_current_list_1,marker = "+", label=r"$1 \rightarrow 2$")
plt.scatter(L_list,ground_current_list_2, marker = "+",label = r"$2 \rightarrow 3$")
#plt.plot(L_list,ground_current_list_3, label = r"$3 \rightarrow 1$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$L$')
plt.ylabel('Maximum Probability Current $J$')
#plt.legend(bbox_to_anchor=(0.82, 1))
plt.legend()
#plt.savefig('./plots/groundstate_current_L.pdf')

##################################################################
# Log-log plot of ground state current with number of sites 
phi = 0.49999999
L_list = np.arange(7,14,1)
ground_current_list_1 = [ground_state_current(0,t,phi,L_value,particle_list) for L_value in L_list]
ground_current_list_2 = [ground_state_current(1,t,phi,L_value,particle_list) for L_value in L_list]
#ground_current_list_3 = [ground_state_current(2,t,phi,L_value,particle_list) for L_value in L_list]
fig = plt.figure()
plt.scatter(np.log(L_list),np.log(ground_current_list_1),marker = "+", label=r"$1 \rightarrow 2$")
plt.scatter(np.log(L_list),np.log(ground_current_list_2), marker = "+",label = r"$2 \rightarrow 3$")
plt.title('Plot of ground-state current')
plt.xlabel(r'$\log(L)$')
plt.ylabel('Log of Maximum Probability Current $J$')
#plt.legend(bbox_to_anchor=(0.82, 1))
plt.legend()
slope, intercept = np.polyfit((np.log(L_list)), (np.log(ground_current_list_1)), 1)
plt.plot(np.log(L_list), slope*np.log(L_list) + intercept)
#plt.savefig('./plots/groundstate_current_logL.pdf')

####################################################################
# Plot of ground state current for different L and occupation
#phi = 0.49999999
occupation_current_plot(11)

#####################################################################
# Quantum quench plot for current
L = 6
particle_list = [3]
phi = 0.2
#phi = 0.1
time_list = np.linspace(0,8,1000)
current_list_1 = [current_quantum_quench(time, 0, 1,t,phi,L,particle_list) for time in time_list]
current_list_2 = [current_quantum_quench(time, 1, 1, t,phi,L,particle_list) for time in time_list]
current_list_3 = [current_quantum_quench(time, 2, 1, t,phi,L,particle_list) for time in time_list]
gradual_current_list_1 = [current_gradual_quantum_quench(time, 0, 1,t,phi,L,particle_list) for time in time_list]
gradual_current_list_2 = [current_gradual_quantum_quench(time, 1, 1, t,phi,L,particle_list) for time in time_list]
gradual_current_list_3 = [current_gradual_quantum_quench(time, 2, 1, t,phi,L,particle_list) for time in time_list]
fig = plt.figure()
plt.plot(time_list, current_list_1, label = 'current on site 1')
plt.plot(time_list, current_list_2, label = 'current on site 2')
plt.plot(time_list, current_list_3, linestyle = '-', label = 'current on site 3')
plt.plot(time_list, gradual_current_list_1, label = 'gradual current on site 1')
plt.plot(time_list, gradual_current_list_2, label = 'gradual current on site 2')
plt.plot(time_list, gradual_current_list_3, label = 'gradual current on site 3')
plt.xlabel('time')
plt.ylabel('Probability Current')
plt.title('Plot of evolution of ground-state current')
plt.legend()
#plt.legend(bbox_to_anchor=(0.64, 0.15))
fig.tight_layout()
#plt.savefig('./plots/site1_quench_phi01.pdf')

current_gradual_perturbation(1,0,1, t, phi, L, particle_list)
