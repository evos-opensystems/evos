import numpy as np 
import os 
import matplotlib.pyplot as plt

########LOAD DATA
parent_dir = os.getcwd()

#ED LINDBLAD
ed_lind_dirname = 'data_lindblad_ed/max_bosons_4/dt_0.02/t_max_5.0/mu_l_0.5/mu_r_-0.5'
os.chdir(ed_lind_dirname)

n_bos_lind = np.loadtxt('n_bos')
n_system_lind = np.loadtxt('n_system')
U_lind = np.loadtxt('U_from_full_state')
S_lind = np.loadtxt('S')
n_0_lind = np.loadtxt('n_0')
n_3_lind = np.loadtxt('n_3')
f_neq_lind = np.loadtxt('f_neq')
f_eq_vector_lind = np.loadtxt('f_eq_vector')
sec_ord_coherence_funct_lind = np.loadtxt('sec_ord_coherence_funct')
os.chdir(parent_dir)


# MPS QJ
qj_mps_dirname = 'data_qj_mps/max_bosons_4/dt_0.02/t_max_5/mu_l_0.5/mu_r_-0.5/n_trajectories_1000/first_trajectory_0'
os.chdir(qj_mps_dirname)
nf_qj_mps = np.load('nf_av.npy')
nb_qj_mps = np.load('nb_av.npy')
rdm_phon_qj_mps = np.load('rdm_phon_av.npy')
bond_dim = np.load('bond_dim_av.npy')
#phonon_entanglement_entropy = np.load('phonon_entanglement_entropy_av.npy')
#phonon_energy = np.load('phonon_energy_av.npy')
phys_dim = np.load('phys_dim_av.npy')
free_energy_neq_qj_mps = np.load('free_energy_neq_av.npy')
sec_ord_coherence_funct_qj_mps = np.load('sec_ord_coherence_funct_av.npy')

os.chdir(parent_dir)


#ED QJ
qj_ed_dirname = 'data_qj_ed/max_bosons_4/dt_0.02/t_max_5/mu_l_0.5/mu_r_-0.5/n_trajectories_1000/first_trajectory_0'
os.chdir(qj_ed_dirname)
n_1_av_ed_qj = np.loadtxt('n_1_av')
n_bos_av_ed_qj = np.loadtxt('n_bos_av')
free_energy_neq_qj_ed = np.loadtxt('free_energy_neq_av')
sec_ord_coherence_funct_qj_ed = np.loadtxt('sec_ord_coherence_funct_av')
os.chdir(parent_dir)


#PLOTTER
time_v = np.linspace(0, 5, len(sec_ord_coherence_funct_lind) )
fig, ax = plt.subplots()

#g2
#plt.plot(time_v, sec_ord_coherence_funct_lind, label='sec_ord_coherence_funct ed')
#plt.plot(time_v, sec_ord_coherence_funct_qj_mps[0,1:], label='sec_ord_coherence_funct_qj mps')
#plt.plot(time_v, sec_ord_coherence_funct_qj_ed[1:], label='sec_ord_coherence_funct_qj ed')

#W_f
#plt.plot(time_v, f_neq_lind-f_eq_vector_lind, label='W_f lind')
#plt.plot(time_v, free_energy_neq_qj_mps[0,1:] -f_eq_vector_lind, label='W_f qj mps')
#plt.plot(time_v, free_energy_neq_qj_ed[1:] -f_eq_vector_lind, label='W_f qj ed')

##fermionic occupations
#system
plt.plot(time_v, n_system_lind, label='n_system_lind ed')
plt.plot(time_v, nf_qj_mps[1,1:], label='nf_qj_mps 1')
plt.plot(time_v, n_1_av_ed_qj[1:], label='nf_qj_ed 1')

#bond dimension
# for i in range(5):
#     ax.plot(time_v, bond_dim[i,1:], label='bdim site '+str(i))
#     ax.vlines(x=0.8, ymin=0, ymax=0.4, color='red')


plt.legend()
os.chdir('plots')
fig.savefig('plotter_ed_lind_mps_qj.png')
