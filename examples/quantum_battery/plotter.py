import numpy as np 
import os 
import matplotlib.pyplot as plt

#LOAD DATA
os.chdir('data_lindblad_ed')
n_system_lind = np.loadtxt('n_system')
U_lind = np.loadtxt('U_from_full_state')
S_lind = np.loadtxt('S')
os.chdir('..')

os.chdir('data_qj_ed')
n_system_qj = np.loadtxt('n_system_av')
# U_qj = np.loadtxt('U_av')
os.chdir('..')

os.chdir('data_qj_mps')
n_system_qj_mps = np.loadtxt('n_av')
block_entropies_qj_mps = np.loadtxt('block_entropies_av')
rdm_phon_qj_mps = np.load('rdm_phon.npy')
print(rdm_phon_qj_mps.shape)
quit()
os.chdir('..')

#PLOT
time_v_lind = np.linspace(0,10, len(n_system_lind) )
time_v_qj = np.linspace(0,10, n_system_qj_mps.shape[1] )

fig, ax = plt.subplots()
# ax.plot(time_v_lind, n_system_lind, label='n_system_lind')
# ax.plot(time_v_qj, n_system_qj_mps[2,:], label='n_system_qj_mps')
# ax.plot(time_v_qj, n_system_qj, label='n_system_qj')

ax.plot(time_v_lind, S_lind, label='S_lind')
ax.plot(time_v_qj, block_entropies_qj_mps[3,:], label='n_system_qj_mps')

plt.legend()
fig.savefig('test_mps.png')
