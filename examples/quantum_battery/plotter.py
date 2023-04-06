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
n_3_qj = np.loadtxt('n_3_av')

# U_qj = np.loadtxt('U_av')
os.chdir('..')

os.chdir('data_qj_mps')
n_qj_mps = np.load('n_av.npy')
block_entropies_qj_mps = np.load('block_entropies_av.npy')
rdm_phon_qj_mps = np.load('rdm_phon_av.npy')

os.chdir('..')

#PLOT
time_v_lind = np.linspace(0,10, len(n_system_lind) )
time_v_qj = np.linspace(0,10, n_qj_mps.shape[1] )

fig, ax = plt.subplots()
# ax.plot(time_v_lind, n_system_lind, label='n_system_lind')
# ax.plot(time_v_qj, n_system_qj_mps[0,:], label='n_system_qj_mps')
# ax.plot(time_v_qj, n_system_qj, label='n_system_qj_ed')

ax.plot(time_v_qj, n_qj_mps[6,:], label='n_3_qj_mps')
ax.plot(time_v_qj, n_3_qj, label='n_3_qj_ed')

#ax.plot(time_v_qj, np.abs(n_qj_mps[6,:] - n_3_qj ), label='ed qj - mps qj')
#ax.vlines(x=0.8, ymin=0, ymax=0.4, color='red')

#ENTROPY
#ax.plot(time_v_lind, S_lind, label='S_lind')
##ax.plot(time_v_qj, block_entropies_qj_mps[3,:], label='n_system_qj_mps')

#RDM
#ax.plot(time_v_qj, rdm_phon_qj_mps[1,1,:], label='rdm[1,1] mps')


plt.legend()
fig.savefig('test_mps.png')
