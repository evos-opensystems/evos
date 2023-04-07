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
n_bos_qj = np.loadtxt('n_bos_av')

# U_qj = np.loadtxt('U_av')
os.chdir('..')

os.chdir('data_qj_mps')
n_qj_mps = np.load('n_av.npy')
block_entropies_qj_mps = np.load('block_entropies_av.npy')
rdm_phon_qj_mps = np.load('rdm_phon_av.npy')
bond_dim = np.load('bond_dim_av.npy')
phonon_entanglement_entropy = np.load('phonon_entanglement_entropy_av.npy')
phonon_energy = np.load('phonon_energy_av.npy')
#print(bond_dim[:,20] )
#quit()
os.chdir('..')

#PLOT
time_v_lind = np.linspace(0,20, len(n_system_lind) )
time_v_qj = np.linspace(0,20, n_qj_mps.shape[1] )

fig, ax = plt.subplots()
# ax.plot(time_v_lind, n_system_lind, label='n_system_lind')
# ax.plot(time_v_qj, n_system_qj_mps[0,:], label='n_system_qj_mps')
# ax.plot(time_v_qj, n_system_qj, label='n_system_qj_ed')

#N ON RIGHT LEAD 6
# ax.plot(time_v_qj, n_qj_mps[6,:], label='n_3_qj_mps')
# ax.plot(time_v_qj, n_3_qj, label='n_3_qj_ed')

#RDM
#for i in range(3):
    #plt.plot(time_v_qj, rdm_phon_qj_mps[i,i,:], label='mode {} mps'.format(i))

#plt.plot(time_v_qj, rdm_phon_qj_mps[i,i,:], label='mode {} mps'.format(i))

#OCC BOSONIC SITE
# plt.plot(time_v_qj, n_bos_qj, label='n_bos_ed')
# plt.plot(time_v_qj, n_qj_mps[4,:], label='n_bos_mps')
#plt.plot(time_v_qj, n_qj_mps[5,:], label='i')

#V N ENTROPY
T_l = 1./0.5 #beta_l = 0.5 #FIXME: change this if changed in 'mps_qj.py'
plt.plot(time_v_qj, phonon_entanglement_entropy[0,:], label='ent entropy')
plt.plot(time_v_qj, phonon_energy, label='phon energy')
plt.plot(time_v_qj, phonon_energy - T_l * phonon_entanglement_entropy, label='W_f')


#BOND DIM
#for i in range(8):
    #ax.plot(time_v_qj, bond_dim[i,:], label='bdim site '+str(i))
#ax.plot(time_v_qj, np.abs(n_qj_mps[6,:] - n_3_qj ), label='ed qj - mps qj')
#ax.vlines(x=0.8, ymin=0, ymax=0.4, color='red')

#ENTROPY
#ax.plot(time_v_lind, S_lind, label='S_lind')
##ax.plot(time_v_qj, block_entropies_qj_mps[3,:], label='n_system_qj_mps')

#RDM
#ax.plot(time_v_qj, rdm_phon_qj_mps[1,1,:], label='rdm[1,1] mps')


plt.legend()
fig.savefig('test_mps.png')
