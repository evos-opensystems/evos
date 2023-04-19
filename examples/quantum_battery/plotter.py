import numpy as np 
import os 
import matplotlib.pyplot as plt

########LOAD DATA
parent_dir = os.getcwd()
#LINDBLAD #FIXME: change dir by hand
os.chdir('data_lindblad_ed/max_bosons_4/dt_0.02/t_max_5.0/mu_l_0.5/mu_r_-0.5')
n_bos_lind = np.loadtxt('n_bos')
n_system_lind = np.loadtxt('n_system')
U_lind = np.loadtxt('U_from_full_state')
S_lind = np.loadtxt('S')
n_0_lind = np.loadtxt('n_0')
n_3_lind = np.loadtxt('n_3')
os.chdir(parent_dir)

#ED QJ
os.chdir('data_qj_ed')
n_system_qj = np.loadtxt('n_system_av')
n_3_qj = np.loadtxt('n_3_av')
n_bos_qj = np.loadtxt('n_bos_av')
n_0_qj = np.loadtxt('n_0_av')
n_1_qj = np.loadtxt('n_1_av')
# U_qj = np.loadtxt('U_av')
os.chdir('..')


# MPS QJ
os.chdir('data_qj_mps')
nf_qj_mps = np.load('nf_av.npy')
nb_qj_mps = np.load('nb_av.npy')
block_entropies_qj_mps = np.load('block_entropies_av.npy')
rdm_phon_qj_mps = np.load('rdm_phon_av.npy')
bond_dim = np.load('bond_dim_av.npy')
phonon_entanglement_entropy = np.load('phonon_entanglement_entropy_av.npy')
phonon_energy = np.load('phonon_energy_av.npy')
phys_dim = np.load('phys_dim_av.npy')
#print(bond_dim[:,20] )
#quit()
os.chdir('..')

#MPS SCHRO KRYLOV
os.chdir('data_schro_kry_mps')
n_qj_kry_mps = np.load('n.npy')
block_entropies_qj_kry_mps = np.load('block_entropies.npy')
rdm_phon_qj_kry_mps = np.load('rdm_phon.npy')
bond_dim_kry = np.load('bond_dim.npy')
phonon_entanglement_entropy_kry = np.load('phonon_entanglement_entropy.npy')
phonon_energy_kry = np.load('phonon_energy.npy')
phys_dim_kry = np.load('phys_dim.npy')
os.chdir('..')

#MPS LINDBLAD
os.chdir('data_mps_lindblad')
n_mps_lindblad = np.loadtxt('n_exp')
n_b_mps_lindblad = np.loadtxt('n_b_exp')

os.chdir('..')




##########PLOT
time_v_lind = np.linspace(0, 5, len(n_bos_lind) )
#time_v_qj = np.linspace(0, 100,nf_qj_mps.shape[1] )
time_v_qj = np.linspace(0, 5,nf_qj_mps.shape[1] )

fig, ax = plt.subplots()

#MPS LINDBLAD
# for site in np.arange(0, 8, 2):
#     plt.plot(time_v_qj[1:], n_mps_lindblad[site, :], label='mps lind n, site '+str(site) )

# plt.plot(time_v_qj[1:], n_mps_lindblad[0,:], label='mps lind n, site 0 ' )
# plt.plot(time_v_qj[1:], n_mps_lindblad[1,:], label='mps lind n, site 1 ' )
# plt.plot(time_v_qj[1:], n_mps_lindblad[4,:], label='mps lind n, site 4 ' )


# N ON RIGHT LEAD 6
# ax.plot(time_v_qj,nf_qj_mps[4,:], label='n_4_qj_mps')
# ax.plot(time_v_qj, n_3_qj, label='n_3_qj_ed')
# ax.plot(time_v_qj, n_qj_kry_mps[6,:], label='n_3_kry_mps')
#plt.plot(time_v_lind, n_3_lind, label='n_3_lind ed')

# N ON DOT
#ax.plot(time_v_qj,nf_qj_mps[1,:], label='n_1_qj_mps')
#ax.plot(time_v_qj, n_system_qj, label='n_sys_qj_ed')
#ax.plot(time_v_qj, n_system_qj-nf_qj_mps[1,:], label='n_system_qj-n_1_qj_mps')
#plt.plot(time_v_lind, n_system_lind, label='n_system_lind ed')




# N ON LEFT LEAD
#ax.plot(time_v_qj,nf_qj_mps[0,:], label='n_0_qj_mps')
# ax.plot(time_v_qj, n_qj_kry_mps[0,:], label='n_0_kry_mps')
#ax.plot(time_v_qj, n_0_qj, label='n_0_qj_ed')
#plt.plot(time_v_lind, n_0_lind, label='n_0_lind ed')


#OCC BOSONIC SITE
plt.plot(time_v_qj[:], n_bos_qj[:], label='n_bos_ed')
plt.plot(time_v_qj[:],nb_qj_mps[2,:], label='n_bos_mps')
#plt.plot(time_v_qj,nb_qj_mps[2,:] - n_bos_qj[:], label='n_bos_mps - n_bos_ed')

#plt.plot(time_v_qj,nf_qj_mps[4,:], label='n_bos_mps')
#plt.plot(time_v_qj, n_bos_qj -nf_qj_mps[4,:], label='n_bos_mps')
#plt.plot(time_v_lind, n_bos_lind, label='n_bos_lind ed')
#plt.plot(time_v_lind, n_bos_lind -n_bos_qj[1:] , label='n_bos_lind-n_bos_qj')

#plt.plot(time_v_qj,nf_qj_mps[4,:] +nf_qj_mps[5,:], label='n_bos_mps_qj phys + aux')
#plt.plot(time_v_qj, n_qj_kry_mps[4,:], label='n_bos_mps_kry')
#plt.plot(time_v_qj, n_qj_kry_mps[4,:]+n_qj_kry_mps[5,:], label='n_bos_mps_kry phys + aux')

#ERROR ON RIGHT LEAD
#ax.plot(time_v_qj, np.abs(nf_qj_mps[6,:] - n_3_qj ), label='ed qj - mps qj')
#ax.plot(time_v_qj, np.abs(nf_qj_mps[6,:] -n_qj_kry_mps[6,:]  ), label='mps qj - mps kry')
#ax.plot(time_v_qj, np.abs(n_qj_kry_mps[6,:] - n_3_qj ), label='mps kry - ed qj')

#RDM
#for i in range(3):
    #plt.plot(time_v_qj, rdm_phon_qj_mps[i,i,:], label='mode {} mps'.format(i))

#plt.plot(time_v_qj, rdm_phon_qj_mps[i,i,:], label='mode {} mps'.format(i))

#V N ENTROPY
T_l = 1./0.5 #beta_l = 0.5 #FIXME: change this if changed in 'mps_qj.py'
#plt.plot(time_v_qj, phonon_entanglement_entropy[0,:], label='ent entropy')
#plt.plot(time_v_qj, phonon_energy[0,:], label='phon energy')
#plt.plot(time_v_qj, phonon_energy[0,:] - T_l * phonon_entanglement_entropy[0,:], label='W_f')
#plt.plot(time_v_qj,nf_qj_mps[4,:], label='n_bos_mps')


#BOND DIM
i=0
#for i in range(8):
# ax.plot(time_v_qj, bond_dim[i,:], label='bdim site '+str(i))
#ax.vlines(x=0.8, ymin=0, ymax=0.4, color='red')

#PHYSDIM
i=5
#for i in range(8):
#ax.plot(time_v_qj, phys_dim[i,:], label='phys dim site '+str(i))

#ENTROPY
#ax.plot(time_v_lind, S_lind, label='S_lind')
##ax.plot(time_v_qj, block_entropies_qj_mps[3,:], label='n_system_qj_mps')

#RDM
#ax.plot(time_v_qj, rdm_phon_qj_mps[1,1,:], label='rdm[1,1] mps')

plt.ylabel('n')
plt.xlabel('time')
plt.title('Evolution with jumps')
plt.legend()
fig.savefig('test_mps.png')
