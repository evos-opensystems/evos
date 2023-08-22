import os
import numpy as np
import matplotlib.pyplot as plt

#parameters
max_bosons = 4 
n_timesteps = 100 #FIXME

#change to directory with MPS-RDM and load RDM and g_2
parent_dir = os.getcwd()
os.chdir('data_mps_lindblad/max_bosons_' + str(max_bosons) + '/dt_0.05/t_max_5.0/mu_l_0.5/mu_r_-0.5')
g_2 = np.load('g_2.npy')
phonon_rdm = np.load('phonon_rdm.npy')
os.chdir(parent_dir)

#load ED g_2
os.chdir('data_lindblad_ed/max_bosons_4/dt_0.05/t_max_5.0/mu_l_0.5/mu_r_-0.5')
sec_ord_coherence_funct = np.loadtxt('sec_ord_coherence_funct')
os.chdir(parent_dir)


#compute g2 from rdm
def compute_sec_ord_coherence_funct(rdm):
    numerator = 0.
    denominator = 0.
    for mode in range(max_bosons+1):
        numerator += mode * (mode - 1) * rdm[ mode, mode ]
        denominator += (mode * rdm[ mode, mode ])
    denominator = denominator ** 2
    sec_ord_coherence_funct = numerator/denominator
    #print('numerator = ',numerator)
    return sec_ord_coherence_funct    

g_2_rdm = np.zeros(n_timesteps)
for time in range(n_timesteps):
    g_2_rdm[time] =  compute_sec_ord_coherence_funct( phonon_rdm[:,:,time] )
    
#compare g2 from rdm with g2 from exp. value
time_v = range(n_timesteps)
fig = plt.figure()
plt.plot(time_v, g_2, label='g_2')
plt.plot(time_v, g_2_rdm, label='g_2_rdm')
plt.plot(time_v, sec_ord_coherence_funct, label='g_2 rdm ed')

plt.legend()
fig.savefig('phonon_rdm_post_processing.png') 
# 

# plt.plot(time_v, n_exp[0,:], label='n0')
# plt.plot(time_v, n_exp[1,:], label='n1')
# plt.plot(time_v, n_b_exp[4,:], label='nb4')
# plt.plot(time_v, n_exp[8,:], label='n8')
# #plt.plot(time_v, phys_dim_phon[:], label='phys_dim_phon')

  

#compute ergotropy


#save obs in the directory where RDM was loaded