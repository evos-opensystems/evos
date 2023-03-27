import numpy as np
import os
import matplotlib.pyplot as plt


#qj_ed
os.chdir('qj_n_tot5')
n_system_av = np.loadtxt('n_system_av')
os.chdir('..')

#qj_mps
os.chdir('qj_mps_n_tot5')
n_system_av_mps = np.loadtxt('n_system_av')
os.chdir('..')

#lindblad
os.chdir('lindblad_n_tot5')
nf_system_lind = np.loadtxt('nf_system')
os.chdir('..')


#PLOT
t_max = 20
dt = 0.05
time_v = np.arange( 0, t_max + dt , dt )

fig = plt.figure()
plt.plot(time_v[:-1], n_system_av[:-1], label='n_system_av ed qj', color='red')
plt.plot(time_v[:-1], n_system_av_mps[:-1], label='n_system_av mps qj', color='blue')

plt.plot(time_v[:-1], nf_system_lind, label='n_system_av lind', color='orange')

#plt.plot(time_v, n_system_av-n_system_av_mps, label='n_system_av ed qj', color='red')

plt.legend()
fig.savefig('ed_qj_comp.png')