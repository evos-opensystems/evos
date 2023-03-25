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



#PLOT
t_max = 20
dt = 0.01
time_v = np.arange( 0, t_max + dt , dt )

fig = plt.figure()
plt.plot(time_v[:-1], n_system_av, label='n_system_av ed qj', color='red')
plt.plot(time_v[:-1], n_system_av_mps, label='n_system_av ed qj', color='blue')

#plt.plot(time_v, n_system_av-n_system_av_mps, label='n_system_av ed qj', color='red')

plt.legend()
fig.savefig('ed_qj_comp.png')