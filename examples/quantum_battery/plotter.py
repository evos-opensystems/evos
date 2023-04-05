import numpy as np 
import os 
import matplotlib.pyplot as plt

#LOAD DATA
os.chdir('data_lindblad_ed')
n_system_lind = np.loadtxt('n_system')
U_lind = np.loadtxt('U_from_full_state')
os.chdir('..')

os.chdir('data_qj_ed')
n_system_qj = np.loadtxt('n_system_av')
U_qj = np.loadtxt('U_av')
os.chdir('..')

#PLOT
time_v_lind = np.linspace(0,100, len(n_system_lind))
time_v_qj = np.linspace(0,100, len(n_system_qj))

# plt.plot(time_v_lind, n_system_lind, label='n_system_lind')
# plt.plot(time_v_qj, n_system_qj, label='n_system_qj')

plt.plot(time_v_lind, U_lind, label='U_lind')
plt.plot(time_v_qj, U_qj, label='U_qj')

plt.legend()
plt.show()