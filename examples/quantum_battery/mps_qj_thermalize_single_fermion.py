"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""
import evos.src.methods.mps_quantum_jumps as mps_quantum_jumps
import evos.src.observables.observables as observables

import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as la
import sys
#import math
import os
np.set_printoptions(threshold=sys.maxsize)