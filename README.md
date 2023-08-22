# EVOS (EvolVing Open Systems)

Welcome to EVOS. Have fun with open systems!

## Installation 

To install EVOS, run **make install** in the evos directory.

## Getting started

To understand how to use evos, have a look at some examples that you can find in **evos/examples/get_started_examples**. 

## Structure

EVOS is a python package that computes the dynamics of open system either with exact diagonalization (**ED**) or with matrix-product states (**MPS**).
The **MPS** calculations are performed with [syten](https://syten.eu), which needs to be installed separately.
In **evos/evos/src/** you find the three main folders of the package:

- **lattice**: contains all the implemented ED lattices, including spins, electrons, spinless fermions and phonons.
- **methods**: contains all the time-evolution methods implemented in EVOS, both for **ED** and **MPS**, like a Lindblad and a Quantum Jumps solver. In the subdirectory **partial_traces** the methods for computing partial traces for different lattices can be found
- **observables**: contains a class that can be used to compute observables  both with **ED** and **MPS**. It is particularly useful when
one needs to track many observables or one when uses trajectory methods, since averages, errors and trajectories preprocessing are taken care of.


## Documentation

You can access the documentation by running **make doc** in **evos**.

## References

1. Quantum jumps: [A. Daley] (https://arxiv.org/abs/1405.6694).

2. Matrix-product state algorithms for Markovian and non-Markovian systems: [M. Moroder] (https://arxiv.org/abs/2207.08243).

3. Mesoscopic leads method: [A. Lacerda] (https://arxiv.org/abs/2206.01090).

4. Vectorization of lindblad equation for time evolution MPS: [S. Wolff] (https://arxiv.org/abs/2004.01133).

5. Vectorization of lindblad equation for direct steady-state calculation with MPS: [H.P. Casagrande] (https://arxiv.org/abs/2009.08200).

## To Add

1. ED and MPS hops
2. MPS time evolution with vectorized Lindbladian
3. MPS steady-state calculation with $\hat{L}^{\dagger} \hat{L} $

## To Test

1. mps_quantum_jumps_time_dep.py has to be tested.

## To Fix

1. remove observables.py and switch to observables_pickled.py in all examples and scripts