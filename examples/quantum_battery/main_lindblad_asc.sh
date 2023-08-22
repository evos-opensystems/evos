#!/bin/bash

#SBATCH --job-name=lind_qb

#SBATCH --nodes=1 #8-8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mattia.moroder@campus.lmu.de
#SBATCH --partition=th-cl,cluster,th-ws,large  #,large  #th-ws,th-cl,cluster

#SBATCH --time=1-00:00:00    #6-00:00:00                    
#SBATCH --mem=100GB   #250GB
#SBATCH --output=/project/th-scratch/m/Mattia.Moroder/evos/examples/quantum_battery/slurm/output_%j.sout
#SBATCH --error=/project/th-scratch/m/Mattia.Moroder/evos/examples/quantum_battery/slurm/output_%j.serr

#SBATCH --constraint=avx2  #needed for using more than 1 cpu per task!

python3 ed_lindblad.py  -b 40 -dt 1000 -t_max 10000 -mu_l 1 -mu_r -1