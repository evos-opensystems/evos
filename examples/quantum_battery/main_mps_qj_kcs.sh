#!/bin/bash

#SBATCH --job-name=qjumpdt005g1kappa8
##SBATCH -D /dss/dsskcsfs01/pn34ze/pn34ze-dss-0007/evos/examples/quantum_battery
#SBATCH --nodes=1 #8-8 4-4 1-1
#SBATCH --ntasks-per-node=30  #25
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mattia.moroder@campus.lmu.de

#SBATCH --output=/dss/dsskcsfs01/pn34ze/pn34ze-dss-0007/evos/examples/quantum_battery/slurm/output_%j.sout
#SBATCH --error=/dss/dsskcsfs01/pn34ze/pn34ze-dss-0007/evos/examples/quantum_battery/slurm/output_%j.serr

#SBATCH --time=3:00:00   #3-00:00:00
#SBATCH --mem=180GB   #180GB #369GB 
#SBATCH --export=ALL
#SBATCH --clusters=KCS
#SBATCH --get-user-env
#SBATCH --partition=kcs_batch


python3 mps_qj_new_lattice.py