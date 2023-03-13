#!/bin/bash

#SBATCH --job-name=singletraj

#SBATCH --nodes=1 #8-8
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1

#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Reka.schwengelbeck@physik.uni-muenchen.de
#SBATCH --partition=th-cl,cluster,large  

#SBATCH --exclude=th-cl-naples27


#SBATCH --time=00:20:00     #3-00:00:00                    
#SBATCH --mem=10GB   #180GB
#SBATCH --output=/project/th-scratch/r/Reka.Schwengelbeck/evos/examples/open_extended_hubbard_model_mesoscopic_leads/slurm/output_%j.sout
#SBATCH --error=/project/th-scratch/r/Reka.Schwengelbeck/evos/examples/open_extended_hubbard_model_mesoscopic_leads/slurm/output_%j.serr
#SBATCH --constraint=avx2  #needed for using more than 1 cpu per task!
##SBATCH --get-user-env

##export PYTHONPATH=/scratch/r/Reka.Schwengelbeck/SyTeN_cpl:${PYTHONPATH}
##export PATH=/scratch/r/Reka.Schwengelbeck/SyTeN_cpl:${PATH}


source /project/th-scratch/s/Sebastian.Paeckel/init_syten.sh
#./main_test.sh

python3 ext_fermi_hubb_quantumjumps.py 

#echo hello

#which python3