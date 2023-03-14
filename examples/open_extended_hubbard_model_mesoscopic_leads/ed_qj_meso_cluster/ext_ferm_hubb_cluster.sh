#!/bin/bash

#SBATCH --job-name=edmpscomp

#SBATCH --nodes=1 #8-8
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1

#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Reka.schwengelbeck@physik.uni-muenchen.de
#SBATCH --partition=th-cl,cluster,large  
##SBATCH --chdir=/project/th-scratch/r/Reka.Schwengelbeck/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/cluster_qj

#SBATCH --exclude=th-cl-naples27


#SBATCH --time=3-00:00:00     #3-00:00:00                    
#SBATCH --mem=600GB   #180GB
#SBATCH --output=/project/th-scratch/r/Reka.Schwengelbeck/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/cluster_qj/slurm/output_%j.sout
#SBATCH --error=/project/th-scratch/r/Reka.Schwengelbeck/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/cluster_qj/slurm/output_%j.serr
#SBATCH --constraint=avx2  #needed for using more than 1 cpu per task!
##SBATCH --get-user-env

##export PYTHONPATH=/scratch/r/Reka.Schwengelbeck/SyTeN_cpl:${PYTHONPATH}
##export PATH=/scratch/r/Reka.Schwengelbeck/SyTeN_cpl:${PATH}


source /project/th-scratch/s/Sebastian.Paeckel/init_syten.sh
#./main_test.sh
first_trajectory=$a

echo first trajectory $a

python3 ext_fermi_hubb_quantumjumps.py $first_trajectory

#echo hello

#which python3