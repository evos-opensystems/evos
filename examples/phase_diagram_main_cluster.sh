#!/usr/bin/bash

#PARAMETERS
#SBATCH --job-name MyJob
#SBATCH --comment "No one will read this anyway..."

#SBATCH --mail-type=ALL
#SBATCH --mail-user=Reka.schwengelbeck@physik.uni-muenchen.de


#coupling parameters
t=1 #2
U=1 #3
V=1 #4


#delete existing results directory
#rm -r 0

for U in {-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5}; do for V in {-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5};
do
    python3 ed_extended_hubbard_mesoscopic_leads.py $t $U $V
    python3 fit_optical_cond.py $t $U $V

done; done 