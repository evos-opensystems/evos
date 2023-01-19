#!/usr/bin/bash

#PARAMETERS


#coupling parameters
t=1 #2
U=1 #3
V=1 #4


#delete existing results directory
#rm -r 0

for U in {1..2};
do
    python3 ed_extended_hubbard_mesoscopic_leads.py $t $U $V
    python3 fit_optical_cond.py $t $U $V

done











