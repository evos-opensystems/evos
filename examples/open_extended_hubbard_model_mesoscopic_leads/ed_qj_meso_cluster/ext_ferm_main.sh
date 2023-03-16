#!/bin/bash


for b in {600,650,700,750,800,850}; 
do
    a=$b
    export a
    sbatch ext_ferm_hubb_cluster.sh 

done;
