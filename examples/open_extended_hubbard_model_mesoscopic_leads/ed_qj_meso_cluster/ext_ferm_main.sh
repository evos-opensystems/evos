#!/bin/bash


for b in {0,100,200,300,400,500,600,700,800,900}; 
do
    a=$b
    export a
    sbatch ext_ferm_hubb_cluster.sh 

done;
