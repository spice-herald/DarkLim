#!/bin/bash

#for i in $(seq 1 40);
#for i in $(seq 41 80);
#for i in $(seq 0 60);
#for i in $(seq 0 200);
#
#declare -a masses=(1e-3 3e-3 1e-2 3e-2 5.4e-2 1e-1 3e-1 1e0 3e0 1e1)
declare -a masses=(5.4e-2 1e-1 1.7e-1 3e-1 5.4e-1 1e0 1.7e1 3e0 5.4e0 1.0e1 3.0e1 1.0e2)
for mass in ${masses[@]};
do
    sbatch runsim_gaas.sh $mass 100 1    30 1000 #surface, bad. 
    sbatch runsim_gaas.sh $mass 100 30   30  100 #surface, better. 
    sbatch runsim_gaas.sh $mass 100 100  30   50 #surface, good. 
    sbatch runsim_gaas.sh $mass 100 100 100    2 #UG, good. 
done

#sbatch runsim.sh 11 28 0 

