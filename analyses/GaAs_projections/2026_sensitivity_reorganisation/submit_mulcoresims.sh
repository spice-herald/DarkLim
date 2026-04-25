#!/bin/bash

#for i in $(seq 1 40);
#for i in $(seq 41 80);
#for i in $(seq 0 60);
#for i in $(seq 0 200);
for i in $(seq 0 200);
do
    sbatch runsim.sh $((5000+$i)) $((6000+$i)) $i nominal_gaas_constants.yaml;
done

#sbatch runsim.sh 11 28 0 

