#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --mail-user knut.moraa@uzh.ch
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mem-per-cpu=16GB

python simulate_2fold_coincidence.py $1 $2 $3 $4
