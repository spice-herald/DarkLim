#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --mail-user knut.moraa@uzh.ch
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mem-per-cpu=32GB

python fast_inference.py  --wimp_mass $1 --nexp $2 --LEE_improvement $3 --t_days $4 --flat_rate_DRU $5 --constant_file gaas_defaults.yaml
