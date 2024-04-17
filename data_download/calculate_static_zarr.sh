#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition cpuonly

module purge
module load jupyter/tensorflow
source ~/py39/bin/activate
which python

python calculate_static_zarr.py
