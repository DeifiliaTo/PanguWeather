#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition cpuonly
#SBATCH --account hk-project-p0021348

module purge
module load jupyter/tensorflow eccodes-2.30.2_i22_ompi40
source ~/py39/bin/activate
which python

python calculate_static_zarr_all.py
