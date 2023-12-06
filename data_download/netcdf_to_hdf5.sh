#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition cpuonly

python write_pressure_to_hdf5.py --nc_path ../../PANGU_ERA5_data_v0/ --h5_path ../../PANGU_ERA5_data_v0/ --year 1980