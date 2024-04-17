#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=5:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --account='hk-project-p0021348'
#SBATCH --output='validate.out'

module purge # Unload all models.
module load jupyter/tensorflow eccodes-2.30.2_i22_ompi40

source ~/py39/bin/activate
which python

# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12341

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python -u Validate.py --path_to_model='/home/hk-project-epais/ke4365/pangu-weather/trained_models/pangu/20240404_884476961/119_pangu20240404_884476961.pt' --model='pangu'
