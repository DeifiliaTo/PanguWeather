#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --account='hk-project-p0021348'
#SBATCH --output='test.out'
#SBATCH --job-name='test'
#SBATCH --constraint='LSDF'

module purge # Unload all models.
module load jupyter/tensorflow eccodes-2.30.2_i22_ompi40

source ~/py39_pangu/bin/activate
which python

# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12340

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


srun python -u train.py
