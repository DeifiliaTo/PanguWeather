import networks.pangu as pangu
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
import pandas as pd
from netCDF4 import Dataset as DS
import numpy as np
import torch
import glob
import logging
import h5py
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from importlib import reload  # Python 3.4
import utils
from networks.pangu import PanguModel as PanguModel
from utils.data_loader_multifiles import get_data_loader
import torch
from torch import arange
import torch.distributed as dist
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

params = {}
params['train_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train/'
params['valid_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/valid/'
params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means.h5'
params['surface_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means.h5'
params['dt'] = 1
params['n_history'] = 1
params['in_channels'] = 8
params['out_channels'] = 8
params['roll'] = False
params['add_noise'] = False
params['batch_size'] = 2
params['num_data_workers'] = 4
params['data_distributed'] = True

rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
slurm_localid = int(os.getenv("SLURM_LOCALID"))
gpus_per_node = torch.cuda.device_count()

if slurm_job_gpus is not None:
    gpu = rank % gpus_per_node
    assert gpu == slurm_localid
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(device)
    
    # Initialize DDP.
    if params['data_distributed']:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")

else:
    # Initialize DDP.
    if params['data_distributed']:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method="env://")
    # set device to CPU
    device = 'cpu'

if dist.is_initialized(): 
    print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
else:
    print("Running in serial")

# Define patch size, data loader, model
patch_size = (2, 4, 4)
C          = 192
train_data_loader, train_dataset, train_sampler = get_data_loader(params, params['train_data_path'], dist.is_initialized(), train=True, device=device, patch_size=patch_size)
model = PanguModel(device=device, C=C, patch_size=patch_size)
model = model.float().to(device)

# DDP wrapper if GPUs are available
if dist.is_initialized() and gpus_per_node > 0:
    model = DDP( # Wrap model with DDP.
            model, 
            device_ids=[slurm_localid], 
            output_device=slurm_localid,
            find_unused_parameters=True
    )

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
start = time.perf_counter() # Measure training time.

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

loss_history = []

for epoch in range(2):
    model.train()
    
    loss1 = torch.nn.L1Loss()
    loss2 = torch.nn.L1Loss()

    for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output

        input, input_surface, target, target_surface = data[0], data[1], data[2], data[3]
        input = input.float().to(device)
        input_surface = input_surface.float().to(device)
        target = target.float().to(device)
        target_surface = target_surface.float().to(device)

        # Call the model and get the output
        output, output_surface = model(input, input_surface)

        # We use the MAE loss to train the model
        # The weight of surface loss is 0.25
        # Different weight can be applied for different fields if needed
        loss = 0.75 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
        # Call the backward algorithm and calculate the gradient of parameters
        optimizer.zero_grad()
        loss.backward()
        
        # Update model parameters with Adam optimizer
        # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
        optimizer.step()

        dist.all_reduce(loss) # Allreduce rank-local mini-batch losses.
        loss /= world_size    # Average allreduced rank-local mini-batch losses over all ranks.
        loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.

        if rank == 0:
            print(f'Epoch: {epoch+1:03d}/{5:03d} '
                  f'| Batch {i:04d}/{len(train_data_loader):04d} '
                  f'| Averaged Loss: {loss:.4f}')

save_path = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/pangum1.pt'
torch.save(model.state_dict(), save_path)
dist.destroy_process_group()