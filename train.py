import networks.pangu as pangu
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
import torch
import matplotlib.pyplot as plt
from importlib import reload  # Python 3.4
from networks.pangu import PanguModel as PanguModel
from networks.PanguLite import PanguModel as PanguModelLite
from networks.relativeBias import PanguModel as RelativeBiasModel
from networks.noBias import PanguModel as NoBiasModel
from utils.data_loader_multifiles import get_data_loader
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import utils.eval as eval
from torch.utils.tensorboard import SummaryWriter

params = {}
# Zarr data
params['zarr_path'] = '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
params['train_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train_JQ/'
params['valid_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/valid_JQ/'
params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy' # '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means_netcdf.npy'
params['surface_static_data_path'] =  '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy'  # '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means_netcdf.npy'
params['dt'] = 1
params['num_data_workers'] = 2
params['data_distributed'] = True
params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
params['num_epochs'] = 2
num_epochs = params['num_epochs']
params['C'] = 192
params['subset_size'] = 100
params['validation_subset_size'] = 100
       
rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
slurm_localid = int(os.getenv("SLURM_LOCALID"))
gpus_per_node = torch.cuda.device_count()

# Specify model
params['model'] = 'panguLite'
# pangu        = full run
# panguLite    = light model
# relativeBias = relative bias
# noBias       = no bias
# 2D           = 2D transformer

# Save directory
base_save_path = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/'
params['save_path'] = {
    'pangu': base_save_path + 'pangu.pt',
    'panguLite': base_save_path + 'panguLite.pt',
    'relativeBias': base_save_path + 'relativeBias.pt',
    'noBias': base_save_path + 'noBias.pt',
    '2D': base_save_path + 'twoDimensional.pt'
}

# initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite. PS is (2, 4, 4) for all other sizes.
if params['model'] == 'panguLite':
    params['patch_size'] = (2, 8, 8)
    params['batch_size'] = 7
else:
    params['patch_size'] = (2, 4, 4)
    params['batch_size'] = 2

# Define writer
writer = SummaryWriter()

# Initialize GPUs and dataloaders
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
C          = params['C']
train_data_loader, train_dataset, train_sampler = get_data_loader(params, params['train_data_path'], dist.is_initialized(), train=True, device=device, patch_size=params['patch_size'], subset_size=params['subset_size'])
valid_data_loader, valid_dataset = get_data_loader(params, params['valid_data_path'], dist.is_initialized(), train=False, device=device, patch_size=params['patch_size'], subset_size=params['validation_subset_size'])

if params['model'] == 'pangu':
    model = PanguModel(device=device, C=C, patch_size=params['patch_size'])
elif params['model'] == 'panguLite':
    model = PanguModelLite(device=device, C=C, patch_size=params['patch_size'])
elif params['model'] == 'relativeBias':
    model = RelativeBiasModel(device=device, C=C, patch_size=params['patch_size'])
elif params['model'] == 'noBias':
    model = NoBiasModel(device=device, C=C, patch_size=params['patch_size'])
elif params['model'] == '2D':
    raise NotImplementedError("2D model is not yet implemented")
else: 
    raise NotImplementedError(params['model'] + ' model does not exist.')

model = model.to(torch.float32).to(device)

# DDP wrapper if GPUs are available
if dist.is_initialized() and gpus_per_node > 0:
    model = DDP( # Wrap model with DDP.
            model, 
            device_ids=[slurm_localid], 
            output_device=slurm_localid,
            find_unused_parameters=False
    )

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
scaler = GradScaler()
start = time.perf_counter() # Measure training time.

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

# Training loop
train_loss_history = []
valid_loss_history = []
best_val_loss = 100000           # Arbitrary large number

loss1 = torch.nn.L1Loss()
loss2 = torch.nn.L1Loss()

for epoch in range(params['num_epochs']):
    start_epoch_time = time.perf_counter()
    model.train()
    
    train_sampler.set_epoch(epoch)
    epoch_average_loss = 0
    for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
        optimizer.zero_grad()
        input, input_surface, target, target_surface = data[0], data[1], data[2], data[3]
        input = input.to(torch.float32).to(device)
        input_surface = input_surface.to(torch.float32).to(device)
        target = target.to(torch.float32).to(device)
        target_surface = target_surface.to(torch.float32).to(device)
        with autocast():
            # Call the model and get the output
            output, output_surface = model(input, input_surface)

            # We use the MAE loss to train the model
            # The weight of surface loss is 0.25
            # Different weight can be applied for different fields if needed
            loss = 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
            # Call the backward algorithm and calculate the gradient of parameters
        torch.cuda.empty_cache()
        scaler.scale(loss).backward()
            
        # Update model parameters with Adam optimizer
        # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
#        optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        #print(optimizer.param_groups)

        dist.all_reduce(loss) # Allreduce rank-local mini-batch losses.
        loss /= world_size    # Average allreduced rank-local mini-batch losses over all ranks.
        train_loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.
        epoch_average_loss += loss.item()
        #if rank == 0:
        #    print(model.)
        if rank == 0:
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Batch {i:04d}/{len(train_data_loader):04d} '
                  f'| Averaged Loss: {loss:.4f}')
    epoch_average_loss /= len(train_data_loader)
    if rank == 0:
        writer.add_scalar("Loss/train", epoch_average_loss, epoch)       
        
    end_epoch_time = time.perf_counter()

    if rank == 0:
        print(f'Epoch: {int(epoch+1):03d}/{int(num_epochs):03d} '
              f'Elapsed time: {end_epoch_time - start_epoch_time:04f}')
    
    model.eval()
    with torch.no_grad():
        # Get rank-local numbers of correctly classified and overall samples in training and validation set.
        val_loss = eval.get_loss(model, valid_data_loader, device, loss1, loss2)
        dist.all_reduce(val_loss)
        if rank == 0:
            valid_loss_history.append(val_loss.item())
            writer.add_scalar("Loss/Val", val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save model to path
                torch.save(model.state_dict(), params['save_path'][params['model']])
            print(f'Epoch: {int(epoch)+1:03d}/{int(num_epochs):03d} '
            f'| Train: {epoch_average_loss :.2f}'
            f'| Validation: {val_loss :.2f}')
        
                
# How can we verify that at least one model will be saved? Currently only saves when in case of the best validation loss
dist.destroy_process_group()