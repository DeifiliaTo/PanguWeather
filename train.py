import networks.pangu as pangu
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
import torch
import matplotlib.pyplot as plt
from importlib import reload  # Python 3.4
from networks.pangu import PanguModel as PanguModel
from networks.relativeBias import PanguModel as RelativeBiasModel
from utils.data_loader_multifiles import get_data_loader
import torch
import torch.distributed as dist
import utils.data_loader_multifiles as data_loader_multifiles
from utils.data_loader_multifiles import get_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import utils.eval as eval

params = {}
params['train_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train_JQ/'
params['valid_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/valid_JQ/'
params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means_netcdf.npy'
params['surface_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means_netcdf.npy'
params['dt'] = 1
params['n_history'] = 1
params['in_channels'] = 8
params['out_channels'] = 8
params['roll'] = False
params['add_noise'] = False
params['batch_size'] = 2
params['num_data_workers'] = 2
params['data_distributed'] = True
params['filetype'] = 'netcdf' # either hdf5 or netcdf
params['num_epochs'] = 1
num_epochs = params['num_epochs']

rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
slurm_localid = int(os.getenv("SLURM_LOCALID"))
gpus_per_node = torch.cuda.device_count()

# Define dictionaries for variables and pressure levels
variable = {
    'Z': 0,
    'Q': 1,
    'T': 2,
    'U': 3,
    'V': 4
}
pressure_level = {
    1000: 0,
    925: 1,
    850: 2,
    700: 3,
    600: 4,
    500: 5,
    400: 6,
    300: 7,
    250: 8,
    200: 9,
    150: 10,
    100: 11,
    50: 12
}

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
patch_size = (2, 4, 4)
C          = 192
train_data_loader, train_dataset, train_sampler = get_data_loader(params, params['train_data_path'], dist.is_initialized(), train=True, device=device, patch_size=patch_size)
valid_data_loader, valid_dataset = get_data_loader(params, params['valid_data_path'], dist.is_initialized(), train=False, device=device, patch_size=patch_size)
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

# Training loop
loss_history = []
train_acc_history = []
valid_acc_history = []

for epoch in range(num_epochs):
    start_epoch_time = time.perf_counter()
    model.train()
    
    loss1 = torch.nn.L1Loss()
    loss2 = torch.nn.L1Loss()

    train_sampler.set_epoch(epoch)

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
        loss = 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
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
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Batch {i:04d}/{len(train_data_loader):04d} '
                  f'| Averaged Loss: {loss:.4f}')
        
    end_epoch_time = time.perf_counter()

    if rank == 0:
        print(f'Epoch: {int(epoch+1):03d}/{int(num_epochs):03d} '
              f'Elapsed time: {end_epoch_time - start_epoch_time:04f}')
    save_path = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/pangurelbias.pt'
    torch.save(model.state_dict(), save_path)
   # model.eval()

    #with torch.no_grad():
        # Get rank-local numbers of correctly classified and overall samples in training and validation set.
#        right_train = eval.get_loss(model, train_data_loader, device, loss1, loss2) 
    #    right_valid = eval.get_loss(model, valid_data_loader, device, loss1, loss2)

 #       torch.distributed.all_reduce(right_train)
    #    torch.distributed.all_reduce(right_valid)
            
#        train_acc_history.append(right_train)
    #    valid_acc_history.append(right_valid)

        #if rank == 0:
        #    print(f'Epoch: {epoch+1:03d}/{int(num_epochs):03d} '
        #      f'| Train: {right_train :.2f}% '
        #      f'| Validation: {right_valid :.2f}%')
                

save_path = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/pangum2.pt'
torch.save(model.state_dict(), save_path)
dist.destroy_process_group()