import datetime
import json
import os
import time
import logging

import torch
import torch.distributed as dist
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

import matplotlib.pyplot as plt
import numpy as np

import utils.eval as eval
from networks.noBias import PanguModel as NoBiasModel
from networks.pangu import PanguModel as PanguModel
from networks.PanguLite import PanguModel as PanguModelLite
from networks.PanguLite2DAttention import PanguModel as PanguLite2D
from networks.PanguLite2DAttentionPosEmbed import PanguModel as TwoDimPosEmbLite
from networks.PositionalEmbedding import PanguModel as PositionalEmbedding
from networks.relativeBias import PanguModel as RelativeBiasModel
from networks.Three_layers import PanguModel as ThreeLayerModel
from networks.TwoDimensional import PanguModel as TwoDimPosEmb
from utils.data_loader_multifiles import get_data_loader

def load_model(path_to_model, model, optimizer=None):
    state = torch.load(path_to_model, map_location=torch.device('cpu'))

    print("loading model")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        new_checkpoint = {}
        for key, value in state['model_state'].items():
            new_key = key.replace('module.', '') if 'module.' in key else key
            new_checkpoint[new_key] = value
        model.load_state_dict(new_checkpoint)
        
    
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state'])
        print("loading optimizer state")
    return model, optimizer

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    """
    Initialize DistributedDataParallel 
    
    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)
    
    # Initialize DistributedDataParallel.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    
    return device, slurm_localid, rank, world_size

def inference_loop(params, device, slurm_localid):
    """
    train the model.

    params: Dict
    device: String
    slurm_localid: int
    """
    # Define patch size, data loader, model
    dim          = params['C']
    two_dimensional = False
    if params['model'] == '2D' or params['model'] == '2Dim192' or params['model'] == '2DPosEmb' or params['0odel'] == '2DPosEmbLite':
        two_dimensional = True
        params['patch_size'] = params['patch_size'][-2:] # Patch size should 2 values
    train_data_loader = get_data_loader(params, params['train_data_path'], dist.is_initialized(), mode='train', patch_size=params['patch_size'], two_dimensional=two_dimensional)

    # Initialize model based on key
    if params['model'] == 'pangu':
        model = PanguModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'panguLite':
        model = PanguModelLite(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'relativeBias':
        model = RelativeBiasModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'noBiasLite':
        model = NoBiasModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == '2D':
        model = PanguLite2D(device=device, dim=int(1.5*dim), patch_size=params['patch_size'])
    elif params['model'] == 'threeLayer':
        model = ThreeLayerModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'positionEmbedding':
        model = PositionalEmbedding(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == '2DPosEmb':
        model = TwoDimPosEmb(device=device, dim=int(1.5*dim), patch_size=params['patch_size'])
    elif params['model'] == '2DPosEmbLite':
        model = TwoDimPosEmbLite(device=device, dim=int(1.5*dim), patch_size=params['patch_size'])
    elif params['model'] == '2Dim192':
        model = PanguLite2D(device=device, dim=int(dim), patch_size=params['patch_size'])
    else: 
        raise NotImplementedError(params['model'] + ' model does not exist.')

    model = model.to(torch.float32).to(device)
    
    # DistributedDataParallel wrapper
    model = DistributedDataParallel(
            model, 
            device_ids=[slurm_localid], 
            output_device=slurm_localid
    )
    
    state = torch.load(params['model_path'], map_location='cpu') 
    model.load_state_dict(state['model_state'])
    model.eval()

    for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
        input, input_surface, _, _ = data[0], data[1], data[2][0], data[3][0]
        input = input.to(torch.float32).to(device)
        input_surface = input_surface.to(torch.float32).to(device)
    
        with torch.autocast(device_type="cuda"):
            # Call the model and get the output
            output, output_surface = model(input, input_surface)
            if params['model'] == '2D' or params['model'] == '2Dim192' or params['model'] == '2DPosEmb' or params['model'] == '2DPosEmbLite':
                output = output.reshape(-1, 5, 13, output.shape[-2], output.shape[-1])
        print("Result generated", output[0][0][0][0])
        plt.pcolor(np.flipud(output[0][0][1].cpu().detach()))        
        plt.savefig('trained_models/test/images/forecast' + str(torch.distributed.get_rank())+ '_image' + str(i) + '.png')

                
if __name__ == '__main__':
    params = {}
    params['train_data_path'] =  'era_subset.zarr' 
    params['valid_data_path'] =  'era_subset.zarr' 
    params['pressure_static_data_path'] = 'constant_masks/pressure_zarr.npy' 
    params['surface_static_data_path'] =  'constant_masks/surface_zarr.npy'  
    params['dt'] = 24
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['C'] = 192
    params['Lite'] = True
    params['model_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/twoDimensionalLite/20240521_306979085/97_2D20240521_306979085.pt' 

    # Specify model
    params['model'] = '2D'
    # pangu        = full run
    # panguLite    = light model
    # relativeBias = relative bias
    # noBias       = no bias
    # 2D           = 2D transformer
    # threeLayer   = 1 more down/upsampling layer
    # positionEmbedding     = absolute position embedding

    # Save directory
    # CHANGE TO YOUR OUTPUT SAVE DIRECTORY
    base_save_dir = 'trained_models/test/'
        
    
    # Set seeds for reproducibility
    set_all_seeds(1)
    
    device, slurm_localid, rank, world_size = init_distributed()
    
    params['batch_size'] = 12
    # initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite.
    # patch size is (2, 4, 4) for all other sizes.
    if params['Lite']:
        params['patch_size'] = (2, 8, 8)
        
        params['lat_crop']   = (3, 4) # Do not change if input image size of (721, 1440)
        params['lon_crop']   = (0, 0) # Do not change if input image size of (721, 1440)
    else:
        params['patch_size'] = (2, 4, 4)
        params['batch_size'] = 1
        params['lat_crop']   = (1, 2) # Do not change if input image size of (721, 1440)
        params['lon_crop']   = (0, 0) # Do not change if input image size of (721, 1440)

    # CHANGE TO YOUR DATA DIRECTORIES
    if params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
        params['delta_T_divisor'] = 6 # Required for WeatherBench2 download with 6-hourly time resolution
    elif params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
        params['delta_T_divisor'] = 1 # Required for WeatherBench2 download with hourly time resolution
    else:
        params['delta_T_divisor'] = 6 # Baseline assumption is 6-hourly subsampled data

    inference_loop(params, device, slurm_localid)
                        
    dist.destroy_process_group()
    
