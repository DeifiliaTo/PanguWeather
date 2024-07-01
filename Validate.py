import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils.eval as eval
from networks.noBias import PanguModel as NoBiasModel
from networks.pangu import PanguModel as PanguModel
from networks.PanguLite import PanguModel as PanguModelLite
from networks.relativeBias import PanguModel as RelativeBiasModel
from networks.Three_layers import PanguModel as ThreeLayerModel
from utils.data_loader_multifiles import get_data_loader


def init_distributed(params):
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    gpus_per_node = torch.cuda.device_count()


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

    return device, slurm_localid, gpus_per_node, rank, world_size

def validation(params, device, slurm_localid, gpus_per_node):
    # Define patch size, data loader, model
    C          = params['C']
    test_data_loader, test_dataset = get_data_loader(params, params['train_data_path'], dist.is_initialized(), mode='testing', device=device, patch_size=params['patch_size'], subset_size=params['subset_size'], forecast_length=params['forecast_length'])

    if params['model'] == 'pangu':
        model = PanguModel(device=device, C=C, patch_size=params['patch_size'])
    elif params['model'] == 'panguLite':
        model = PanguModelLite(device=device, C=C, patch_size=params['patch_size'])
    elif params['model'] == 'relativeBias':
        model = RelativeBiasModel(device=device, C=C, patch_size=params['patch_size'])
    elif params['model'] == 'noBias':
        model = NoBiasModel(device=device, C=C, patch_size=params['patch_size'])
    elif params['model'] == 'threeLayer':
        model = ThreeLayerModel(device=device, C=C, patch_size=params['patch_size'])
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

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    state = torch.load(params['path_to_model'])
    model.load_state_dict(state['model_state'])

    # Training loop
    loss1 = torch.nn.L1Loss()
    loss2 = torch.nn.L1Loss()

    model.eval()
    with torch.no_grad():
        # Get rank-local numbers of correctly classified and overall samples in training and validation set.
        MSE, acc, total_samples, dt_validation = eval.get_validation_loss(model, test_data_loader, device, lat_crop=params['lat_crop'], lon_crop=params['lon_crop'], forecast_length=params['forecast_length'])
        
        if rank == 0:
            print(f'| MSE T850: {MSE[0][0] :.3f}, {MSE[0][1] :.3f}, {MSE[0][2] :.3f}, {MSE[0][3] :.3f}, {MSE[0][4] :.3f}')
            print(f'| MSE Z500: {MSE[1][0] :.3f}, {MSE[1][1] :.3f}, {MSE[1][2] :.3f}, {MSE[1][3] :.3f}, {MSE[1][4] :.3f}')
            print(f'| MSE T2M:  {MSE[2][0] :.3f}, {MSE[2][1] :.3f}, {MSE[2][2] :.3f}, {MSE[2][3] :.3f}, {MSE[2][4] :.3f}')
            print(f'| MSE U10:  {MSE[3][0] :.3f}, {MSE[3][1] :.3f}, {MSE[3][2] :.3f}, {MSE[3][3] :.3f}, {MSE[3][4] :.3f}')
            print(f'| MSE V10:  {MSE[4][0] :.3f}, {MSE[4][1] :.3f}, {MSE[4][2] :.3f}, {MSE[4][3] :.3f}, {MSE[4][4] :.3f}')
            print(f'| ACC T850: {acc[0][0] :.3f}, {acc[0][1] :.3f}, {acc[0][2] :.3f}, {acc[0][3] :.3f}, {acc[0][4] :.3f}')
            print(f'| ACC Z500: {acc[1][0] :.3f}, {acc[1][1] :.3f}, {acc[1][2] :.3f}, {acc[1][3] :.3f}, {acc[1][4] :.3f}')
            print(f'| ACC T2M:  {acc[2][0] :.3f}, {acc[2][1] :.3f}, {acc[2][2] :.3f}, {acc[2][3] :.3f}, {acc[2][4] :.3f}')
            print(f'| ACC U10:  {acc[3][0] :.3f}, {acc[3][1] :.3f}, {acc[3][2] :.3f}, {acc[3][3] :.3f}, {acc[3][4] :.3f}')
            print(f'| ACC V10:  {acc[4][0] :.3f}, {acc[4][1] :.3f}, {acc[4][2] :.3f}, {acc[4][3] :.3f}, {acc[4][4] :.3f}')
            print(f'| Validation time: {dt_validation : .3f}')
            print(f'| Total samples: {total_samples :.3f}')
                    
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_model', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    params = {}
    params['train_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['valid_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy' 
    params['surface_static_data_path'] =  '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy'  
    params['dt'] = 24
    params['num_data_workers'] = 2
    params['data_distributed'] = True
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['C'] = 192
    params['subset_size'] = 400

    params['Lite'] = False
    params['daily'] = False
    params['forecast_length'] = 5

    # Modify delta_T_divisor based on data path
    if params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
        params['delta_T_divisor'] = 6
    elif params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
        params['delta_T_divisor'] = 1

    # Specify model
    params['model'] = args.model
    params['path_to_model'] = args.path_to_model

    # Set seeds for reproducability
    #torch.backends.cudnn.benchmark = True # This can allegedly improve computational time by ~20%.
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    device, slurm_localid, gpus_per_node, rank, world_size = init_distributed(params)

    if rank == 0:
        print("Reading model from", params['model'])

    # initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite. PS is (2, 4, 4) for all other sizes.
    if params['Lite']:
        params['patch_size'] = (2, 8, 8)
        params['batch_size'] = 6
        params['lat_crop']   = (3, 4)
        params['lon_crop']   = (0, 0)
    else:
        params['patch_size'] = (2, 4, 4)
        params['batch_size'] = 2
        params['lat_crop']   = (1, 2)
        params['lon_crop']   = (0, 0)

    validation(params, device, slurm_localid, gpus_per_node)
    