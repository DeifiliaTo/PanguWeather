import sys

import torch

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/hk-project-epais/ke4365/pangu-weather')
import datetime
import json
import os
import time

import numpy as np
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

import utils.eval as eval
from networks.noBias import PanguModel as NoBiasModel
from networks.pangu import PanguModel as PanguModel
from networks.PanguLite import PanguModel as PanguModelLite
from networks.PanguLite2DAttention import PanguModel as PanguLite2D
from networks.PositionalEmbedding import PanguModel as PositionalEmbedding
from networks.relativeBias import PanguModel as RelativeBiasModel
from networks.Three_layers import PanguModel as ThreeLayerModel
from utils.data_loader_multifiles import get_data_loader


def init_distributed(params):
    """
    Initialize DistributedDataParallel or set to CPU.
    
    params: Dict
        dictionary specifying run parameters
    
    Returns
    -------
    device: String
    slurm_localid: int
    gpus_per_node: int
    rank: int
    world_size: int
    """
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
        # Initialize DistributedDataParallel.
        if params['data_distributed']:
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    else:
        # Initialize DistributedDataParallel.
        if params['data_distributed']:
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method="env://")
        # set device to CPU
        device = 'cpu'

    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    else:
        print("Running in serial")

    return device, slurm_localid, gpus_per_node, rank, world_size

def training_loop(params, device, slurm_localid, gpus_per_node):
    """
    train the model.

    params: Dict
    device: String
    slurm_localid: int
    gpus_per_node: int
    """
    # Define patch size, data loader, model
    dim          = params['C']
    two_dimensional = False
    if params['model'] == '2D' or params['model'] == '2Dim192':
        two_dimensional = True
    train_data_loader = get_data_loader(params, params['train_data_path'], dist.is_initialized(), mode='train', device=device, patch_size=params['patch_size'], subset_size=params['subset_size'], two_dimensional=two_dimensional)
    valid_data_loader = get_data_loader(params, params['valid_data_path'], dist.is_initialized(), mode='validation', device=device, patch_size=params['patch_size'], subset_size=params['validation_subset_size'], two_dimensional=two_dimensional)

    if params['model'] == 'pangu':
        model = PanguModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'panguLite':
        model = PanguModelLite(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'relativeBias':
        model = RelativeBiasModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'noBiasLite':
        model = NoBiasModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == '2D':
        model = PanguLite2D(device=device, dim=int(1.5*dim), patch_size=params['patch_size'][1:])
    elif params['model'] == 'threeLayer':
        model = ThreeLayerModel(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == 'positionEmbedding':
        model = PositionalEmbedding(device=device, dim=dim, patch_size=params['patch_size'])
    elif params['model'] == '2Dim192':
        model = PanguLite2D(device=device, dim=int(dim), patch_size=params['patch_size'][1:])
    else: 
        raise NotImplementedError(params['model'] + ' model does not exist.')

    model = model.to(torch.float32).to(device)

    # DistributedDataParallel wrapper if GPUs are available
    if dist.is_initialized() and gpus_per_node > 0:
        model = DistributedDataParallel( # Wrap model with DistributedDataParallel.
                model, 
                device_ids=[slurm_localid], 
                output_device=slurm_localid,
                find_unused_parameters=False
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
    
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if params['restart']:
        if rank == 0:
            print("params['save_dir']", params['save_dir'] + str(params['save_counter']))
        state = torch.load(params['save_dir'] + str(params['save_counter']) + '_' + params['model'] + params['hash'] + '.pt', map_location='cpu')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        optimizer.param_groups[0]['lr'] = 2.5e-4
        start_epoch = state['epoch'] + 1
        model = model.to(device)
    else:
        start_epoch = 0

    # Training loop
    train_loss_history = []
    valid_loss_history = []
    best_val_loss = 100000           # Arbitrary large number

    loss1 = torch.nn.L1Loss()
    loss2 = torch.nn.L1Loss()

    if params['save_counter'] is not None and params['restart'] is True:
        save_counter = params['save_counter'] 
    else:
        save_counter = 0

    # Z, Q, T, U, V
    total_weights = np.zeros((num_epochs, 4))
    n_cycles = 2
    interval = 2*np.pi*n_cycles
    fraction = interval / 4 / n_cycles

    for i in range(4):
        total_weights[:, i] = np.cos(np.linspace(-fraction*i, interval- fraction*i, num_epochs)) + 1.1
    
    # Z, Q, T, U, V
    pressure_weights = total_weights[:, [1, 3, 2, 0, 0]] / total_weights[:, [1, 3, 2, 0, 0]].sum(1, keepdims=1) * (3.00 + 0.6 + 1.5 + 0.77 + 0.54)
    # MSP, U, V, T
    surface_weights = total_weights[:, [1, 0, 0, 2]] / total_weights[:, [1, 0, 0, 2]].sum(1, keepdims=1) * (1.5 + 0.77 + 0.66 + 3.0)#
    
    scaler = GradScaler()

    early_stopping = 0 # tracks how many epochs have passed since the validation loss has improved

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if rank == 0 and epoch == start_epoch:
            lr = optimizer.param_groups[0]['lr'] # write to variable so that we can print...
            print("learning rate:", lr)
        start_epoch_time = time.perf_counter()
        model.train()
        
        train_sampler.set_epoch(epoch)
        epoch_average_loss = 0
        loss = 0
        for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
            input, input_surface, target, target_surface = data[0], data[1], data[2][0], data[3][0]
            input = input.to(torch.float32).to(device)
            input_surface = input_surface.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            target_surface = target_surface.to(torch.float32).to(device)

            with torch.autocast(device_type="cuda"):
                optimizer.zero_grad()
                # Call the model and get the output
                output, output_surface = model(input, input_surface)
                if params['model'] == '2D' or params['model'] == '2Dim192':
                    output = output.reshape(-1, 5, 13, output.shape[-2], output.shape[-1])
                    target = target.reshape(-1, 5, 13, target.shape[-2], target.shape[-1])

                # We use the MAE loss to train the model
                pressure_weights_epoch = torch.tensor(pressure_weights[epoch - start_epoch]).to(torch.float32).view(1, 5, 1, 1, 1).to(device)
                surface_weights_epoch  = torch.tensor(surface_weights[epoch - start_epoch]).to(torch.float32).view(1, 4, 1, 1).to(device)

                output, output_surface = output * pressure_weights_epoch, output_surface * surface_weights_epoch
                target, target_surface = target * pressure_weights_epoch, target_surface * surface_weights_epoch
                loss = 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            # Call the backward algorithm and calcula the gradient of parameters
            scaler.step(optimizer)
            scaler.update()
            
            
            dist.all_reduce(loss) # Allreduce rank-local mini-batch losses.
            
            if rank == 0:
                loss /= world_size    # Average allreduced rank-local mini-batch losses over all ranks.
                train_loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.
                epoch_average_loss += loss.item()
            
                print(f'Epoch: {epoch+1:03d}/{int(start_epoch+num_epochs):03d} '
                    f'| Batch {i:04d}/{len(train_data_loader):04d} '
                    f'| Averaged Loss: {loss:.4f}')
        
        epoch_average_loss /= len(train_data_loader)
        
        end_epoch_time = time.perf_counter()

        if rank == 0:
            print(f'Epoch: {int(epoch+1):03d}/{int(start_epoch+num_epochs):03d} '
                f'Elapsed training time: {end_epoch_time - start_epoch_time:04f}')
        
        model.eval()
        with torch.no_grad():
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            val_loss, mse, acc, total_samples, dt_validation = eval.get_loss(model, valid_data_loader, device, loss1, loss2, lat_crop=params['lat_crop'], lon_crop=params['lon_crop'], world_size=world_size)

            if rank == 0:
                valid_loss_history.append(val_loss[0])
                if (val_loss[0] < best_val_loss):
                    early_stopping = 0
                    best_val_loss = val_loss[0]
                    # save model to path
                    save_path = params['save_dir'] + str(save_counter) + "_" + params['model'] + params['hash'] + '.pt'
                    state = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, save_path)
                    save_counter += 1
                else:
                    early_stopping += 1
                
                lr = optimizer.param_groups[0]['lr'] # write to variable so that we can print...

                print(f'Epoch Average: {int(epoch)+1:03d}/{int(num_epochs):03d} ')
                print(f'| Mean L1 Training Loss: {epoch_average_loss :.5f} ')
                print(f'| Mean L1 Validation Loss: {val_loss[0] :.5f} ')
                print(f'| MSE T850: {mse[0][0] :.3f} ')
                print(f'| MSE Z500: {mse[1][0] :.3f} ')
                print(f'| MSE U850: {mse[2][0] :.3f} ')
                print(f'| MSE V850: {mse[3][0] :.3f} ')
                print(f'| MSE Q850: {mse[4][0]*1000 :.3f} ')
                print(f'| MSE T2M:  {mse[5][0] :.3f} ')
                print(f'| MSE U10:  {mse[6][0] :.3f} ')
                print(f'| MSE V10:  {mse[7][0] :.3f} ')
                print(f'| MSE MSL:  {mse[8][0] :.3f} ')
                print(f'| ACC T850: {acc[0][0] :.3f} ')
                print(f'| ACC Z500: {acc[1][0] :.3f} ')
                print(f'| ACC U850: {acc[2][0] :.3f} ')
                print(f'| ACC V850: {acc[3][0] :.3f} ')
                print(f'| ACC Q850: {acc[4][0] :.3f} ')
                print(f'| ACC T2M:  {acc[5][0] :.3f} ')
                print(f'| ACC U10:  {acc[6][0] :.3f} ')
                print(f'| ACC V10:  {acc[7][0] :.3f} ')
                print(f'| ACC MSL:  {acc[8][0] :.3f} ')
                print(f'| Validation time: {dt_validation : .5f}')
                print(f'| Total samples: {total_samples :.3f}')
                print(f'| Learning Rate: {lr :.7f}')
                print(f'| Pressure_weights: {pressure_weights_epoch[0][0].flatten()[0] :.3f}, {pressure_weights_epoch[0][1].flatten()[0] :.3f}, {pressure_weights_epoch[0][2].flatten()[0] :.3f}, {pressure_weights_epoch[0][3].flatten()[0] :.3f}, {pressure_weights_epoch[0][4].flatten()[0] :.3f}')
                print(f'| Surface_weights: {surface_weights_epoch[0][0].flatten()[0] :.3f}, {surface_weights_epoch[0][1].flatten()[0] :.3f}, {surface_weights_epoch[0][2].flatten()[0] :.3f}, {surface_weights_epoch[0][3].flatten()[0] :.3f}')

            torch.distributed.barrier()
                    
    # How can we verify that at least one model will be saved? Currently only saves when in case of the best validation loss
    dist.destroy_process_group()

if __name__ == '__main__':
    params = {}
    params['train_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['valid_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['pressure_static_data_path'] = 'constant_masks/pressure_zarr.npy' 
    params['surface_static_data_path'] =  'constant_masks/surface_zarr.npy'  
    params['dt'] = 24
    params['num_data_workers'] = 2
    params['data_distributed'] = True
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['num_epochs'] = 200
    num_epochs = params['num_epochs']
    params['C'] = 192
    params['subset_size'] = None # 100000
    params['validation_subset_size'] = None # 1500
    params['restart'] = True
    params['hash'] = "20240503_387727210"
    params['Lite'] = True
    params['daily'] = False
    params['save_counter'] = 83
    params['model_name'] = 'cos_weights/'

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
    base_save_dir = 'loss_schedule_exp/trained_models/'
        
    
    # Set seeds for reproducability
    #torch.backends.cudnn.benchmark = True # This can allegedly improve computational time by ~20%.
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    device, slurm_localid, gpus_per_node, rank, world_size = init_distributed(params)

    hash_key = params['hash']
    group = dist.new_group(list(range(world_size)))

    if rank == 0:
        if not params['restart']: # If this is a new run
            dt = datetime.datetime.now()
            date_string = dt.strftime("%Y%m%d")
            hash_key = date_string + "_" + str(abs(hash(datetime.datetime.now())))[:-10]
            
        elif params['hash'] is None:
            raise ValueError("If reloading from checkpoint, must specify a hash key to read from")
        else:
            hash_key = params['hash']

    if not params['restart']:
        hash_key_list = [hash_key]
        dist.broadcast_object_list(hash_key_list, src=0, group=group)
        hash_key = hash_key_list[0]
        params['hash'] = hash_key
    else: # this is kind of a dummy thing...
        hash_key_list = [hash_key]
        dist.broadcast_object_list(hash_key_list, src=0, group=group)
    

    params['save_dir'] = base_save_dir + params['model_name'] + hash_key + '/' 
            
    if rank == 0:
        print("Model will be saved in", params['save_dir'])

        # If starting a new run
        if not params['restart']:
            try:
                os.mkdir(params['save_dir'])
            except ValueError:
                raise ValueError("Could not create directory")
    
            # dump parameters into json directory
            with open(params['save_dir'] + "params_" + hash_key + '.json', 'w') as params_file:
                json.dump(params, params_file)
        
    # initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite. PS is (2, 4, 4) for all other sizes.
    if params['Lite']:
        params['patch_size'] = (2, 8, 8)
        if params['model'] == '2D' or params['model'] == '2Dim192':
            params['batch_size'] = 12
        else:
            params['batch_size'] = 6
        params['lat_crop']   = (3, 4)
        params['lon_crop']   = (0, 0)
    else:
        params['patch_size'] = (2, 4, 4)
        params['batch_size'] = 2
        params['lat_crop']   = (1, 2)
        params['lon_crop']   = (0, 0)

    if params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
        params['delta_T_divisor'] = 6
    elif params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
        params['delta_T_divisor'] = 1
#    if params['']

    training_loop(params, device, slurm_localid, gpus_per_node)
    