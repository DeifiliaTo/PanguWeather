import torch
from networks.pangu import PanguModel as PanguModel
from networks.PanguLite import PanguModel as PanguModelLite
from networks.relativeBias import PanguModel as RelativeBiasModel
from networks.noBias import PanguModel as NoBiasModel
from utils.data_loader_multifiles import get_data_loader
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import utils.eval as eval
import time, datetime, json

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

def training_loop(params, device, slurm_localid, gpus_per_node):
    # Define patch size, data loader, model
    C          = params['C']
    train_data_loader, train_dataset, train_sampler = get_data_loader(params, params['train_data_path'], dist.is_initialized(), mode='train', device=device, patch_size=params['patch_size'], subset_size=params['subset_size'])
    valid_data_loader, valid_dataset = get_data_loader(params, params['valid_data_path'], dist.is_initialized(), mode='validation', device=device, patch_size=params['patch_size'], subset_size=params['validation_subset_size'])

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params['num_epochs'])
    #scaler = GradScaler()
    start = time.perf_counter() # Measure training time.

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if params['restart']:
        print("params['save_dir'][params['model']]", params['save_dir'][params['model']])
        state = torch.load(params['save_dir'][params['model']] + str(params['save_counter']) + '_' + params['model'] + params['hash'] + '.pt')#, map_location=f'cuda:{rank}')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])

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
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).view(1, 5, 1, 1, 1).to(device)
    surface_weights  = torch.tensor([1.5, 0.77, 0.66, 3.0]).view(1, 4, 1, 1).to(device)

    for epoch in range(params['num_epochs']):
        start_epoch_time = time.perf_counter()
        model.train()
        
        train_sampler.set_epoch(epoch)
        epoch_average_loss = 0
        for i, data in enumerate(train_data_loader):        # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
            input, input_surface, target, target_surface = data[0], data[1], data[2][0], data[3][0]
            input = input.to(torch.float32).to(device)
            input_surface = input_surface.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            target_surface = target_surface.to(torch.float32).to(device)
            
            # Call the model and get the output
            output, output_surface = model(input, input_surface)

            # We use the MAE loss to train the model
            # The weight of surface loss is 0.25
            # Different weight can be applied for different fields if needed
            output, output_surface = output * pressure_weights, output_surface * surface_weights
            target, target_surface = target * pressure_weights, target_surface * surface_weights
            loss = 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
            # Call the backward algorithm and calcula the gradient of parameters
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            # Update model parameters with Adam optimizer
            # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
            #scaler.step(optimizer)
            #scaler.update()
            #print(optimizer.param_groups)
            
            dist.all_reduce(loss) # Allreduce rank-local mini-batch losses.
            
            if rank == 0:
                loss /= world_size    # Average allreduced rank-local mini-batch losses over all ranks.
                train_loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.
                epoch_average_loss += loss.item()
            
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                    f'| Batch {i:04d}/{len(train_data_loader):04d} '
                    f'| Averaged Loss: {loss:.4f}')
        scheduler.step()
        epoch_average_loss /= len(train_data_loader)
        
        end_epoch_time = time.perf_counter()

        if rank == 0:
            print(f'Epoch: {int(epoch+1):03d}/{int(num_epochs):03d} '
                f'Elapsed training time: {end_epoch_time - start_epoch_time:04f}')
        
        model.eval()
        with torch.no_grad():
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            val_loss, MSE, acc, total_samples, dt_validation = eval.get_loss(model, valid_data_loader, device, loss1, loss2, lat_crop=params['lat_crop'], lon_crop=params['lon_crop'], world_size=world_size)

            if rank == 0:
                valid_loss_history.append(val_loss[0])
                if (val_loss[0] < best_val_loss) or (epoch % 20 == 0):
                    best_val_loss = val_loss[0]
                    # save model to path
                    save_path = params['save_dir'][params['model']] + str(save_counter) + "_" + params['model'] + params['hash'] + '.pt'
                    state = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    }
                    torch.save(state, save_path)
                    save_counter += 1
                
                print(f'Epoch Average: {int(epoch)+1:03d}/{int(num_epochs):03d} ')
                print(f'| Mean L1 Training Loss: {epoch_average_loss :.3f} ')
                print(f'| Mean L1 Validation Loss: {val_loss[0] :.3f} ')
                print(f'| MSE T850: {MSE[0][0] :.3f} ')
                print(f'| MSE Z500: {MSE[1][0] :.3f} ')
                print(f'| MSE T2M:  {MSE[2][0] :.3f} ')
                print(f'| MSE U10:  {MSE[3][0] :.3f} ')
                print(f'| MSE V10:  {MSE[4][0] :.3f} ')
                print(f'| ACC T850: {acc[0][0] :.3f} ')
                print(f'| ACC Z500: {acc[1][0] :.3f} ')
                print(f'| ACC T2M:  {acc[2][0] :.3f} ')
                print(f'| ACC U10:  {acc[3][0] :.3f} ')
                print(f'| ACC V10:  {acc[4][0] :.3f} ')
                print(f'| Validation time: {dt_validation : .3f}')
                print(f'| Total samples: {total_samples :.3f}')
                    
    # How can we verify that at least one model will be saved? Currently only saves when in case of the best validation loss
    dist.destroy_process_group()

if __name__ == '__main__':
    params = {}
    params['train_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['valid_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy' 
    params['surface_static_data_path'] =  '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy'  
    params['dt'] = 24
    params['num_data_workers'] = 2
    params['data_distributed'] = True
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['num_epochs'] = 50
    num_epochs = params['num_epochs']
    params['C'] = 192
    params['subset_size'] = 4000
    params['validation_subset_size'] = 500
    params['restart'] = False
    params['hash'] = None # "20240326_" + str(673944995)
    params['Lite'] = True
    params['daily'] = False
    params['save_counter'] = None # 44

    # Specify model
    params['model'] = 'panguLite'
    # pangu        = full run
    # panguLite    = light model
    # relativeBias = relative bias
    # noBias       = no bias
    # 2D           = 2D transformer

    # Save directory
    base_save_dir = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/'
        
    
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
    

    params['save_dir'] = {
        'pangu':        base_save_dir + 'pangu/' + hash_key + '/',
        'panguLite':    base_save_dir + 'panguLite/' + hash_key + '/',
        'relativeBias': base_save_dir + 'relativeBiasLite/' + hash_key + '/',
        'noBias':       base_save_dir + 'noBiasLite/' + hash_key + '/',
        '2D':           base_save_dir + 'twoDimensionalLite/' + hash_key + '/'
    }
            
    if rank == 0:
        print("Model will be saved in", params['save_dir'][params['model']])

        # If starting a new run
        if not params['restart']:
            try:
                os.mkdir(params['save_dir'][params['model']])
            except:
                raise ValueError("Could not create directory")
    
            # dump parameters into json directory
            with open(params['save_dir'][params['model']] + "params_" + hash_key + '.json', 'w') as params_file:
                json.dump(params, params_file)
        
    # initialize patch size: currently, patch size is only (2, 8, 8) for PanguLite. PS is (2, 4, 4) for all other sizes.
    if params['Lite'] == True:
        params['patch_size'] = (2, 8, 8)
        params['batch_size'] = 5
        params['lat_crop']   = (3, 4)
        params['lon_crop']   = (0, 0)
    else:
        params['patch_size'] = (2, 4, 4)
        params['batch_size'] = 2
        params['lat_crop']   = (1, 2)
        params['lon_crop']   = (0, 0)

    training_loop(params, device, slurm_localid, gpus_per_node)
    