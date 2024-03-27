import torch
import numpy as np
import time

def get_loss(model, data_loader, device, loss1, loss2, lat_crop, lon_crop, world_size=4):
    """
    Compute the mean loss of all samples in a given dataset.
    
    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    
    Returns
    -------
    int : The number of correctly predicted samples.
    int : The overall number of samples in the dataset.
    """

    loss = torch.zeros(1).to(device)

    static_plevel = np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy')
    mean_plevel   = torch.tensor(static_plevel[0].reshape(1, 5, 13, 1, 1)).to(device)
    std_plevel    = torch.tensor(static_plevel[1].reshape(1, 5, 13, 1, 1)).to(device)

    static_slevel = np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(1, 4, 1, 1)).to(device)
    std_slevel    = torch.tensor(static_slevel[1].reshape(1, 4, 1, 1)).to(device)

    climatology_plevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/pressure/pressure_climatology.npy')).to(device)
    climatology_slevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/surface/surface_climatology.npy')).to(device)

    climatology_plevel = climatology_plevel - mean_plevel
    climatology_slevel = climatology_slevel - mean_slevel

    Nlat = 721
    weights = torch.tensor(calc_weight(Nlat, np.linspace(np.pi/2.0, -np.pi/2.0, Nlat))).to(torch.float32).to(device)
    SE = {
        'T2M':  torch.zeros(1).to(device),
        'U10':  torch.zeros(1).to(device),
        'V10':  torch.zeros(1).to(device),
        'T850': torch.zeros(1).to(device),
        'Z500': torch.zeros(1).to(device)
    }
    ACC = {
        'T2M':  torch.zeros(1).to(device),
        'U10':  torch.zeros(1).to(device),
        'V10':  torch.zeros(1).to(device),
        'T850': torch.zeros(1).to(device),
        'Z500': torch.zeros(1).to(device)
    }
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).view(1, 5, 1, 1, 1).to(device)
    surface_weights  = torch.tensor([1.5, 0.77, 0.66, 3.0]).view(1, 4, 1, 1).to(device)

    start_validation_time = time.perf_counter()
    total_samples = torch.tensor([0]).to(device) # count samples on each processor

    for i, data in enumerate(data_loader):
    
        input, input_surface, target, target_surface = data[0], data[1], data[2][0], data[3][0]
        input = input.to(torch.float32).to(device)
        input_surface = input_surface.to(torch.float32).to(device)
        target = target.to(torch.float32).to(device)
        target_surface = target_surface.to(torch.float32).to(device)

        # Call the model and get the output
        output, output_surface = model(input, input_surface)
        output, output_surface = output * pressure_weights, output_surface * surface_weights
        target, target_surface = target * pressure_weights, target_surface * surface_weights
        
        # We use the MAE loss to train the model
        # The weight of surface loss is 0.25
        # Different weight can be applied for different fields if needed
        loss[0] += 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)

        output, output_surface = output / pressure_weights, output_surface / surface_weights
        target, target_surface = target / pressure_weights, target_surface / surface_weights


        SE['T850'][0] += calc_SE(output, target, variable='T', pressure_level=850, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['Z500'][0] += calc_SE(output, target, variable='Z', pressure_level=500, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['T2M'][0] += calc_SE(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['U10'][0] += calc_SE(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['V10'][0] += calc_SE(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['T850'][0] += calc_ACC(output, target, variable='T', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['Z500'][0] += calc_ACC(output, target, variable='Z', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['T2M'][0] += calc_ACC(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['U10'][0] += calc_ACC(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['V10'][0] += calc_ACC(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)

        total_samples[0] += input.shape[0]
    
    torch.distributed.all_reduce(total_samples)
    torch.distributed.all_reduce(loss)
    torch.distributed.all_reduce(SE['T850'])
    torch.distributed.all_reduce(SE['Z500'])
    torch.distributed.all_reduce(SE['T2M'])
    torch.distributed.all_reduce(SE['U10'])
    torch.distributed.all_reduce(SE['V10'])
    torch.distributed.all_reduce(ACC['T850'])
    torch.distributed.all_reduce(ACC['Z500'])
    torch.distributed.all_reduce(ACC['T2M'])
    torch.distributed.all_reduce(ACC['U10'])
    torch.distributed.all_reduce(ACC['V10'])

    end_validation_time = time.perf_counter()


    loss /= world_size # average the loss for world size
    
    # Reduce all values on each core
    SE['T850'] = torch.sqrt(SE['T850']/total_samples)
    SE['Z500'] = torch.sqrt(SE['Z500']/total_samples)
    SE['T2M']  = torch.sqrt(SE['T2M']/total_samples)
    SE['U10']  = torch.sqrt(SE['U10']/total_samples)
    SE['V10']  = torch.sqrt(SE['V10']/total_samples)

    ACC['T850'] /= total_samples
    ACC['Z500'] /= total_samples
    ACC['T2M'] /= total_samples
    ACC['U10'] /= total_samples
    ACC['V10'] /= total_samples

    return loss.tolist(), (SE['T850'].tolist(), SE['Z500'].tolist(), SE['T2M'].tolist(), SE['U10'].tolist(), SE['V10'].tolist()), (ACC['T850'].tolist(), ACC['Z500'].tolist(), ACC['T2M'].tolist(), ACC['U10'].tolist(), ACC['V10'].tolist()), total_samples.item(), end_validation_time - start_validation_time


def get_validation_loss(model, data_loader, device, loss1, loss2, lat_crop, lon_crop, world_size=4, forecast_length=5):
    """
    Compute the mean loss of all samples in a given dataset.
    
    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    
    Returns
    -------
    int : The number of correctly predicted samples.
    int : The overall number of samples in the dataset.
    """

    loss, num_examples = 0, 0
    rmse = 0
    static_plevel = np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy')
    mean_plevel   = torch.tensor(static_plevel[0].reshape(1, 5, 13, 1, 1)).to(device)
    std_plevel    = torch.tensor(static_plevel[1].reshape(1, 5, 13, 1, 1)).to(device)

    static_slevel = np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(1, 4, 1, 1)).to(device)
    std_slevel    = torch.tensor(static_slevel[1].reshape(1, 4, 1, 1)).to(device)

    climatology_plevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/pressure/pressure_climatology.npy')).to(device)
    climatology_slevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/surface/surface_climatology.npy')).to(device)

    climatology_plevel = climatology_plevel - mean_plevel
    climatology_slevel = climatology_slevel - mean_slevel

    Nlat = 721
    weights = torch.tensor(calc_weight(Nlat, np.linspace(np.pi/2.0, -np.pi/2.0, Nlat))).to(torch.float32).to(device)
    SE = {
        'T2M':  torch.zeros(forecast_length).to(device),
        'U10':  torch.zeros(forecast_length).to(device),
        'V10':  torch.zeros(forecast_length).to(device),
        'T850': torch.zeros(forecast_length).to(device),
        'Z500': torch.zeros(forecast_length).to(device)
    }
    ACC = {
        'T2M':  torch.zeros(forecast_length).to(device),
        'U10':  torch.zeros(forecast_length).to(device),
        'V10':  torch.zeros(forecast_length).to(device),
        'T850': torch.zeros(forecast_length).to(device),
        'Z500': torch.zeros(forecast_length).to(device)
    }
    start_validation_time = time.perf_counter()
    
    total_samples = torch.tensor([0]).to(device) # count samples on each processor

    for i, data in enumerate(data_loader):
        if torch.distributed.get_rank() == 0:
            print(i)
        input, input_surface, target, target_surface = data[0], data[1], data[2], data[3]
        ## Offload to GPU
        input = input.to(torch.float32).to(device)
        input_surface = input_surface.to(torch.float32).to(device)
        
        
        for j in range(forecast_length):
            target_ = target[j].to(torch.float32).to(device)
            target_surface_ = target_surface[j].to(torch.float32).to(device)
            # offload target data to GPU
            # Call the model and get the output
            output, output_surface = model(input, input_surface)

            SE['T850'][j]  += calc_SE(output, target_, variable='T', pressure_level=850, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            SE['Z500'][j]  += calc_SE(output, target_, variable='Z', pressure_level=500, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            SE['T2M'][j]   += calc_SE(output_surface, target_surface_, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            SE['U10'][j]   += calc_SE(output_surface, target_surface_, variable='U10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            SE['V10'][j]   += calc_SE(output_surface, target_surface_, variable='V10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            ACC['T850'][j] += calc_ACC(output, target_, variable='T', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,   std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            ACC['Z500'][j] += calc_ACC(output, target_, variable='Z', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,   std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            ACC['T2M'][j]  += calc_ACC(output_surface, target_surface_, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            ACC['U10'][j]  += calc_ACC(output_surface, target_surface_, variable='U10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            ACC['V10'][j]  += calc_ACC(output_surface, target_surface_, variable='V10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)

            # autoregressive forecast means output is new input
            input, input_surface = output, output_surface
        total_samples[0] += input.shape[0]

    torch.distributed.all_reduce(total_samples)
    torch.distributed.all_reduce(SE['T850'])
    torch.distributed.all_reduce(SE['Z500'])
    torch.distributed.all_reduce(SE['T2M'])
    torch.distributed.all_reduce(SE['U10'])
    torch.distributed.all_reduce(SE['V10'])
    torch.distributed.all_reduce(ACC['T850'])
    torch.distributed.all_reduce(ACC['Z500'])
    torch.distributed.all_reduce(ACC['T2M'])
    torch.distributed.all_reduce(ACC['U10'])
    torch.distributed.all_reduce(ACC['V10'])

    end_validation_time = time.perf_counter()

    total_samples = len(data_loader.dataset) - len(data_loader.dataset) % input.shape[0] # find total number of data points
    loss /= world_size # average the loss for world size
    # Reduce all values on each core
    SE['T850'] = torch.sqrt(SE['T850']/total_samples)
    SE['Z500'] = torch.sqrt(SE['Z500']/total_samples)
    SE['T2M']  = torch.sqrt(SE['T2M']/total_samples)
    SE['U10']  = torch.sqrt(SE['U10']/total_samples)
    SE['V10']  = torch.sqrt(SE['V10']/total_samples)
    
    ACC['T850'] /= total_samples
    ACC['Z500'] /= total_samples
    ACC['T2M'] /= total_samples
    ACC['U10'] /= total_samples
    ACC['V10'] /= total_samples

    return  (SE['T850'].tolist(), SE['Z500'].tolist(), SE['T2M'].tolist(), SE['U10'].tolist(), SE['V10'].tolist()), (ACC['T850'].tolist(), ACC['Z500'].tolist(), ACC['T2M'].tolist(), ACC['U10'].tolist(), ACC['V10'].tolist()), total_samples, end_validation_time - start_validation_time


def calc_weight(Nlat, latitude, cossum=458.36551167):
    # Nlat = number of total latitude points
    # latitude = degree of latitude in radians
    return Nlat * np.cos(latitude) / cossum

# upper_variable options: [Z, Q, T, U, V]
# surface_variable options: [MSLP, U10, V10, T2M]
# PL options: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
upper_variable_indexing = {'Z': 0, 'Q': 1, 'T': 2, 'U': 3, 'V': 4}
surface_variable_indexing = {'MSLP': 0, 'U10': 1, 'V10': 2, 'T2M': 3}
# TODO: check the padding indexing!
PL_indexing = {1000: 0, 925: 1, 850: 2, 700: 3, 600: 4, 500: 5, 400: 6, 300: 7, 250: 8, 200: 9, 150: 10, 100: 11, 50: 12}

# TODO: change order and add appropriate template/auto var for mean and std_plevel
def calc_SE(result_values, target_values, variable, pressure_level=500, weights=None, Nlat=721, Nlon=1440, upper_variable=True, mean_plevel=None, std_plevel=None, mean_slevel=None, std_slevel=None, lat_crop=(3,4), lon_crop=(0,0)):
    divisor = torch.sqrt(torch.tensor([1440.0*721])).to(result_values.device)
    # results_values shape: [batch_size, 5 variables, 14 pressure levels, Nlat, Nlon]
    if upper_variable:
        L2 = ((target_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None]) * 
              std_plevel[0,upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten() / divisor
              )**2
    else:
        L2 = ((target_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None]) * 
              std_slevel[0, surface_variable_indexing[variable]].flatten() / divisor
              )**2
    SE = torch.einsum('i, aij->ai', weights, L2).sum(dim=1)
    SE = SE.flatten().sum() 
    
    return SE.item()

def calc_ACC(result_values, target_values, variable, pressure_level, weights=None, Nlat=721, Nlon=1440, upper_variable=True, climatology_plevel=None, climatology_slevel=None, std_plevel=None, std_slevel=None, lat_crop=(3,4), lon_crop=(0,0)):
    # results_values shape: [batch_size, 5 variables, 14 pressure levels, Nlat, Nlon]
    if upper_variable:
        target_anomoly = (target_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] *
                          std_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten() - 
                          climatology_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]])
        result_anomoly = (result_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] *
                          std_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten() - 
                          climatology_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]])
        numerator = torch.sum(torch.einsum('i,aij->ai', weights, torch.mul(target_anomoly, result_anomoly)), dim=(1))
        denom1    = torch.sum(torch.einsum('i,aij->ai', weights, target_anomoly**2), dim=(1))
        denom2    = torch.sum(torch.einsum('i,aij->ai', weights, result_anomoly**2), dim=(1))
        acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
        acc_sum   = torch.sum(acc_vec)
    else:
        target_anomoly = (target_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] *
                          std_slevel[0, surface_variable_indexing[variable]].flatten() - 
                          climatology_slevel[0, surface_variable_indexing[variable]])
        result_anomoly = (result_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] *
                          std_slevel[0, surface_variable_indexing[variable]].flatten() - 
                          climatology_slevel[0, surface_variable_indexing[variable]])
        numerator = torch.sum(torch.einsum('i,aij->ai', weights, torch.mul(target_anomoly, result_anomoly)), dim=(1))
        denom1    = torch.sum(torch.einsum('i,aij->ai', weights, target_anomoly**2), dim=(1))
        denom2    = torch.sum(torch.einsum('i,aij->ai', weights, result_anomoly**2), dim=(1))
        acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
        acc_sum   = torch.sum(acc_vec)
        
    return acc_sum.item()
