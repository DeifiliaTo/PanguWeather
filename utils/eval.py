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

    climatology_plevel = ((climatology_plevel - mean_plevel)/std_plevel)
    climatology_slevel = ((climatology_slevel - mean_slevel)/std_slevel)

    Nlat = 721
    weights = torch.tensor(calc_weight(Nlat, np.deg2rad(np.linspace(np.pi/2.0, -np.pi/2.0, Nlat)))).unsqueeze(0).to(torch.float32).to(device)
    SE = {
        'T2M': 0,
        'U10': 0,
        'V10': 0,
        'T500': 0,
        'Z500': 0
    }
    ACC = {
        'T2M': 0,
        'U10': 0,
        'V10': 0,
        'T500': 0,
        'Z500': 0
    }
    start_validation_time = time.perf_counter()
    
    for i, data in enumerate(data_loader):
    
        input, input_surface, target, target_surface = data[0], data[1], data[2], data[3]
        input = input.to(torch.float32).to(device)
        input_surface = input_surface.to(torch.float32).to(device)
        target = target.to(torch.float32).to(device)
        target_surface = target_surface.to(torch.float32).to(device)

        # Call the model and get the output
        output, output_surface = model(input, input_surface)
            
        # We use the MAE loss to train the model
        # The weight of surface loss is 0.25
        # Different weight can be applied for different fields if needed
        loss += 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)


        SE['T500'] += calc_SE(output, target, variable='T', pressure_level=500, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['Z500'] += calc_SE(output, target, variable='Z', pressure_level=500, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['T2M'] += calc_SE(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['U10'] += calc_SE(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        SE['V10'] += calc_SE(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, mean_plevel=mean_plevel, std_plevel=std_plevel, mean_slevel=mean_slevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['T500'] += calc_ACC(output, target, variable='T', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['Z500'] += calc_ACC(output, target, variable='Z', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['T2M'] += calc_ACC(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['U10'] += calc_ACC(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        ACC['V10'] += calc_ACC(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)


    torch.distributed.all_reduce(loss)
    torch.distributed.all_reduce(SE['T500'])
    torch.distributed.all_reduce(SE['Z500'])
    torch.distributed.all_reduce(SE['T2M'])
    torch.distributed.all_reduce(SE['U10'])
    torch.distributed.all_reduce(SE['V10'])
    torch.distributed.all_reduce(ACC['T500'])
    torch.distributed.all_reduce(ACC['Z500'])
    torch.distributed.all_reduce(ACC['T2M'])
    torch.distributed.all_reduce(ACC['U10'])
    torch.distributed.all_reduce(ACC['V10'])

    end_validation_time = time.perf_counter()


    total_samples = len(data_loader.dataset) # find total number of data points
    loss /= world_size # average the loss for world size
    
    # Reduce all values on each core
    SE['T500'] = torch.sqrt(SE['T500']/total_samples)
    SE['Z500'] = torch.sqrt(SE['Z500']/total_samples)
    SE['T2M']  = torch.sqrt(SE['T2M']/total_samples)
    SE['U10']  = torch.sqrt(SE['U10']/total_samples)
    SE['V10']  = torch.sqrt(SE['V10']/total_samples)

    ACC['T500'] /= total_samples
    ACC['Z500'] /= total_samples
    ACC['T2M'] /= total_samples
    ACC['U10'] /= total_samples
    ACC['V10'] /= total_samples

    return loss, (SE['T500'].item(), SE['Z500'].item(), SE['T2M'].item(), SE['U10'].item(), SE['V10'].item()), (ACC['T500'].item(), ACC['Z500'].item(), ACC['T2M'].item(), ACC['U10'].item(), ACC['V10'].item()), total_samples, end_validation_time - start_validation_time

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
    SE    = 0
    # results_values shape: [batch_size, 5 variables, 14 pressure levels, Nlat, Nlon]
    if upper_variable:
        L2 = (target_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None])**2
    else:
        L2 = (target_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None])**2

    SE = torch.matmul(weights, L2/Nlon/Nlat).flatten().sum()
    if upper_variable:
        SE = (SE * mean_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten() +
                   std_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten())

    else: #TODO fill in surface level
        SE = (SE * mean_slevel[0, surface_variable_indexing[variable]].flatten() +
                   std_slevel[0, surface_variable_indexing[variable]].flatten())
    return SE

def calc_ACC(result_values, target_values, variable, pressure_level, weights=None, Nlat=721, Nlon=1440, upper_variable=True, climatology_plevel=None, climatology_slevel=None, lat_crop=(0,1), lon_crop=(0,0)):
    # results_values shape: [batch_size, 5 variables, 14 pressure levels, Nlat, Nlon]
    if upper_variable:
        target_anomoly = (target_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
                          climatology_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]])
        result_anomoly = (result_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
                          climatology_plevel[0, upper_variable_indexing[variable], PL_indexing[pressure_level]])
        numerator = torch.sum(torch.matmul(weights, torch.mul(target_anomoly, result_anomoly)), dim=(1, 2))
        denom1    = torch.sum(torch.matmul(weights, target_anomoly**2), dim=(1, 2))
        denom2    = torch.sum(torch.matmul(weights, result_anomoly**2), dim=(1, 2))
        acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
        acc_sum   = torch.sum(acc_vec)
    else:
        target_anomoly = (target_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
                          climatology_slevel[0, surface_variable_indexing[variable]])
        result_anomoly = (result_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
                          climatology_slevel[0, surface_variable_indexing[variable]])
        numerator = torch.sum(torch.matmul(weights, torch.mul(target_anomoly, result_anomoly)), dim=(1, 2))
        denom1    = torch.sum(torch.matmul(weights, target_anomoly**2), dim=(1, 2))
        denom2    = torch.sum(torch.matmul(weights, result_anomoly**2), dim=(1, 2))
        acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
        acc_sum   = torch.sum(acc_vec)
        
    return acc_sum
