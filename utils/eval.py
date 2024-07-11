import time

import numpy as np
import torch


def get_loss(model, data_loader, device, loss1, loss2, lat_crop, lon_crop, world_size=4, pressure_weights_epoch=None, surface_weights_epoch=None):
    """
    Compute the mean loss of all samples in a given dataset.
    
    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    device : String
            device (cpu or gpu) that the code is running on #
    loss1 : torch.nn.L1Loss
            loss function for pressure levels
    loss2 : torch.nn.L2Loss
            loss function for surface levels
    lat_crop : Tuple(int, int)
            cropping along the latitude dimension of the data required to obtain original dimensions
            i.e., if the original image has size (721, 1440), and the patch size is (8, 8),
            the cropping will be (3, 4) for latitude.
    lon_crop : Tuple(int, int)
            cropping along the longitude dimension of the data required to obtain original dimensions
            i.e., if the original image has size (721, 1440), and the patch size is (8, 8),
            the cropping will be (0, 0) for longitude.
    world_size : int
            number of parallel processes, i.e., individual GPUs.
    pressure_weights_epoch: tuple(float, float, float, float, float)
            weighting factor for the pressure variables (Z, Q, T, U, V)
    surface_weights_epoch: tuple(float, float, float, float, float)
            weighting factor for the surface variables (MSLP, U10, V10, T2M)

    Returns
    -------
        l2_loss: float
                normalized MSE
        mse : Tuple(float)
                The mean squared errors of hard-coded atmospheric variables
        acc : Tuple(float)
                The anomaly correlation coefficient
        time : float
                time to run validation script
        num_samples : int
                total number of samples
    
    """
    loss = torch.zeros(1).to(device)

    static_plevel = np.load('constant_masks/pressure_zarr.npy')
    mean_plevel   = torch.tensor(static_plevel[0].reshape(1, 5, 13, 1, 1)).to(device)
    std_plevel    = torch.tensor(static_plevel[1].reshape(1, 5, 13, 1, 1)).to(device)

    static_slevel = np.load('constant_masks/surface_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(1, 4, 1, 1)).to(device)
    std_slevel    = torch.tensor(static_slevel[1].reshape(1, 4, 1, 1)).to(device)

    climatology_plevel = torch.tensor(np.load('constant_masks/pressure_climatology.npy')).to(device)
    climatology_slevel = torch.tensor(np.load('constant_masks/surface_climatology.npy')).to(device)

    climatology_plevel = climatology_plevel - mean_plevel
    climatology_slevel = climatology_slevel - mean_slevel

    n_lat = 721
    weights = torch.tensor(calc_weight(n_lat)).to(torch.float32).to(device)
    mse = {
        'T2M':  torch.zeros(1).to(device),
        'U10':  torch.zeros(1).to(device),
        'V10':  torch.zeros(1).to(device),
        'T850': torch.zeros(1).to(device),
        'Z500': torch.zeros(1).to(device),
        'U850': torch.zeros(1).to(device),
        'V850': torch.zeros(1).to(device),
        'Q850': torch.zeros(1).to(device),
        'MSL':  torch.zeros(1).to(device)
    }
    acc = {
        'T2M':  torch.zeros(1).to(device),
        'U10':  torch.zeros(1).to(device),
        'V10':  torch.zeros(1).to(device),
        'T850': torch.zeros(1).to(device),
        'Z500': torch.zeros(1).to(device),
        'U850': torch.zeros(1).to(device),
        'V850': torch.zeros(1).to(device),
        'Q850': torch.zeros(1).to(device),
        'MSL':  torch.zeros(1).to(device)
    }
    if pressure_weights_epoch is None:
        pressure_weights_epoch = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).view(1, 5, 1, 1, 1).to(device)
    if surface_weights_epoch is None:
        surface_weights_epoch  = torch.tensor([1.5, 0.77, 0.66, 3.0]).view(1, 4, 1, 1).to(device)

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
        # If there is a 2D model, we need to reshape the pressure level data back to 3 dimensionis
        if len(output.shape) == 4:
            output = output.reshape(-1, 5, 13, output.shape[-2], output.shape[-1])
            target = target.reshape(-1, 5, 13, target.shape[-2], target.shape[-1])
        
        # Multiply by the pressure and surface weights
        output, output_surface = output * pressure_weights_epoch, output_surface * surface_weights_epoch
        target, target_surface = target * pressure_weights_epoch, target_surface * surface_weights_epoch
        
        # MAE is used to train and evaluate the model.
        # Currently, pressure losses are weighted with 1, surface losses weighted with 0.25
        loss[0] += 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)

        # undo the pressure and surface weights for MSE and ACC metrics
        output, output_surface = output / pressure_weights_epoch, output_surface / surface_weights_epoch
        target, target_surface = target / pressure_weights_epoch, target_surface / surface_weights_epoch

        # Accumulate MSE and ACC for the data samples
        mse['T850'][0] += calc_mse(output, target, variable='T', pressure_level=850, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['Z500'][0] += calc_mse(output, target, variable='Z', pressure_level=500, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['T2M'][0]  += calc_mse(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['U10'][0]  += calc_mse(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['V10'][0]  += calc_mse(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['MSL'][0]  += calc_mse(output_surface, target_surface, variable='MSL', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['U850'][0] += calc_mse(output, target, variable='U', pressure_level=850, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['V850'][0] += calc_mse(output, target, variable='V', pressure_level=850, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        mse['Q850'][0] += calc_mse(output, target, variable='Q', pressure_level=850, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)

        acc['T850'][0] += calc_acc(output, target, variable='T', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['Z500'][0] += calc_acc(output, target, variable='Z', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['T2M'][0]  += calc_acc(output_surface, target_surface, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['U10'][0]  += calc_acc(output_surface, target_surface, variable='U10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['V10'][0]  += calc_acc(output_surface, target_surface, variable='V10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['MSL'][0]  += calc_acc(output_surface, target_surface, variable='MSL', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['U850'][0] += calc_acc(output, target, variable='U', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['V850'][0] += calc_acc(output, target, variable='V', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
        acc['Q850'][0] += calc_acc(output, target, variable='Q', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,  std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)

        # Accumulate total number of samples
        total_samples[0] += input.shape[0]
    
    # All-reduce operations
    torch.distributed.all_reduce(total_samples)
    torch.distributed.all_reduce(loss)

    for variable in ['T850', 'Z500', 'T2M', 'U10', 'V10', 'U850', 'V850', 'Q850', 'MSL']:
        torch.distributed.all_reduce(mse[variable])
        torch.distributed.all_reduce(acc[variable])

    end_validation_time = time.perf_counter()

    # Average loss values
    loss /= world_size # average the loss for world size
    loss /= len(data_loader)
    
    # take root and mean of MSE and take mean of ACC
    for variable in ['T850', 'Z500', 'T2M', 'U10', 'V10', 'U850', 'V850', 'Q850', 'MSL']:
        mse[variable] = torch.sqrt(mse[variable]/total_samples)
        acc[variable] = acc[variable] / total_samples

    return loss.tolist(), (mse['T850'].tolist(), mse['Z500'].tolist(), mse['U850'].tolist(), mse['V850'].tolist(), mse['Q850'].tolist(), mse['T2M'].tolist(), mse['U10'].tolist(), mse['V10'].tolist(), mse['MSL'].tolist()), (acc['T850'].tolist(), acc['Z500'].tolist(), acc['U850'].tolist(), acc['V850'].tolist(), acc['Q850'].tolist(), acc['T2M'].tolist(), acc['U10'].tolist(), acc['V10'].tolist(), acc['MSL'].tolist()), total_samples.item(), end_validation_time - start_validation_time


def get_validation_loss(model, data_loader, device, lat_crop, lon_crop, world_size=4, forecast_length=5):
    """
    Compute the mean loss of all samples in a given dataset.
    
    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    device : String
            device (cpu or gpu) that the code is running on #
    lat_crop : Tuple(int, int)
            cropping along the latitude dimension of the data required to obtain original dimensions
            i.e., if the original image has size (721, 1440), and the patch size is (8, 8),
            the cropping will be (3, 4) for latitude.
    lon_crop : Tuple(int, int)
            cropping along the longitude dimension of the data required to obtain original dimensions
            i.e., if the original image has size (721, 1440), and the patch size is (8, 8),
            the cropping will be (0, 0) for longitude.
    world_size : int
            number of parallel processes, i.e., individual GPUs.
    forecast_length: int
            number of auto-regressive forecast steps to be run

    Returns
    -------
        mse : Tuple(float)
                The mean squared errors of hard-coded atmospheric variables
        acc : Tuple(float)
                The anomaly correlation coefficient
        num_samples : int
                total number of samples
        time : float
                time to run validation script
    """
    loss = 0
    
    static_plevel = np.load('constant_masks/pressure_zarr.npy')
    mean_plevel   = torch.tensor(static_plevel[0].reshape(1, 5, 13, 1, 1)).to(device)
    std_plevel    = torch.tensor(static_plevel[1].reshape(1, 5, 13, 1, 1)).to(device)

    static_slevel = np.load('constant_masks/surface_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(1, 4, 1, 1)).to(device)
    std_slevel    = torch.tensor(static_slevel[1].reshape(1, 4, 1, 1)).to(device)

    climatology_plevel = torch.tensor(np.load('constant_masks/pressure_climatology.npy')).to(device)
    climatology_slevel = torch.tensor(np.load('constant_masks/surface_climatology.npy')).to(device)

    climatology_plevel = climatology_plevel - mean_plevel
    climatology_slevel = climatology_slevel - mean_slevel

    n_lat = 721
    weights = torch.tensor(calc_weight(n_lat, np.linspace(np.pi/2.0, -np.pi/2.0, n_lat))).to(torch.float32).to(device)
    mse = {
        'T2M':  torch.zeros(forecast_length).to(device),
        'U10':  torch.zeros(forecast_length).to(device),
        'V10':  torch.zeros(forecast_length).to(device),
        'T850': torch.zeros(forecast_length).to(device),
        'Z500': torch.zeros(forecast_length).to(device)
    }
    acc = {
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

            mse['T850'][j]  += calc_mse(output, target_, variable='T', pressure_level=850, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            mse['Z500'][j]  += calc_mse(output, target_, variable='Z', pressure_level=500, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            mse['T2M'][j]   += calc_mse(output_surface, target_surface_, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            mse['U10'][j]   += calc_mse(output_surface, target_surface_, variable='U10', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            mse['V10'][j]   += calc_mse(output_surface, target_surface_, variable='V10', upper_variable=False, pressure_level=None, weights=weights, std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            acc['T850'][j] += calc_acc(output, target_, variable='T', pressure_level=850, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,   std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            acc['Z500'][j] += calc_acc(output, target_, variable='Z', pressure_level=500, weights=weights, climatology_plevel=climatology_plevel, climatology_slevel=climatology_slevel,   std_plevel=std_plevel, std_slevel=std_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            acc['T2M'][j]  += calc_acc(output_surface, target_surface_, variable='T2M', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            acc['U10'][j]  += calc_acc(output_surface, target_surface_, variable='U10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)
            acc['V10'][j]  += calc_acc(output_surface, target_surface_, variable='V10', upper_variable=False, pressure_level=None, weights=weights, climatology_plevel=climatology_plevel, std_plevel=std_plevel, std_slevel=std_slevel, climatology_slevel=climatology_slevel, lat_crop=lat_crop, lon_crop=lon_crop)

            # autoregressive forecast means output is new input
            input, input_surface = output, output_surface
        total_samples[0] += input.shape[0]

    torch.distributed.all_reduce(total_samples)

    # Reduce all values on each core to get total SE
    for variable in ['T850', 'Z500', 'T2M', 'U10', 'V10']:
        torch.distributed.all_reduce(mse[variable])
        torch.distributed.all_reduce(acc[variable])
    
    end_validation_time = time.perf_counter()

    # find total number of data points
    total_samples = len(data_loader.dataset) - len(data_loader.dataset) % input.shape[0] 
    # average the loss for world size
    loss /= world_size 
    
    # Take square root of MSE
    for variable in ['T850', 'Z500', 'T2M', 'U10', 'V10']:
        mse[variable]  = torch.sqrt(mse[variable]/total_samples)
        acc[variable] = acc[variable] / total_samples

    return  (mse['T850'].tolist(), mse['Z500'].tolist(), mse['T2M'].tolist(), mse['U10'].tolist(), mse['V10'].tolist()), (acc['T850'].tolist(), acc['Z500'].tolist(), acc['T2M'].tolist(), acc['U10'].tolist(), acc['V10'].tolist()), total_samples, end_validation_time - start_validation_time


def calc_weight(n_lat, cossum=458.36551167):
    """
    Return latitude-weighted values for loss function.

    n_lat: int
            number of patches in the latitude dimension
    
    Returns
    -------
    weight: np.array(float)
        latitude_based weighting factor
    """
    latitude = np.linspace(np.pi/2.0, -np.pi/2.0, n_lat)
    weight = n_lat * np.cos(latitude) / cossum
    return weight

# upper_variable options: [Z, Q, T, U, V]
# surface_variable options: [MSLP, U10, V10, T2M]
# PL options: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
upper_variable_indexing = {'Z': 0, 'Q': 1, 'T': 2, 'U': 3, 'V': 4}
surface_variable_indexing = {'MSL': 0, 'U10': 1, 'V10': 2, 'T2M': 3}
PL_indexing = {1000: 0, 925: 1, 850: 2, 700: 3, 600: 4, 500: 5, 400: 6, 300: 7, 250: 8, 200: 9, 150: 10, 100: 11, 50: 12}

def calc_mse(result_values, target_values, variable, pressure_level=500, weights=None, n_lat=721, n_lon=1440, upper_variable=True, std_plevel=None, std_slevel=None, lat_crop=(3,4), lon_crop=(0,0)):
    """
    Calculate the MSE, returned in relevant standard units (K, m/s, etc.).

    result_values: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
        n_vert default: 14
    target_values: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
        n_vert default: 14
    variable: String
        key word specifying the variable
    pressure_level: int
        pressure level that MSE should be performed on
    weights: Tensor
        latitude_based weighting factor of length n_lat
    n_lat: int
        number of pixels in the lat dimension in the original image
        default: 721
    n_lon: int
        number of pixels in the lat dimension in the original image
        default: 1440
    upper_variable: bool
        True if mse is calculated for a pressure variable
        False if mse is calculated for a surface variable
    std_pLevel: Tensor
        of shape (1, n_fields, n_pressure_levels, 1, 1)
        n_pressure_levels default: 13
    std_slevel: Tensor
        of shape (1, n_surface_fieelds, 1, 1)
    lat_crop: Tuple(int, int)
        cropping applied to the left and right dimensions to obtain the original size of the image
        if image size if (721, 1440) and patch_size is (2, 4, 4), lat_crop = (1, 2)
        if image size if (721, 1440) and patch_size is (2, 8, 8), lat_crop = (3, 4)
    lon_crop: Tuple(int, int)
        cropping applied to the top and bottom dimensions to obtain the original size of the image
        if image size if (721, 1440) and patch_size is (2, 4, 4), lon_crop = (0, 0)
        if image size if (721, 1440) and patch_size is (2, 8, 8), lon_crop = (0, 0)

    Returns
    -------
    mean_squared_error: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
    """
    divisor = torch.sqrt(torch.tensor([n_lon*n_lat])).to(result_values.device)
    # MSE = ((target - result) * stdev) ** 2
    if upper_variable:
        mean_squared_error = ((target_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, upper_variable_indexing[variable], PL_indexing[pressure_level], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None]) * 
              std_plevel[0,upper_variable_indexing[variable], PL_indexing[pressure_level]].flatten() / divisor
              )**2
    else:
        mean_squared_error = ((target_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None] - 
              result_values[:, surface_variable_indexing[variable], lat_crop[0]:-lat_crop[1] or None, lon_crop[0]:-lon_crop[1] or None]) * 
              std_slevel[0, surface_variable_indexing[variable]].flatten() / divisor
              )**2
    mean_squared_error = torch.einsum('i, aij->ai', weights, mean_squared_error).sum(dim=1)
    mean_squared_error = mean_squared_error.flatten().sum() 
    
    return mean_squared_error.item()

def calc_acc(result_values, target_values, variable, pressure_level, weights=None, n_lat=721, n_lon=1440, upper_variable=True, climatology_plevel=None, climatology_slevel=None, std_plevel=None, std_slevel=None, lat_crop=(3,4), lon_crop=(0,0)):
    """
    Calculate the ACC, returned in relevant standard units (K, m/s, etc.).

    result_values: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
        n_vert default: 14
    target_values: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
        n_vert default: 14
    variable: String
        key word specifying the variable
    pressure_level: int
        pressure level that MSE should be performed on
    weights: Tensor
        of length n_lat
    n_lat: int
        number of pixels in the lat dimension in the original image
        default: 721
    n_lon: int
        number of pixels in the lat dimension in the original image
        default: 1440
    upper_variable: bool
        True if mse is calculated for a pressure variable
        False if mse is calculated for a surface variable
    std_pLevel: Tensor
        of shape (1, n_fields, n_pressure_levels, 1, 1)
        n_pressure_levels default: 13
    std_slevel: Tensor
        of shape (1, n_surface_fieelds, 1, 1)
    lat_crop: Tuple(int, int)
        cropping applied to the left and right dimensions to obtain the original size of the image
        if image size if (721, 1440) and patch_size is (2, 4, 4), lat_crop = (1, 2)
        if image size if (721, 1440) and patch_size is (2, 8, 8), lat_crop = (3, 4)
    lon_crop: Tuple(int, int)
        cropping applied to the top and bottom dimensions to obtain the original size of the image
        if image size if (721, 1440) and patch_size is (2, 4, 4), lon_crop = (0, 0)
        if image size if (721, 1440) and patch_size is (2, 8, 8), lon_crop = (0, 0)

    Returns
    -------
    acc_sum: Tensor
        of shape (n_batch, n_fields, n_vert, n_lat, n_lon)
    """
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
