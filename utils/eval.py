import torch
import numpy as np


def get_loss(model, data_loader, device, loss1, loss2):
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
    with torch.no_grad():

        loss, num_examples = 0, 0
        rmse = 0

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
            with torch.cuda.amp.autocast():
                loss += 1 * loss1(output, target) + 0.25 * loss2(output_surface, target_surface)
            num_examples += 1
        loss /= len(data_loader)
            
    return loss

def calc_weight(Nlat, latitude, cossum=648.9345973030014):
    # Nlat = number of total latitude points
    # latitude = degree of latitude in radians
    return Nlat * np.cos(latitude) / cossum

def calc_SE(result_values, data_values, variable, pressure_level, var, pl, lats, Nlat=721, Nlon=1440):
    SE    = 0
    for index, row in enumerate(result_values[0, variable[var], pressure_level[pl], 1:-1]):
        weight = calc_weight(Nlat, lats[index - 1])
        diff = (row - data_values[0, variable[var], pressure_level[pl], index].detach().numpy())**2
        SE += weight * diff.sum()
    RMSE /= (Nlat * Nlon)
    RMSE = np.sqrt(RMSE)
    
    return SE

def calc_ACC(result_values, data_values, variable, pressure_level, var, pl, Nlat, lats, mean_plevel):
    ACC   = 0
    denom1 = 0
    denom2 = 0
    for index, row in enumerate(result_values[0, variable[var], pressure_level[pl], 1:-1]):
        weight   = calc_weight(Nlat, lats[index - 1])
        fc_delta   = row - mean_plevel[0, variable[var], pressure_level[pl], 0, 0]
        era5_delta = data_values[0, variable[var], pressure_level[pl], index] - mean_plevel[0, variable[var], pressure_level[pl], 0, 0]
        ACC += weight * (era5_delta * fc_delta).sum()
        denom1 += (weight * fc_delta * fc_delta).sum()
        denom2 += (weight * era5_delta * era5_delta).sum()
    ACC = ACC / (np.sqrt(denom1 * denom2))
    return ACC