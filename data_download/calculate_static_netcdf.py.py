import glob
import random

import numpy as np
import xarray as xr

# List of files
file_list_plevel  = glob.glob('/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/*_pl.nc') 
file_list_surface = glob.glob('/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/*_sfc.nc') 

# Initialize variables for calculation
mean_sum_plevel = None
sq_sum_plevel = None
count = 0

# Process each file in plevels
for file_name in file_list_plevel:
    
    print("processing file", file_name)
    data = xr.open_dataset(file_name)

    num_points = len(data.time)
    rng = random.randint(0, num_points-1)

    # isolate time point and convert to numpy array
    data = data.isel(time=rng)
    data = np.stack([data['Z'].values, data['Q'].values, data['T'].values, data['U'].values, data['V'].values], axis=0)
    data = data.reshape(1, 5, 13, 721, 1440)

    if mean_sum_plevel is None:
        # Initialize arrays for the first file
        mean_sum_plevel = np.zeros((1, data.shape[1], data.shape[2], 1, 1), dtype=np.float32)
        sq_sum_plevel   = np.zeros((1, data.shape[1], data.shape[2], 1, 1), dtype=np.float32)

    # Summation for mean and standard deviation
    mean_sum_plevel += np.mean(data, keepdims=True, axis = (0, 3, 4))[0]
    sq_sum_plevel += np.var(data, keepdims=True, axis = (0, 3, 4))[0]

# Shape of the pressure mean and std should be (5, 13, 1, 1)
# Shape of the surface mean and std should be (4, 1, 1)
mean_plevel = mean_sum_plevel / len(file_list_plevel)
std_plevel  = np.sqrt(sq_sum_plevel/len(file_list_plevel))

mean_plevel = mean_plevel[0, :]
std_plevel  = std_plevel[0, :]
# Save the results
output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means_netcdf.npy'
np.save(output_file, np.stack([mean_plevel, std_plevel]))
print("Output of pressure levels saved in /hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means_netcdf.npy")

# Surface values
# Initialize variables for calculation
mean_sum_surface = None
sq_sum_surface = None
count = 0

# Process each file in plevels
for file_name in file_list_surface:
    
    print("processing file", file_name)
    data = xr.open_dataset(file_name)

    num_points = len(data.time)
    rng = random.randint(0, num_points-1)

    # isolate time point and convert to numpy array
    data = data.isel(time=rng)
    data = np.stack([data['MSL'].values, data['U10M'].values, data['V10M'].values, data['T2M'].values], axis=0)
    data = data.reshape(1, 4, 721, 1440)

    if mean_sum_surface is None:
        # Initialize arrays for the first file
        mean_sum_surface = np.zeros((1, data.shape[1], 1, 1), dtype=np.float32)
        sq_sum_surface   = np.zeros((1, data.shape[1], 1, 1), dtype=np.float32)

    # Summation for mean and standard deviation
    mean_sum_surface += np.mean(data, keepdims=True, axis = (0, 2, 3))[0]
    sq_sum_surface += np.var(data, keepdims=True, axis = (0, 2, 3))[0]

# Shape of the pressure mean and std should be (5, 13, 1, 1)
# Shape of the surface mean and std should be (4, 1, 1)
mean_surface = mean_sum_surface / len(file_list_surface)
std_surface  = np.sqrt(sq_sum_surface/len(file_list_surface))

mean_surface = mean_surface[0, :]
std_surface  = std_surface[0, :]

# Save the results
output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means_netcdf.npy'
np.save(output_file, np.stack([mean_surface, std_surface]))
print(f"Output of pressure levels saved in {output_file}")
