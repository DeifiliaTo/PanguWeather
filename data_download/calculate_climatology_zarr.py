import random

import numpy as np
import xarray as xr

iterations = 10
n_per_iteration = 200
n_random = iterations * n_per_iteration
data_dir = '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr/'
zarr_data = xr.open_dataset(data_dir, engine='zarr')
random_index_total = random.sample(range(len(zarr_data['time'])), n_random)

# file names
output_dir_pressure = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/pressure/'  
output_dir_surface  = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/climatology/surface/' 

for iter in range(iterations):
    print("Iteration", iter)
    random_index = random_index_total[iter*n_per_iteration:(iter+1)*(n_per_iteration)]
    data_sample = zarr_data.isel(time=random_index,level=slice(None, None, -1))

    # Pressure data
    sample_pressure = data_sample[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
    sample_pressure = np.stack([sample_pressure['geopotential'].values, sample_pressure['specific_humidity'].values, sample_pressure['temperature'].values, sample_pressure['u_component_of_wind'].values, sample_pressure['v_component_of_wind'].values], axis=0)
    sample_pressure = np.transpose(sample_pressure, (1, 0, 2, 3, 4))
    sample_pressure_mean = np.mean(sample_pressure, axis=(0)).reshape(5, 13, 721, 1440)
    sample_pressure_std  = np.std(sample_pressure, axis=(0)).reshape(5, 13, 721, 1440)

    # Surface data
    sample_surface = data_sample[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
    sample_surface = np.stack([sample_surface['mean_sea_level_pressure'].values, sample_surface['10m_u_component_of_wind'].values, sample_surface['10m_v_component_of_wind'].values, sample_surface['2m_temperature'].values], axis=0)
    sample_surface = np.transpose(sample_surface, (1, 0, 2, 3))
    sample_surface_mean = np.mean(sample_surface, axis=(0)).reshape(4, 721, 1440)
    sample_surface_std  = np.std(sample_surface, axis=(0)).reshape(4, 721, 1440)

    # Save the results
    output_file_pressure = output_dir_pressure + 'pressure_zarr_' + str(iter) + '.npy'
    np.save(output_file_pressure, sample_pressure_mean)
    print(f"Output of pressure levels saved in {output_file_pressure}")

    output_file_surface = output_dir_surface + 'surface_zarr_' + str(iter) + '.npy'
    np.save(output_file_surface, sample_surface_mean)
    print(f"Output of surface values saved in {output_file_surface}")

print("Calculating overall mean")
baseline_pressure = np.load(output_dir_pressure + 'pressure_zarr_' + str(0) + '.npy')
baseline_surface  = np.load(output_dir_surface + 'surface_zarr_' + str(0) + '.npy')

# Iterate through all saved files and sum
for iter in range(1, iterations):
    output_file_pressure = output_dir_pressure + 'pressure_zarr_' + str(iter) + '.npy'
    output_file_surface = output_dir_surface + 'surface_zarr_' + str(iter) + '.npy'

    baseline_pressure += np.load(output_file_pressure)
    baseline_surface  += np.load(output_file_surface)

baseline_pressure /= iterations
baseline_surface  /= iterations

np.save(output_dir_pressure + 'pressure_climatology.npy', baseline_pressure)
np.save(output_dir_surface  + 'surface_climatology.npy', baseline_surface)

