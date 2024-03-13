import xarray as xr
import numpy as np
import glob
import random

data_dir = '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr/'
n_random = 200
zarr_data = xr.open_dataset(data_dir, engine='zarr')
random_index = random.sample(range(len(zarr_data['time'])), n_random)
data_sample = zarr_data.isel(time=random_index,level=slice(None, None, -1))

# Pressure data
sample_pressure = data_sample[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
sample_pressure = np.stack([sample_pressure['geopotential'].values, sample_pressure['specific_humidity'].values, sample_pressure['temperature'].values, sample_pressure['u_component_of_wind'].values, sample_pressure['v_component_of_wind'].values], axis=0)
sample_pressure = np.transpose(sample_pressure, (1, 0, 2, 3, 4))
sample_pressure_mean = np.mean(sample_pressure, axis=(0, 3, 4)).reshape(5, 13, 1, 1)
sample_pressure_std  = np.std(sample_pressure, axis=(0, 3, 4)).reshape(5, 13, 1, 1)

# Surface data
sample_surface = data_sample[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
sample_surface = np.stack([sample_surface['mean_sea_level_pressure'].values, sample_surface['10m_u_component_of_wind'].values, sample_surface['10m_v_component_of_wind'].values, sample_surface['2m_temperature'].values], axis=0)
sample_surface = np.transpose(sample_surface, (1, 0, 2, 3))
sample_surface_mean = np.mean(sample_surface, axis=(0, 2, 3)).reshape(4, 1, 1)
sample_surface_std  = np.std(sample_surface, axis=(0, 2, 3)).reshape(4, 1, 1)

# Save the results
output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy'
np.save(output_file, np.stack([sample_pressure_mean, sample_pressure_std]))
print(f"Output of pressure levels saved in {output_file}")

output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy'
np.save(output_file, np.stack([sample_surface_mean, sample_surface_std]))
print(f"Output of surface values saved in {output_file}")
