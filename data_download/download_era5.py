import xarray as xr
import dask
import dask.array as da

def clean_dataset(ds):
    for var in ds.variables.values():
        print(var.encoding, var.shape)
        if 'chunks' in var.encoding and len(var.shape)==3:
            del var.encoding['chunks']
    return ds

var_list = ['geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind', 'mean_sea_level_pressure',
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'surface_pressure', 'total_column_water_vapour', 'total_precipitation']

level_indices = [1000,925,850,700,600,500,400,300,250,200,150,100,50]

# Open the Zarr dataset with dask
wb_data = xr.open_dataset('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr', engine='zarr', chunks={'time': 'auto'})

for var in var_list:
    # Subset the dataset to keep only the desired levels
    wb_data_subset = wb_data.sel(level=level_indices,time=slice('1979-01-01',None))[var]
    wb_data_subset = clean_dataset(wb_data_subset)

    # Write the subsetted dataset to a local Zarr file
    wb_data_subset.to_zarr('/path_to_your_data/era5.zarr', mode='a', compute=True)