import h5py
import numpy as np
import os
import glob

# List of h5py files
file_list_plevel  = glob.glob('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train/????.h5') 
file_list_surface = glob.glob('/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train/single_????.h5') 

# Initialize variables for calculation
mean_sum_plevel = None
sq_sum_plevel = None
count = 0

# Process each file in plevels
for file_name in file_list_plevel:

    with h5py.File(file_name, 'r') as f:
        print("processing file", file_name)
        data = f['fields'][:]  

        if mean_sum_plevel is None:
            # Initialize arrays for the first file
            mean_sum_plevel = np.zeros((1, data[0].shape[0], data[0].shape[1], 1, 1), dtype=np.float32)
            sq_sum_plevel   = np.zeros((1, data[0].shape[0], data[0].shape[1], 1, 1), dtype=np.float32)

        # Summation for mean and standard deviation
        mean_sum_plevel += np.mean(f['fields'], keepdims=True, axis = (0, 3, 4))[0]
        sq_sum_plevel += np.var(f['fields'], keepdims=True, axis = (0, 3, 4))[0]

mean_plevel = mean_sum_plevel / len(file_list_plevel)
std_plevel  = np.sqrt(sq_sum_plevel/len(file_list_plevel))

# Save the results
output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means.h5'

with h5py.File(output_file, 'w') as f:
    f.create_dataset('mean', data=mean_plevel)
    f.create_dataset('std_dev', data=std_plevel)

print(f"Output of pressure levels saved in {output_file}")

print("Processing surface data")
# Initialize variables for calculation
mean_sum_surface = None
sq_sum_surface = None
count = 0

# Process each file in surface
for file_name in file_list_surface:

    with h5py.File(file_name, 'r') as f:
        print("processing file", file_name)
        data = f['fields'][:]  

        if sq_sum_surface is None:
            # Initialize arrays for the first file
            mean_sum_surface = np.zeros((1, data[0].shape[0], 1, 1), dtype=np.float32)
            sq_sum_surface   = np.zeros((1, data[0].shape[0], 1, 1), dtype=np.float32)

        # Summation for mean and standard deviation
        mean_sum_surface += np.mean(f['fields'], keepdims=True, axis = (0, 2, 3))[0]
        sq_sum_surface += np.var(f['fields'], keepdims=True, axis = (0, 2, 3))[0]

mean_surface = mean_sum_surface / len(file_list_surface)
std_surface  = np.sqrt(sq_sum_surface/len(file_list_surface))

# Save the results
output_file = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means.h5'

with h5py.File(output_file, 'w') as f:
    f.create_dataset('mean', data=mean_surface)
    f.create_dataset('std_dev', data=std_surface)
