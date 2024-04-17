#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import logging
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import h5py
import pandas as pd
import xarray as xr
#import cv2
#from utils.img_utils import reshape_fields

# params: dictionary, see lines ~533+ of train.py
# files_pattern: train_data_path
# distributed: = dist.is_initialized() = True/False from the environment
# train = True/False boolean

def get_data_loader(params, files_pattern, distributed, train, device, patch_size, subset_size=None):

  dataset = GetDataset(params, files_pattern, train, device, patch_size)
  
  # If we are setting a subset
  if subset_size is not None:
      subset_indices = torch.randperm(len(dataset))[:subset_size]
      dataset = Subset(dataset, subset_indices)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params['batch_size']),
                          num_workers=params['num_data_workers'],
                          shuffle=False, #(sampler is None),
                          sampler=sampler,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
    def __init__(self, params, file_path, train, device, patch_size):
        self.params = params
        self.file_path = file_path
        self.train = train
        self.dt = params['dt']
        self.filetype = params['filetype']
        self._get_files_stats(file_path, daily=params['Lite'])
        self.level_ordering = range(13-1, -1, -1)
        if params['filetype'] == 'hdf5':
            self.p_means = h5py.File(params['pressure_static_data_path'])
            self.s_means = h5py.File(params['surface_static_data_path'])
        elif params['filetype'] == 'netcdf':
            self.p_mean = np.load(params['pressure_static_data_path'])[0].reshape(5, 13, 1, 1)
            self.p_std  = np.load(params['pressure_static_data_path'])[1].reshape(5, 13, 1, 1)
            self.s_mean = np.load(params['surface_static_data_path'])[0].reshape(4, 1, 1)
            self.s_std  = np.load(params['surface_static_data_path'])[1].reshape(4, 1, 1)
        elif params['filetype'] == 'zarr':
            self.p_mean = np.load(params['pressure_static_data_path'])[0].reshape(5, 13, 1, 1)
            self.p_std  = np.load(params['pressure_static_data_path'])[1].reshape(5, 13, 1, 1)
            self.s_mean = np.load(params['surface_static_data_path'])[0].reshape(4, 1, 1)
            self.s_std  = np.load(params['surface_static_data_path'])[1].reshape(4, 1, 1)
        else:
            raise ValueError("File type must be hdf5, netcdf or zarr.")
        self.patch_size = patch_size
        self.device = device
        try:
            self.normalize = params.normalize
        except:
            self.normalize = True #by default turn on normalization if not specified in config

    def _get_files_stats(self, file_path, dt=6, daily=False):
        if self.filetype == 'hdf5':
            self.files_paths_pressure = glob.glob(file_path + "/????.h5") # indicates file paths for pressure levels
            self.files_paths_surface = glob.glob(file_path + "/single_????.h5") # indicates file paths for pressure levels
            self.n_years = len(self.files_paths_pressure)
        elif self.filetype == 'netcdf':
            self.files_paths_pressure = glob.glob(file_path + '/*_pl.nc')
            self.files_paths_surface = glob.glob(file_path + '/*_sfc.nc')
            self.n_years = len(self.files_paths_pressure)
        elif self.filetype == 'zarr':
            self.files_paths_pressure = []
            self.files_paths_surface = []
        else:
            raise ValueError("Input data must be in either the hdf5 or netcdf format")
        
        self.files_paths_pressure.sort()
        self.files_paths_surface.sort()
        
        assert len(self.files_paths_pressure) == len(self.files_paths_surface), "Number of years not identical in pressure vs. surface level data."
        
        if self.filetype == 'hdf5':
            with h5py.File(self.files_paths_pressure[0], 'r') as _f:
                logging.info("Getting file stats from {}".format(self.files_paths_pressure[0]))
                self.n_samples_per_year = _f['fields'].shape[0]
                #original image shape (before padding)
                self.img_shape_x = _f['fields'].shape[2]
                self.img_shape_y = _f['fields'].shape[3]
                self.n_in_channels = 13 #TODO
                self.n_samples_total = self.n_years * self.n_samples_per_year
            self.files_pressure = [None for _ in range(self.n_years)]
            self.files_surface  = [None for _ in range(self.n_years)]
        elif self.filetype == 'netcdf':
            logging.info("Getting file stats from {}".format(self.files_paths_pressure[0]))
            ds = xr.open_dataset(self.files_paths_surface[0])
            #original image shape (before padding)
            self.img_shape_x = len(ds.lon) # 1440
            self.img_shape_y = len(ds.lat) # 721
            self.n_in_channels = len(ds.data_vars)
            n_points = 0
            self.cumulative_points_per_file = []
            self.points_per_file = []
            # Need to iterate through files to find the length because of leap years, incorrectly organized data, etc.
            for path in self.files_paths_surface:
                ds = xr.open_dataset(path)
                n_points = int(len(ds.time))
                self.points_per_file.append(n_points)
            self.cumulative_points_per_file = np.cumsum(np.array(self.points_per_file))
            self.n_samples_total = self.cumulative_points_per_file[-1]
            self.files_pressure = [None for _ in range(self.n_years)]
            self.files_surface  = [None for _ in range(self.n_years)]
        elif self.filetype == 'zarr':
            self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
            times = pd.to_datetime(self.zarr_data['time'].values)
            if self.train and daily: # training case, lite
                train_years = times[(times.year < 2018) & (times.year > 2006)]
                self.zarr_data = self.zarr_data.sel(time=train_years)
            elif self.train and not daily: # training case, 1979-2017
                train_years = times[(times.year < 2018) & (times.year > 1978)]
                self.zarr_data = self.zarr_data.sel(time=train_years)
            else:           # validation
                validation_years = times[times.year == 2019]
                self.zarr_data = self.zarr_data.sel(time=validation_years)
            if daily:
                times = pd.to_datetime(self.zarr_data['time'].values)
                midnight_times = times[times.hour == 0]
                self.zarr_data = self.zarr_data.sel(time=midnight_times)

            self.n_samples_total = len(self.zarr_data['time'])
            self.img_shape_x     = len(self.zarr_data['latitude'])
            self.img_shape_y     = len(self.zarr_data['longitude'])
            # N channels
            self.n_in_channels   = len(self.zarr_data.data_vars)
        else:
            raise ValueError("File type doesn't match one of hdf5, netcdf, or zarr")
        
        # logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(file_path, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
        logging.info("Delta t: {} hours".format(6*self.dt))
        
    def _open_pressure_file(self, file_idx):
        if self.filetype == 'hdf5':
            _file = h5py.File(self.files_paths_pressure[file_idx], 'r')
            self.files_pressure[file_idx] = _file
        else:
            _file = xr.open_dataset(self.files_paths_pressure[file_idx])
            self.files_pressure[file_idx] = _file
            
    def _open_surface_file(self, file_idx):
        if self.filetype == 'hdf5':
            _file = h5py.File(self.files_paths_surface[file_idx], 'r')
            self.files_surface[file_idx] = _file
        else:
            _file = xr.open_dataset(self.files_paths_surface[file_idx])
            self.files_surface[file_idx] = _file
      
    def __len__(self):
        return self.n_samples_total - 1 # -1 to avoid last data point
    
    def __getitem__(self, global_idx, normalize=True, forecast_length=1):
        # TODO: not yet safety checked for edge cases or other errors
        # TODO: Not compatible (from the data aspect) with different dt times as in PGW
        if self.filetype == 'hdf5':
            year_idx  = int(global_idx/self.n_samples_per_year) #which year we are on
            local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

            # open image file
            # TODO: where to close the respective files?
            if self.files_pressure[year_idx] is None:
                self._open_pressure_file(year_idx)
        
            if self.files_surface[year_idx] is None:
                self._open_surface_file(year_idx)
            
            step = self.dt
            target_step = local_idx + step
            if target_step == self.n_samples_per_year:
                target_step = local_idx

            if normalize:
                t1 = torch.as_tensor((self.files_pressure[year_idx]['fields'][local_idx] - self.p_means['mean']) / self.p_means['std_dev'])
                t2 = torch.as_tensor((self.files_surface[year_idx]['fields'][local_idx] - self.s_means['mean']) / self.s_means['std_dev'])
                t3 = torch.as_tensor((self.files_pressure[year_idx]['fields'][target_step] - self.p_means['mean']) / self.p_means['std_dev'])
                t4 = torch.as_tensor((self.files_surface[year_idx]['fields'][target_step] - self.s_means['mean']) / self.s_means['std_dev'])
            else:
                t1 = torch.as_tensor(self.files_pressure[year_idx]['fields'][local_idx])
                t2 = torch.as_tensor(self.files_surface[year_idx]['fields'][local_idx])
                t3 = torch.as_tensor(self.files_pressure[year_idx]['fields'][target_step])
                t4 = torch.as_tensor(self.files_surface[year_idx]['fields'][target_step])
            
            t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
            return t1, t2, t3, t4
        
        elif self.filetype == 'netcdf': # dealing directly with netcdf files
            step = self.dt
            target_step = global_idx + step
            
            # Get file and local_idx for the input steps
            input_file_idx  = np.argmax(self.cumulative_points_per_file > global_idx) #which year we are on
            if input_file_idx > 0:
                input_local_idx = global_idx - self.cumulative_points_per_file[input_file_idx - 1] #which sample in that year we are on - determines indices for centering
            else:
                input_local_idx = global_idx
                # If we are on the last value, output_file_idx = 0 because self.cumulative_points_per_file is never > target_step
                if global_idx == self.n_samples_total:
                    output_file_idx =  -1 # last file value
                    output_local_idx = -2 # last value in file # will break if there is only 1 data point in the last file
            
            # Get file and local_idx for the output step
            output_file_idx  = np.argmax(self.cumulative_points_per_file > target_step) #which year we are on
            # case for which output_file_idx = last 
            if output_file_idx > 0:
                output_local_idx = target_step - self.cumulative_points_per_file[output_file_idx - 1] #which sample in that year we are on - determines indices for centering
            else: # = 0
                output_local_idx = target_step
                # If we are on the last value, output_file_idx = 0 because self.cumulative_points_per_file is never > target_step
                if target_step == self.n_samples_total:
                    output_file_idx =  -1 # last file value
                    output_local_idx = -1 # last value in file
            
            # open files
            if self.files_pressure[input_file_idx] is None:
                self._open_pressure_file(input_file_idx)
        
            if self.files_surface[input_file_idx] is None:
                self._open_surface_file(input_file_idx)
            
            if self.files_pressure[output_file_idx] is None:
                self._open_pressure_file(output_file_idx)

            if self.files_surface[output_file_idx] is None:
                self._open_surface_file(output_file_idx)
            
            if target_step >= self.n_samples_total: # If we are at the very last step
                target_step = input_local_idx

            # Isolate data from time point and convert to numpy array
            input_time_pressure = self.files_pressure[input_file_idx].isel(time=input_local_idx)
            input_time_surface  = self.files_surface[input_file_idx].isel(time=input_local_idx)

            input_time_pressure = np.stack([input_time_pressure['Z'].values, input_time_pressure['Q'].values, input_time_pressure['T'].values, input_time_pressure['U'].values, input_time_pressure['V'].values], axis=0)
            input_time_surface  = np.stack([input_time_surface['MSL'].values, input_time_surface['U10M'].values, input_time_surface['V10M'].values, input_time_surface['T2M'].values], axis=0)

            output_time_pressure = self.files_pressure[output_file_idx].isel(time=output_local_idx)
            output_time_surface  = self.files_surface[output_file_idx].isel(time=output_local_idx)

            output_time_pressure = np.stack([output_time_pressure['Z'].values,  output_time_pressure['Q'].values,   output_time_pressure['T'].values,   output_time_pressure['U'].values, output_time_pressure['V'].values], axis=0)
            output_time_surface  = np.stack([output_time_surface['MSL'].values, output_time_surface['U10M'].values, output_time_surface['V10M'].values, output_time_surface['T2M'].values], axis=0)
            
            if normalize:
                # p_ and s_means is a stack of the mean and standard deviation values
                t1 = torch.as_tensor((input_time_pressure - self.p_mean) / self.p_std)
                t2 = torch.as_tensor((input_time_surface - self.s_mean)  / self.s_std)
                t3 = torch.as_tensor((output_time_pressure - self.p_mean) / self.p_std)
                t4 = torch.as_tensor((output_time_surface - self.s_mean)  / self.s_std)
            else:
                t1 = torch.as_tensor(input_time_pressure)
                t2 = torch.as_tensor(input_time_surface)
                t3 = torch.as_tensor(output_time_pressure)
                t4 = torch.as_tensor(output_time_surface)
            
            t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
            return t1, t2, t3, t4
        
        elif self.filetype == 'zarr':
            
            step = self.dt
            
            output_file_idx = global_idx + 1
            if output_file_idx == self.__len__(): 
                output_file_idx -= 1
            
            # Isolate data from time point and convert to numpy array
            # WeatherBench data stores from low --> high pressure levels
            # We convert to high --> low
            input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
            input_surface  = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            output_pressure = self.zarr_data.isel(time=output_file_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
            output_surface  = self.zarr_data.isel(time=output_file_idx, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            
            # Stack and convert to numpy array
            input_pressure = np.stack([input_pressure['geopotential'].values, input_pressure['specific_humidity'].values, input_pressure['temperature'].values, input_pressure['u_component_of_wind'].values, input_pressure['v_component_of_wind'].values], axis=0)
            input_surface  = np.stack([input_surface['mean_sea_level_pressure'].values, input_surface['10m_u_component_of_wind'].values, input_surface['10m_v_component_of_wind'].values, input_surface['2m_temperature'].values], axis=0)
            output_pressure = np.stack([output_pressure['geopotential'].values, output_pressure['specific_humidity'].values, output_pressure['temperature'].values, output_pressure['u_component_of_wind'].values, output_pressure['v_component_of_wind'].values], axis=0)
            output_surface  = np.stack([output_surface['mean_sea_level_pressure'].values, output_surface['10m_u_component_of_wind'].values, output_surface['10m_v_component_of_wind'].values, output_surface['2m_temperature'].values], axis=0)

            if normalize:
                # p_ and s_means is a stack of the mean and standard deviation values
                t1 = torch.as_tensor((input_pressure - self.p_mean) / self.p_std)
                t2 = torch.as_tensor((input_surface - self.s_mean)  / self.s_std)
                t3 = torch.as_tensor((output_pressure - self.p_mean) / self.p_std)
                t4 = torch.as_tensor((output_surface - self.s_mean)  / self.s_std)
            else:
                t1 = torch.as_tensor(input_pressure)
                t2 = torch.as_tensor(input_surface)
                t3 = torch.as_tensor(output_pressure)
                t4 = torch.as_tensor(output_surface)
            
            t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
            return t1, t2, t3, t4
        
    def _pad_data(self, t1, t2, t3, t4):
        # perform padding for patch embedding step
        input_shape = t1.shape  # shape is (5 variables x 13 pressure levels x 721 latitude x 1440 longitude)
        
        x1_pad    = (self.patch_size[0] - (input_shape[1] % self.patch_size[0])) % self.patch_size[0] // 2
        x2_pad    = (self.patch_size[0] - (input_shape[1] % self.patch_size[0])) % self.patch_size[0] - x1_pad
        y1_pad    = (self.patch_size[1] - (input_shape[2] % self.patch_size[1])) % self.patch_size[1] // 2
        y2_pad    = (self.patch_size[1] - (input_shape[2] % self.patch_size[1])) % self.patch_size[1] - y1_pad
        z1_pad    = (self.patch_size[2] - (input_shape[3] % self.patch_size[2])) % self.patch_size[2] // 2
        z2_pad    = (self.patch_size[2] - (input_shape[3] % self.patch_size[2])) % self.patch_size[2] - z1_pad

        # pad pressure fields input and output
        t1 = torch.nn.functional.pad(t1, pad=(z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)
        t3 = torch.nn.functional.pad(t3, pad=(z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)
        
        # pad 
        t2  = torch.nn.functional.pad(t2, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)
        t4  = torch.nn.functional.pad(t4, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

        return t1, t2, t3, t4
    
