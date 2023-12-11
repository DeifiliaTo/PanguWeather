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
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
#import cv2
#from utils.img_utils import reshape_fields

# params: dictionary, see lines ~533+ of train.py
# files_pattern: train_data_path
# distributed: = dist.is_initialized() = True/False from the environment
# train = True/False boolean

def get_data_loader(params, files_pattern, distributed, train, device, patch_size):

  dataset = GetDataset(params, files_pattern, train, device, patch_size)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params['batch_size']),
                          num_workers=params['num_data_workers'],
                          shuffle=False, #(sampler is None),
                          sampler=sampler if train else None,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
    def __init__(self, params, file_path, train, device, patch_size):
        self.params = params
        self.file_path= file_path
        self.train = train
        print("params", params)
        self.dt = params['dt']
        self.n_history = params['n_history']
        self.in_channels = np.array(params['in_channels'])
        self.out_channels = np.array(params['out_channels'])
        self.n_in_channels  = 5#len(self.in_channels)
        self.n_out_channels = 5#len(self.out_channels)
        self.roll = params['roll']
        self._get_files_stats(file_path)
        self.add_noise = params['add_noise'] if train else False
        self.p_means = h5py.File(params['pressure_static_data_path'])
        self.s_means = h5py.File(params['surface_static_data_path'])
        self.patch_size = patch_size
        self.device = device
        print("dEVICE FOUND IS", device)
        try:
            self.normalize = params.normalize
        except:
            self.normalize = True #by default turn on normalization if not specified in config

    def _get_files_stats(self, file_path, dt=6):
        self.files_paths_pressure = glob.glob(file_path + "/????.h5") # indicates file paths for pressure levels
        self.files_paths_surface = glob.glob(file_path + "/single_????.h5") # indicates file paths for pressure levels

        
        self.files_paths_pressure.sort()
        self.files_paths_surface.sort()
        
        assert len(self.files_paths_pressure) == len(self.files_paths_surface), "Number of years not identical in pressure vs. surface level data."
    
        self.n_years = len(self.files_paths_pressure)
        with h5py.File(self.files_paths_pressure[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths_pressure[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            #original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]
            self.n_in_channels = 13 #TODO
        print("number of samples per year", self.n_samples_per_year)
        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files_pressure = [None for _ in range(self.n_years)]
        self.files_surface = [None for _ in range(self.n_years)]
        
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(file_path, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
        logging.info("Delta t: {} hours".format(6*self.dt))
        
    def _open_pressure_file(self, year_idx):
        _file = h5py.File(self.files_paths_pressure[year_idx], 'r')
        self.files_pressure[year_idx] = _file
        
    def _open_surface_file(self, year_idx):
        print(self.files_paths_surface[year_idx])
        _file = h5py.File(self.files_paths_surface[year_idx], 'r')
        self.files_surface[year_idx] = _file
      
    def __len__(self):
        return self.n_samples_total
    
    def __getitem__(self, global_idx, normalize=True):
        # TODO: not yet safety checked for edge cases or other errors
        # TODO: Not compatible (from the data aspect) with different dt times as in PGW
        year_idx  = int(global_idx/self.n_samples_per_year) #which year we are on
        local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

        #open image file
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
            t1 = self.files_pressure[year_idx]['fields'][local_idx]
            t2 = self.files_surface[year_idx]['fields'][local_idx]
            t3 = self.files_pressure[year_idx]['fields'][target_step]
            t4 = self.files_surface[year_idx]['fields'][target_step]
        
        t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
        return t1, t2, t3, t4

        
    def _pad_data(self, t1, t2, t3, t4):
        # perform padding for patch embedding step
        input_shape = t1.shape  # shape is (5 variables x 13 pressure levels x 721 latitude x 1440 longitude)
        input_surface_shape = t2.shape
        
        
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
    
