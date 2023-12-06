# Write pressure level data to hdf5 data

import pandas as pd
from netCDF4 import Dataset as DS
import numpy as np
import torch
import glob
import logging
import h5py
import argparse
import os
import time
 
from torch.utils.data import DataLoader, Dataset


def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    if os.path.isfile(src):
        batch = 2**4
        
        Nimgtot = 4#src_shape[0]

        Nimg = Nimgtot
        base = 0
        end = Nimgtot
        idx = base

        with h5py.File(dest, 'a') as fdest:
            for variable_name in varslist:
                
                if frmt == 'nc':
                    print("src name", src)
                    fsrc = DS(src, 'r', format="NETCDF4")
                    print(fsrc['time'])
                    Nimg = fsrc['time'].shape[0]
                    end = Nimg
                    batch = end # TODO: currently not suitable for parallel writes
                    fsrc = fsrc.variables[varslist]
                elif frmt == 'h5':
                    fsrc = h5py.File(src, 'r')[varslist]
                

                if 'fields' not in fdest:
                    fdest.create_dataset('fields', shape=(Nimg, 4, 721, 1440), dtype='f') # dims: 4 time points
                                                                            #       4 variables
                                                                            #       721 latitude
                                                                            #       1440 longitude
                
                start = time.time()
                
                while idx<end:
                    if end - idx < batch:
                        if len(fsrc.shape) == 4:
                            ims = fsrc[idx:end,src_idx]
                        else:
                            ims = fsrc[idx:end]
                        print(ims.shape)
                        fdest['fields'][idx:end,  channel_idx, :, :] = ims
                        
                        break
                    else:
                        if len(fsrc.shape) == 4:
                            ims = fsrc[idx:idx+batch,src_idx]
                        else:
                            ims = fsrc[idx:idx+batch]
                        #ims = fsrc[idx:idx+batch]
                        print("ims shape", ims.shape)
                        fdest['fields'][idx:idx+batch,  channel_idx, :, :] = ims
                        idx+=batch
                        ttot = time.time() - start
                        eta = (end - base)/((idx - base)/ttot)
                        hrs = eta//3600
                        mins = (eta - 3600*hrs)//60
                        secs = (eta - 3600*hrs - 60*mins)
    
                ttot = time.time() - start
                hrs = ttot//3600
                mins = (ttot - 3600*hrs)//60
                secs = (ttot - 3600*hrs - 60*mins)
                channel_idx += 1 

def writepltofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    if os.path.isfile(src):
        batch = 0
        
        
        Nimgtot = 96 #src_shape[0]

        Nimg = Nimgtot
        base = 0
        end = Nimgtot
        idx = base

        with h5py.File(dest, 'a') as fdest:
            for variable_name in varslist:
    
                if frmt == 'nc':
                    fsrc = DS(src, 'r', format="NETCDF4")
                    print("src", src)
                    print("fsrc", fsrc)
                    Nimg = fsrc['t'].shape[0]
                    fsrc = fsrc.variables[variable_name]
                    end = Nimg
                    batch = end # TODO: currently not suitable for parallel writes
                elif frmt == 'h5':
                    fsrc = h5py.File(src, 'r')[varslist[0]]
                print("fsrc shape", fsrc.shape)
                print("Nimg ", Nimg)
                if 'fields' not in fdest:
                    fdest.create_dataset('fields', shape=(Nimg, 5, 13, 721, 1440), dtype='f') # dims: 4 time points
                                                                            #       5 variables
                                                                            #       13 pressure levels
                                                                            #       721 latitude
                                                                            #       1440 longitude
                print("batch", batch)
                start = time.time()
                
                while idx<end:
                    if end - idx < batch:
                        if len(fsrc.shape) == 5:
                            ims = fsrc[idx:end,src_idx, :]
                        else:
                            ims = fsrc[idx:end]
                        print(ims.shape)
                        fdest['fields'][idx:end,  channel_idx, :, :, :] = ims
                        break
                    else:
                        if len(fsrc.shape) == 5:
                            ims = fsrc[idx:idx+batch,src_idx, :]
                        else:
                            ims = fsrc[idx:idx+batch]
                        #ims = fsrc[idx:idx+batch]
                        print("ims shape", ims.shape)
                        print("channel idx", channel_idx)
                        print("idx", idx)
                        print(fdest['fields'].shape)
                        fdest['fields'][idx:idx+batch,  channel_idx, :, :, :] = ims
                        print("we break heree")
                        idx+=batch
                        ttot = time.time() - start
                        eta = (end - base)/((idx - base)/ttot)
                        hrs = eta//3600
                        mins = (eta - 3600*hrs)//60
                        secs = (eta - 3600*hrs - 60*mins)
    
                ttot = time.time() - start
                hrs = ttot//3600
                mins = (ttot - 3600*hrs)//60
                secs = (ttot - 3600*hrs - 60*mins)
                channel_idx += 1 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str, default="PANGU_ERA5_data_v0/", help="input path to nc files", required=True)
    parser.add_argument("--h5_path", type=str, default="PANGU_ERA5_data_v0/", help="output path to h5 files", required=True)
    parser.add_argument("--year", type=int, help="Year to convert data", required=True)
    parser.add_argument('-pv','--pressure_variables', nargs='+', default=['z', 'q', 't', 'u'], help='Short names of variables to convert IN ORDER')
    parser.add_argument('-sv','--surface_variables', nargs='+', default=['msl', 'u10', 'v10', 't2m'], help='Short names of variables to convert IN ORDER')

    args = parser.parse_args()

    for var in range(len(args.pressure_variables)):
        writepltofile(src=args.nc_path + str(args.year) + '.nc',
                dest=args.h5_path + str(args.year) + '.h5',
                channel_idx=var,
                varslist=args.pressure_variables[var])
    
    args.surface_variables = ['msl', 'u10', 'v10', 't2m']
    for var in range(len(args.surface_variables)):
        writetofile(src=args.nc_path + 'single_' + str(args.year) + '.nc',
                dest=args.h5_path + 'single_' + str(args.year) + '.h5',
                channel_idx=var,
                varslist=args.surface_variables[var])
    

    