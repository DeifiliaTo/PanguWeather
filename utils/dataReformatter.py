import os
import h5py
from netCDF4 import Dataset as DS
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import time


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
                    fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
                elif frmt == 'h5':
                    fsrc = h5py.File(src, 'r')[varslist[0]]
                print("fsrc shape", fsrc.shape)

                if 'fields' not in fdest:
                    fdest.create_dataset('fields', shape=(4, 4, 721, 1440), dtype='f') # dims: 4 time points
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
