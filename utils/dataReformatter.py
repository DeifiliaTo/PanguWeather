import os

import h5py
from netCDF4 import Dataset


def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    """Convert netCDF to HDF5."""
    if os.path.isfile(src):
        batch = 2**4
        
        n_img_tot = 4#src_shape[0]
        base = 0
        end = n_img_tot
        idx = base

        with h5py.File(dest, 'a') as fdest:
            for variable_name in varslist:
    
                if frmt == 'nc':
                    fsrc = Dataset(src, 'r', format="NETCDF4").variables[variable_name]
                elif frmt == 'h5':
                    fsrc = h5py.File(src, 'r')[varslist[0]]
                print("fsrc shape", fsrc.shape)

                if 'fields' not in fdest:
                    fdest.create_dataset('fields', shape=(4, 4, 721, 1440), dtype='f') # dims: 4 time points
                                                                            #       4 variables
                                                                            #       721 latitude
                                                                            #       1440 longitude
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
                
                channel_idx += 1 
