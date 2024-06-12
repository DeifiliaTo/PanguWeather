# Pangu-Weather Ablation Study
This repository is a modified replication of Pangu-Weather (PGW), published by Bi et al., 2023 in https://www.nature.com/articles/s41586-023-06185-3. The code was developed based on the pseudocode in https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py.

A general 3D-based Transformer is implemented here. The modules (Embedding, Up/downsampling, MLP) are defined under networks/Modules. 
The individual ablated architectures are defined in the networks/ directory. Unless stated otherwise, all models are PanguLite with a 3D attention mechanism and one up/downsampling block. The implemented models are:

- PanguLite: Replicated based on the description by the authors with a patch embedding size of (2, 8, 8)
- PanguLite2DAttention: PGW-Lite with 2D attention mechanism
- PanguLite2DAttentionPosEmbed: PGW-Lite with no bias term, 2D attention mechanism, positional embedding
- PositionalEmbedding: PGW-Lite with no bias term, positional embedding
- Three_layers: PGW-Lite with two sets of up/downsampling layers
- TwoDimensional: PGW-Lite with 2D attention
- noBias: PGW-Lite with no bias term
- pangu: Full replication of PGW
- relativeBias: PGW-Lite with a relative (instead of earth-specific bias) term

# Running the training script
To train the model, the params dict in the train.py file can be used to specify the model, the number of data samples, the lead time, and the paths to the data.
Each new training run will save the data in a specified directory, while generating a new prefix hash in format yyyymmdd_randomhash. This hash is used to restart training at a checkpoint.

```
  params = {}
    params['train_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['valid_data_path'] =  '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
    params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_zarr.npy' 
    params['surface_static_data_path'] =  '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_zarr.npy'  
    params['dt'] = 24                # Lead time
    params['num_data_workers'] = 2
    params['data_distributed'] = True # If running DDP
    params['filetype'] = 'zarr' # hdf5, netcdf, or zarr
    params['num_epochs'] = 100 # Total number of epochs 
    num_epochs = params['num_epochs']
    params['C'] = 192  # Hidden dimension
    params['subset_size'] = None # None: will train with all the data. Entering an integer  means randomly subsampling the trianing data
    params['validation_subset_size'] = None # None: will validate with all the data. Entering an integer  means randomly subsampling the validatoin data
    params['restart'] = False   # False if starting a new training run. True if reloading from a specific epoch
    params['hash'] = "20240522_295466069" # None: if starting a new training run. Specifying the hash here when params['restart'] = True identifies the model to be trained.
    params['Lite'] = True # Lite vs. Full run
    params['daily'] = False # if True, only UTC+00:00 time is used
    params['save_counter'] = 10 # Used to restart run. Specify the last model number that was saved.
```

The model can be specified under
```

    # Specify model
    params['model'] = 'panguLite'
    # pangu        = full run
    # panguLite    = light model
    # relativeBias = relative bias
    # noBias       = no bias
    # 2D           = 2D transformer
    # threeLayer   = 1 more down/upsampling layer
    # positionEmbedding     = absolute position embedding
```

To run the training script, run 
```
sbatch < run.sh
```
while modifying the SLURM commands to match your system
