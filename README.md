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

# Structure of repository
- constant_masks: Stores constant landd, soil type, typography masks. Stores mean and standard deviation of atmospheric fields used for normalization.
- data_download: Scripts for downloading data and calculating static arrays and climatology
- loss_schedule_exp: All training files needed to replicate loss scheduling experiment
- networks: Model zoo
  - Modules: All PyTorch classes required to build Pangu model
- utils: data loader and evaluation helper functions

# Setting up your environment
Dependencies are saved in requirements.txt. To run the code, create a new virtual environment
```
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

# Data
To train the model, approximately 60 TB of ERA5 data will need to be downloaded, specifically, the variables (Z, Q, T, U, V) at pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50), and surface variables ()
The current dataloader is adapted to the following two datasets downloaded from WeatherBench2 (WB2) https://weatherbench2.readthedocs.io/en/latest/data-guide.html.
1. xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
2. xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr')
Dataset(1) is subsampled to a 6-hourly time resolution and to the pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50) hPa. Note that the 
Dataset(2) has an hourly time resolution and contains all 37 pressure levels.
Data can also be obtained from the Copernicus API directly, https://cds.climate.copernicus.eu/api-how-to.

## DataLoader
The data loader is designed for WB2 data, but can be adapted for .netcdf data downloaded directly from the copernicus API by switching params['file_type'] = 'netcdf' in train.py.

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
# pangu        = replication of Bi et al; absolute position bias model + patch size of (2, 4, 4)
# panguLite    = Lite model
# relativeBias = relative bias + Lite model
# noBias       = no bias + Lite model
# 2D           = 2D transformer + Lite model
# threeLayer   = 1 more down/upsampling layer + Lite model
# positionEmbedding     = absolute position embedding + Lite model
```

To run the training script, run 
```
sbatch < run.sh -C LSDF 
```
while modifying the SLURM parameters to match your system
