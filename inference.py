from networks import pangu
import sys
import torch
from utils import data_loader_multifiles
import matplotlib.pyplot as plt
import numpy as np

model_path = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/pangum1.pt'
results_directory = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/results/'
sys.path.append('/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/networks')
model = pangu.PanguModel()

try:
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
except RuntimeError as e:
    # Handle the case where the model was trained with DataParallel
    if 'module.' in str(e):
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key.replace('module.', '') if 'module.' in key else key
            new_checkpoint[new_key] = value
        model.load_state_dict(new_checkpoint)
    else:
        raise  # Re-raise the error if it's not related to 'module.'


params = {}
params['model_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/trained_models/pangum1.pt'
params['train_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/train/'
params['valid_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/valid/'
params['pressure_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/pressure_means.h5'
params['surface_static_data_path'] = '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/static/surface_means.h5'
params['dt'] = 1
params['n_history'] = 1
params['in_channels'] = 8
params['out_channels'] = 8
params['roll'] = False
params['add_noise'] = False
params['batch_size'] = 1
params['num_data_workers'] = 2
params['data_distributed'] = True

patch_size = (2, 4, 4)
train_data_loader, train_dataset, train_sampler = data_loader_multifiles.get_data_loader(params, params['train_data_path'], False, train=True, device='cpu', patch_size=patch_size)

data = next(iter(train_data_loader))
result = model(data[0], data[1])
plt.pcolor(np.flipud(result[1][0][1].detach()))
plt.colorbar()