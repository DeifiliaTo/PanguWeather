import sys
sys.path.append("/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/networks/")
from Modules.Embedding import PatchEmbedding, PatchRecovery
from Modules.Sampling import UpSample, DownSample
from Modules.Attention import EarthSpecificLayer

from torch import nn
import torch.nn as nn

# Common operations for the data, you may design it or simply use deep learning APIs default operations
# linspace: a tensor version of numpy.linspace
# MeshGrid: a tensor version of numpy.meshgrid
# Stack: a tensor version of numpy.stack
# Flatten: a tensor version of numpy.ndarray.flatten
# TensorSum: a tensor version of numpy.sum
# TensorAbs: a tensor version of numpy.abs
# Concatenate: a tensor version of numpy.concatenate
from torch import linspace

# Common functions for training models
from torch import load

# Custom functions to read your data from the disc
import torch

class PanguModel(nn.Module):
  def __init__(self, C=192, patch_size=(2, 4, 4), device='cpu'):
    super().__init__()
    # Drop path rate is linearly increased as the depth increases
    drop_list = linspace(0, 0.2, 8) # used to be drop_path_list
    
    self.C = C
    self.patch_size = patch_size

    # Patch embedding
    self._input_layer = PatchEmbedding(patch_size, dim=self.C, device=device)

    # Four basic layers
    self.layer1 = EarthSpecificLayer(2, self.C, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360), absolute_bias=False)
    self.layer2 = EarthSpecificLayer(6, 2*self.C, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180), absolute_bias=False)
    self.layer3 = EarthSpecificLayer(6, 2*self.C, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180), absolute_bias=False)
    self.layer4 = EarthSpecificLayer(2, self.C, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360), absolute_bias=False)

    # Upsample and downsample
    self.upsample = UpSample(self.C*2, self.C, 8, 180, 91)

    self.downsample = DownSample(self.C, downsampling=(2,2))
    
    # Patch Recovery
    self._output_layer = PatchRecovery(self.patch_size, dim=2*self.C) # added patch size
    
  def forward(self, input, input_surface):
    '''Backbone architecture'''
    # Embed the input fields into patches

    x = self._input_layer(input, input_surface)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 181, 360) 
    
    # Store the tensor for skip-connection
    skip = x.clone()
    
    # Downsample from (8, 360, 181) to (8, 180, 91)
    x = self.downsample(x, 8, 181, 360)
    
    # Layer 2, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 91, 180) 

    # Decoder, composed of two layers
    # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer3(x, 8, 91, 180) 

    # Upsample from (8, 180, 91) to (8, 360, 181)
    x = self.upsample(x)

    # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
    x = self.layer4(x, 8, 181, 360) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), dim=2)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, Z=8, H=181, W=360)
    return output, output_surface
  

def PerlinNoise():
  '''Generate random Perlin noise: we follow https://github.com/pvigier/perlin-numpy/ to calculate the perlin noise.'''
  # Define number of noise
  octaves = 3
  # Define the scaling factor of noise
  noise_scale = 0.2
  # Define the number of periods of noise along the axis
  period_number = 12
  # The size of an input slice
  H, W = 721, 1440
  # Scaling factor between two octaves
  persistence = 0.5
  # see https://github.com/pvigier/perlin-numpy/ for the implementation of GenerateFractalNoise (e.g., from perlin_numpy import generate_fractal_noise_3d)
  perlin_noise = noise_scale*GenerateFractalNoise((H, W), (period_number, period_number), octaves, persistence)
  return perlin_noise
