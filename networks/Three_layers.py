import sys

sys.path.append("/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/networks/")

import torch
from Modules.Attention import EarthSpecificLayerAbsolute
from Modules.Embedding import PatchEmbedding, PatchRecovery
from Modules.Sampling import DownSample, UpSample
from torch import linspace, nn


class PanguModel(nn.Module):
  """Class definition of model with depth of 3, i.e. 2 up and downsampling steps."""

  def __init__(self, dim=192, patch_size=(2, 8, 8), device='cpu'):
    """
    Initialize 3-depth Lite model.

    dim: int
    patch_size = Tuple(int, int, int)
            patch size in the height, lat, long directions
    device: String
            device that the code is offloaded onto.
    """
    super().__init__()
    # Drop path rate is linearly increased as the depth increases
    drop_list = linspace(0, 0.2, 8) # used to be drop_path_list
    
    self.dim = dim
    self.patch_size = patch_size

    # Patch embedding
    self._input_layer = PatchEmbedding(patch_size, dim=self.dim, device=device)

    # Four basic layers
    self.layer1 = EarthSpecificLayerAbsolute(2, self.dim, drop_list[:2], 6,  input_shape=[8, 93], device=device, input_resolution=(8, 93, 180), window_size=torch.tensor([2, 6, 12]))
    self.layer2 = EarthSpecificLayerAbsolute(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 46], device=device, input_resolution=(8, 46, 90), window_size=torch.tensor([2, 6, 12]))
    self.layer3 = EarthSpecificLayerAbsolute(4, 4*self.dim, drop_list[2:], 12, input_shape=[8, 23], device=device, input_resolution=(8, 23, 45), window_size=torch.tensor([2, 6, 12]))
    self.layer4 = EarthSpecificLayerAbsolute(4, 4*self.dim, drop_list[2:], 12, input_shape=[8, 23], device=device, input_resolution=(8, 23, 45), window_size=torch.tensor([2, 6, 12]))
    self.layer5 = EarthSpecificLayerAbsolute(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 46], device=device, input_resolution=(8, 46, 90), window_size=torch.tensor([2, 6, 12]))
    self.layer6 = EarthSpecificLayerAbsolute(2, self.dim, drop_list[:2], 6,  input_shape=[8, 93], device=device, input_resolution=(8, 93, 180), window_size=torch.tensor([2, 6, 12]))

    # Upsample and downsample
    self.upsample  = UpSample(self.dim*2, self.dim, nHeight=8, nLat=46, nLon=90, height_crop=(0,0), lat_crop=(0, 1), lon_crop=(0, 0))
    self.upsample2 = UpSample(self.dim*4, 2*self.dim, nHeight=8, nLat=23, nLon=45, height_crop=(0,0), lat_crop=(0, 0), lon_crop=(0, 0))

    self.downsample  = DownSample(self.dim, downsampling=(2,2))
    self.downsample2 = DownSample(self.dim*2, downsampling=(2,2))
    
    # Patch Recovery
    self._output_layer = PatchRecovery(self.patch_size, dim=2*self.dim) # added patch size
    
  def forward(self, input, input_surface):
    """Forward pass of 3-depth Lite model."""
    # Embed the input fields into patches

    x = self._input_layer(input, input_surface)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 91, 180, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 91, 180) 
    
    # Store the tensor for skip-connection
    skip = x.clone()
    
    # Downsample from (8, 91, 180) to (8, 46, 90)
    x = self.downsample(x, 8, 91, 180)

    # Layer 2, shape (8, 46, 90, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 46, 90) 

    skip2 = x.clone()

    # Downsample from (8, 46, 90) to (8, 23, 45)
    x = self.downsample2(x, 8, 46, 90)

    # Layer 3, shape (8, 23, 45, 4C)
    x = self.layer3(x, 8, 23, 45)
    
    # Layer 4, shape (8, 23, 45, 4C)
    x = self.layer4(x, 8, 23, 45)

    # Upsample from (8, 23, 45) to (8, 46, 90)
    x = self.upsample2(x)
    
    x = x + skip2 
        
    # Layer 5, shape (8, 46, 90, 4C), C = 192 as in the original paper
    x = self.layer5(x, 8, 46, 90) 

    # Upsample from (8, 46, 90) to (8, 91, 180)
    x = self.upsample(x)

    # Layer 4, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer6(x, 8, 91, 180) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), dim=2)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, 8, 91, 180)
    return output, output_surface
