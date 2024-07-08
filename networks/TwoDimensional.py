import sys

sys.path.append("/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/networks/")
import torch
from Modules.Attention import EarthSpecificLayer2DNoBias
from Modules.Embedding import PatchEmbedding2D, PatchRecovery2D
from Modules.Sampling import DownSample2D, UpSample2D
from timm.layers import trunc_normal_
from torch import linspace, nn


class PanguModel(nn.Module):
  """Class definition of PanguLite 2D model."""

  def __init__(self, dim=int(192*1.5), patch_size=(4, 4), device='cpu'):
    """
    Initialize 2D Lite model.

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
    self._input_layer = PatchEmbedding2D(patch_size, dim=self.dim, device=device)
    
    self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 181, self.dim))
    trunc_normal_(self.absolute_pos_embed, std=0.02)

    # Four basic layers
    self.layer1 = EarthSpecificLayer2DNoBias(2, self.dim, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(186, 360), window_size=torch.tensor([6, 12]))
    self.layer2 = EarthSpecificLayer2DNoBias(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(96, 180), window_size=torch.tensor([6, 12]))
    self.layer3 = EarthSpecificLayer2DNoBias(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(96, 180), window_size=torch.tensor([6, 12]))
    self.layer4 = EarthSpecificLayer2DNoBias(2, self.dim, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(186, 360), window_size=torch.tensor([6, 12]))

    # Upsample and downsample
    self.upsample = UpSample2D(self.dim*2, self.dim, n_lat=91, n_lon=180, lat_crop=(0, 1), lon_crop=(0, 0))

    self.downsample = DownSample2D(self.dim, downsampling=(2,2))
    
    # Patch Recovery
    self._output_layer = PatchRecovery2D(self.patch_size, dim=2*self.dim) # added patch size
    
  def forward(self, input, input_surface):
    """Forward pass of 2D Lite model."""
    # Embed the input fields into patches

    x = self._input_layer(input, input_surface)
    x = x.reshape(x.shape[0], 181, 360, self.dim) + self.absolute_pos_embed.unsqueeze(2)
    x = x.reshape(x.shape[0], -1, self.dim)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 181, 360, C), C = 192 as in the original paper
    x = self.layer1(x, 181, 360) 
    
    # Store the tensor for skip-connection
    skip = x.clone()
    
    # Downsample from (8, 181, 360) to (8, 91, 180)
    x = self.downsample(x, 181, 360)

    # Layer 2, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer2(x, 91, 180) 

    # Decoder, composed of two layers
    # Layer 3, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer3(x, 91, 180) 

    # Upsample from (8, 91, 180) to (8, 181, 360)
    x = self.upsample(x)

    # Layer 4, shape (8, 181, 360, 2C), C = 192 as in the original paper
    x = self.layer4(x, 181, 360) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), dim=2)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, 181, 360)
    return output, output_surface