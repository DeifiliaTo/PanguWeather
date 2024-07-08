import sys

sys.path.append("/hkfs/work/workspace/scratch/ke4365-pangu/pangu-weather/networks/")
import torch
from Modules.Attention import EarthSpecificLayerNoBias
from Modules.Embedding import PatchEmbedding, PatchRecovery
from Modules.Sampling import DownSample, UpSample
from timm.layers import trunc_normal_
from torch import linspace, nn


class PanguModel(nn.Module):
  """Class definition of learned positional embedding model."""

  def __init__(self, dim=192, patch_size=(2, 8, 8), device='cpu'):
    """
    Initialize positional embedding model.

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

    self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 728, 192))
    trunc_normal_(self.absolute_pos_embed, std=0.02)

    # number of patches

    # Four basic layers
    self.layer1 = EarthSpecificLayerNoBias(2, self.dim, drop_list[:2], 6,  input_shape=[8, 93], device=device, input_resolution=(8, 93, 180), window_size=torch.tensor([2, 6, 12]))
    self.layer2 = EarthSpecificLayerNoBias(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 46], device=device, input_resolution=(8, 46, 90), window_size=torch.tensor([2, 6, 12]))
    self.layer3 = EarthSpecificLayerNoBias(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 46], device=device, input_resolution=(8, 46, 90), window_size=torch.tensor([2, 6, 12]))
    self.layer4 = EarthSpecificLayerNoBias(2, self.dim, drop_list[:2], 6,  input_shape=[8, 93], device=device, input_resolution=(8, 93, 180), window_size=torch.tensor([2, 6, 12]))

    # Upsample and downsample
    self.upsample = UpSample(self.dim*2, self.dim, n_height=8, n_lat=46, n_lon=90, height_crop=(0,0), lat_crop=(0, 1), lon_crop=(0, 0))

    self.downsample = DownSample(self.dim, downsampling=(2,2))
    
    # Patch Recovery
    self._output_layer = PatchRecovery(self.patch_size, dim=2*self.dim) # added patch size
    
  def forward(self, input, input_surface):
    """Forward pass of positional embedding model."""
    # Embed the input fields into patches

    x = self._input_layer(input, input_surface)
    # x: shape(nBatch, 131040, 192)

    x = x.reshape(x.shape[0], 728, 180, 192) + self.absolute_pos_embed.unsqueeze(2)
    x = x.reshape(x.shape[0], 131040, 192)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 91, 180, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 91, 180) 
    
    
    # Store the tensor for skip-connection
    skip = x.clone()
    
    # Downsample from (8, 91, 180) to (8, 46, 90)
    x = self.downsample(x, 8, 91, 180)

    # Layer 2, shape (8, 46, 90, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 46, 90) 

    # Decoder, composed of two layers
    # Layer 3, shape (8, 46, 90, 2C), C = 192 as in the original paper
    x = self.layer3(x, 8, 46, 90) 

    # Upsample from (8, 46, 90) to (8, 91, 180)
    x = self.upsample(x)

    # Layer 4, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer4(x, 8, 91, 180) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), dim=2)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, 8, 91, 180)
    return output, output_surface
