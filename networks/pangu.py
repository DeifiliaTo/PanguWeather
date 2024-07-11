import sys

sys.path.append("./networks/")
import torch
from Modules.Attention import EarthSpecificLayerAbsolute
from Modules.Embedding import PatchEmbedding, PatchRecovery
from Modules.Sampling import DownSample, UpSample
from torch import linspace, nn


class PanguModel(nn.Module):
  """Class definition of Pangu model."""

  def __init__(self, dim=192, patch_size=(2, 4, 4), device='cpu'):
    """
    Initialize Pangu model.

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
    self.layer1 = EarthSpecificLayerAbsolute(2, self.dim, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360), window_size=torch.tensor([2, 6, 12]))
    self.layer2 = EarthSpecificLayerAbsolute(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180), window_size=torch.tensor([2, 6, 12]))
    self.layer3 = EarthSpecificLayerAbsolute(6, 2*self.dim, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180), window_size=torch.tensor([2, 6, 12]))
    self.layer4 = EarthSpecificLayerAbsolute(2, self.dim, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360), window_size=torch.tensor([2, 6, 12]))

    # Upsample and downsample
    self.upsample = UpSample(self.dim*2, self.dim, n_height=8, n_lat=91, n_lon=180, height_crop=(0,0), lat_crop=(0, 1), lon_crop=(0, 0))
    self.downsample = DownSample(self.dim, downsampling=(2,2))
    
    # Patch Recovery
    self._output_layer = PatchRecovery(self.patch_size, dim=2*self.dim) # added patch size
    
  def forward(self, input, input_surface):
    """
    Forward pass of Pangu model.
    
    input: Tensor
      of shape (n_batch,  n_fields, n_vert, n_lat, n_lon)
    input_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon) 
    
    Returns
    -------
    output: Tensor
      of shape (n_batch,  n_fields, n_vert, n_lat, n_lon)
    output_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon) 
    """
    # Embed the input fields into patches

    x = self._input_layer(input, input_surface)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 181, 360, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 181, 360) 
    
    # Store the tensor for skip-connection
    skip = x.clone()
    
    # Downsample from (8, 181, 360) to (8, 91, 180)
    x = self.downsample(x, 8, 181, 360)

    # Layer 2, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 91, 180) 

    # Decoder, composed of two layers
    # Layer 3, shape (8, 91, 180, 2C), C = 192 as in the original paper
    x = self.layer3(x, 8, 91, 180) 

    # Upsample from (8, 91, 180) to (8, 181, 360)
    x = self.upsample(x)

    # Layer 4, shape (8, 181, 360, 2C), C = 192 as in the original paper
    x = self.layer4(x, 8, 181, 360) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), dim=2)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, n_patch_vert=8, n_patch_lat=181, n_patch_lon=360)
    return output, output_surface
