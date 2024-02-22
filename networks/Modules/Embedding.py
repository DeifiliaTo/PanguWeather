import torch
import torch.nn as nn
from torch import permute, reshape
from torch.nn import Conv3d, Conv2d, ConvTranspose3d, ConvTranspose2d
from DataLoader import LoadConstantMask

class PatchEmbedding(nn.Module):
  def __init__(self, patch_size, dim, device):
    '''Patch embedding operation'''
    super().__init__()
    self.patch_size = patch_size
    self.dim = dim
    # Here we use convolution to partition data into cubes
    self.conv = Conv3d(in_channels=5, out_channels=dim, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = Conv2d(in_channels=7, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = LoadConstantMask(patch_size)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
    
  def forward(self, input, input_surface):
    # Input should be padded already, according to the patch size
    input_surface_shape = input_surface.shape
    
    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    input = self.conv(input)

    # Add three constant fields to the surface fields
    # Need to broadcast in this case because we are copying the data over more than 1 dimension
    # Broadcast to 4D data
    land_mask  = torch.broadcast_to(self.land_mask,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    soil_type  = torch.broadcast_to(self.soil_type,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    topography = torch.broadcast_to(self.topography,  (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    
    input_surface = torch.cat((input_surface, land_mask, soil_type, topography), dim=1)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    input_surface = input_surface.unsqueeze(2) # performs broadcasting to add a dimension
    x = torch.cat((input, input_surface), dim=2)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, pressure levels, latitude, longitude, fields)
    x = permute(x, (0, 2, 3, 4, 1))
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))
    
    return x
  
class PatchEmbedding2D(PatchEmbedding):
  def __init__(self, patch_size, dim, device):
    '''Patch embedding operation'''
    super().__init__()
    
  def forward(self, input, input_surface):
    # Input should be padded already, according to the patch size
    input_surface_shape = input_surface.shape
    
    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    input = self.conv(input)

    # Add three constant fields to the surface fields
    # Need to broadcast in this case because we are copying the data over more than 1 dimension
    # Broadcast to 4D data
    land_mask  = torch.broadcast_to(self.land_mask,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    soil_type  = torch.broadcast_to(self.soil_type,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    topography = torch.broadcast_to(self.topography,  (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    
    input_surface = torch.cat((input_surface, land_mask, soil_type, topography), dim=1)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    input_surface = input_surface.unsqueeze(2) # performs broadcasting to add a dimension
    x = torch.cat((input, input_surface), dim=2)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, pressure levels: 2, latitude: 3, longitude:4, fields:1)
    x = permute(x, (0, 3, 4, 1, 2))
    # Dimensions: (nData, latitude * longitude, fields * pressure_levels)
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))
    
    return x
  
class PatchRecovery(nn.Module):
  def __init__(self, patch_size, dim):
    '''Patch recovery operation'''
    super().__init__()
    # Hear we use two transposed convolutions to recover data
    self.conv = ConvTranspose3d(in_channels=dim, out_channels=5, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = ConvTranspose2d(in_channels=dim, out_channels=4, kernel_size=patch_size[1:], stride=patch_size[1:])

  def forward(self, x, Z, H, W):
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    x = permute(x, (0, 2, 1))
    # Dimensions: (nData, fields, pressure level, latitude, longitude)
    x = reshape(x, shape=(x.shape[0], x.shape[1], Z, H, W))

    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :])
    output_surface = self.conv_surface(x[:, :, 0, :, :])

    # Recall: Output is still padded. Cropping only occurs outside of training loop.
    return output, output_surface
  
class PatchRecovery2D(PatchRecovery):
  def __init__(self, patch_size, dim):
    '''Patch recovery operation'''
    super().__init__()

  def forward(self, x, Z, H, W):
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions
    # Dimensions: (nData, latitude * longitude, fields * pressure level)
    x = permute(x, (0, 2, 1))
    # Dimensions: (nData, fields, pressure level, latitude, longitude)
    x = reshape(x, shape=(x.shape[0], x.shape[1], Z, H, W))

    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :])
    output_surface = self.conv_surface(x[:, :, 0, :, :])

    # Recall: Output is still padded. Cropping only occurs outside of training loop.
    return output, output_surface