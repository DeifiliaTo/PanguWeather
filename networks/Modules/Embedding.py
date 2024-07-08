import torch
import torch.nn as nn
from torch import permute, reshape
from torch.nn import Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d

from DataLoader import load_constant_mask, load_constant_mask_2d


class PatchEmbedding(nn.Module):
  """Patch Embedding."""

  def __init__(self, patch_size, dim, device, in_pressure_channels=5, in_surface_channels=7):
    """
    Init.
    
    patch_size: Tuple(int, int, int)
        Number of pixels in (vert, lat, lon) dimensions per patch
    dim: int
        Hidden dimension
    device: String
        Device that the operation is running on
    in_pressure_channels: int
        Number of variables in the pressure levels. For most models, 5 (Z, Q, T, U, V)
    in_surface_channels: int
        Number of variables in the surface + number of masks. 
        For standard model, 4 (MSLP, U10, V10, T2M) + 3 masks = 7
    """
    super().__init__()
    self.patch_size = patch_size
    self.dim = dim
    # Here we use convolution to partition data into cubes
    self.conv = Conv3d(in_channels=in_pressure_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = Conv2d(in_channels=in_surface_channels, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = load_constant_mask(patch_size)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
    
  def forward(self, input, input_surface):
    """
    Forward pass of patch embedding.
    
    input: Tensor
      of shape (n_batch,  n_fields, n_vert, n_lat, n_lon) 
      n_vert, n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (13 x 721 x 1440) with patch size of (2, 8,8) -> (14, 728, 1440)
    input_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon) 
      n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (721 x 1440) with patch size of (8,8) -> (728, 1440)

    Returns
    -------
    x: Tensor
      of shape (n_batch, n_patch_vert*n_patch_lon*n_patch_lat, hidden_dim)
      i.e., for Lite models, (n_patch_vert, n_patch_lon, n_patch_lat) = (8, 91, 180)
    """
    # Input should be padded already, according to the patch size
    input_surface_shape = input_surface.shape
    
    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches
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
    x = torch.cat((input_surface, input), dim=2)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, pressure levels, latitude, longitude, fields)
    x = permute(x, (0, 2, 3, 4, 1))
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))
    
    return x
  
class PatchEmbedding2D(nn.Module):
  """2D Patch Embedding operation."""

  def __init__(self, patch_size, dim, device, in_channels=72):
    """
    Initialize patch embedding operation.
    
    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    dim: int
        Hidden dimension
    device: String
        Device that the operation is running on
    in_channels: int
        Total number of channels 
        equal to n_pressure_levels * n_pressure_fields + n_surface_fields + masks
        = 13 pressure levels * 5 pressure fields + 4 surface fields + 3 masks
    """
    super().__init__()
    self.patch_size = patch_size
    self.dim = dim
    # Here we use convolution to partition data into cubes
    # in_channels = 13 pressure levels x 5 fields + 4 variables + 3 masks
    # i.e., fields are (Z, Q, T, U, V)
    self.conv_surface = Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(patch_size)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
      
  def forward(self, input, input_surface):
    """
    Forward pass of 2D patch embedding.
    
    input: Tensor
      of shape (n_batch,  n_fields*n_vert, n_lat, n_lon) 
      n_vert, n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (721 x 1440) with patch size of (8,8) -> (14, 728, 1440).
      i.e., in standard model, n_variables*n_vert = 5 vars * 13 pressure heights
    input_surface: Tensor
      of shape (n_batch, n_variables, n_lat, n_lon) 
      n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (721 x 1440) with patch size of (8,8) -> (728, 1440)

    Returns
    -------
    x: Tensor
      of shape (n_batch, n_patch_lon*n_patch_lat, hidden_dim)
      i.e., for Lite models, (n_patch_lon, n_patch_lat) = (91, 180)

    """
    # Input should be padded already, according to the patch size
    input_surface_shape = input_surface.shape
  
    # Add three constant fields to the surface fields
    # Need to broadcast in this case because we are copying the data over more than 1 dimension
    # Broadcast to 4D data
    land_mask  = torch.broadcast_to(self.land_mask,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    soil_type  = torch.broadcast_to(self.soil_type,   (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    topography = torch.broadcast_to(self.topography,  (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
      
    input_surface = torch.cat((input_surface, land_mask, soil_type, topography), dim=1)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    # shape: (nData, fields, latitude, longitude)
    input_surface = torch.cat((input_surface, input), dim=1) 
    input_surface = self.conv_surface(input_surface)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, latitude, longitude, fields)
    x = permute(input_surface, (0, 2, 3, 1))
    # Dimensions: (nData,  latitude * longitude, fields)
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))
    
    return x
  
class PatchRecovery(nn.Module):
  """3D Patch recovery operation."""

  def __init__(self, patch_size, dim, out_pressure_channels=5, out_surface_channels=4):
    """
    Patch recovery. 

    A transpose convolution operation is performed over the pressure and surface outputs to recover the forecasted fields.

    patch_size: Tuple(int, int, int)
        Number of pixels in (vert, lat, lon) dimensions per patch
    dim: int
      Hidden dimension
    out_pressure_channels: int
        Number of variables in the pressure levels to be predicted. For most models, 5 (Z, Q, T, U, V)
    out_surface_channels: int
        Number of variables in the surface to be predicted.
        For standard model, 4 (MSLP, U10, V10, T2M)
    """
    super().__init__()
    # Here we use two transposed convolutions to recover data
    self.conv = ConvTranspose3d(in_channels=dim, out_channels=out_pressure_channels, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = ConvTranspose2d(in_channels=dim, out_channels=out_surface_channels, kernel_size=patch_size[1:], stride=patch_size[1:])

  def forward(self, x, n_patch_vert, n_patch_lat, n_patch_lon):
    """
    Perform 3D inverse operation of the patch embedding operation.
    
    x: Tensor
      of shape (n_batch, n_patch_vert*n_patch_lat*n_patch_lon, 2*hidden_dim)
    n_patch_lat: int
      number of patches in the lat dimension
    n_patch_lon: int
      number of patches in the lon dimension

    Returns
    -------
    output: Tensor
      of shape (n_batch, n_variables, n_vert, n_lat, n_lon)
      (n_vert, n_lat, n_lon) are padded to be divisible by the patch size.
    output_surface: Tensor
      of shape (n_batch, n_variables, n_lat, n_lon)
      (n_lat, n_lon) are padded to be divisible by the patch size.
    """
    # Reshape x back to three dimensions
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    
    x = permute(x, (0, 2, 1))
    # Dimensions: (nData, fields, pressure level, latitude, longitude)
    x = reshape(x, shape=(x.shape[0], x.shape[1], n_patch_vert, n_patch_lat, n_patch_lon))

    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :])
    output_surface = self.conv_surface(x[:, :, 0, :, :])

    # Note: Output is still padded. Cropping only occurs outside of training loop.
    return output, output_surface
  
class PatchRecovery2D(nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, dim, out_channels=69):
    """
    2D Patch recovery.

    A transpose convolution operation is performed over the pressure and surface outputs to recover the forecasted fields.

    patch_size: Tuple(int, int, int)
        Number of pixels in (lat, lon) dimensions per patch
    dim: int
      Hidden dimension
    in_channels: int
      Total number of channels 
      equal to n_pressure_levels * n_pressure_fields + n_surface_fields
      = 13 pressure levels * 5 pressure fields + 4 surface fields = 69
    """
    super().__init__()
    # Here we use two transposed convolutions to recover data
    self.conv = ConvTranspose2d(in_channels=dim, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)

  def forward(self, x, n_patch_lat, n_patch_lon):
    """
    2D inverse operation of the patch embedding operation.
    
    x: Tensor
      of shape (n_batch, n_patch_lat*n_patch_lon, 2*hidden_dim)
    n_patch_lat: int
      number of patches in the lat dimension
    n_patch_lon: int
      number of patches in the lon dimension

    Returns
    -------
    output: Tensor
      of shape (n_batch, n_levels * n_fields, n_lat, n_lon)
    output_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon)
    """
    # Reshape x back to three dimensions
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    x = permute(x, (0, 2, 1))
    # Dimensions: (nData, fields, pressure level, latitude, longitude)
    x = reshape(x, shape=(x.shape[0], x.shape[1], n_patch_lat, n_patch_lon))

    # Call the transposed convolution
    merged_output = self.conv(x)
    output = merged_output[:, 4:, :, :]
    output_surface = merged_output[:, :4, :, :]

    # Note: Output is still padded. Cropping only occurs outside of training loop.
    return output, output_surface