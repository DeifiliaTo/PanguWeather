import torch
import torch.nn as nn
from torch import permute, reshape
from torch.nn import LayerNorm, Linear


class DownSample(nn.Module):
  """Downsample."""

  def __init__(self, dim, downsampling=(2,2)):
    """Down-sampling operation.
    
    dim: int
            dimension
    downsample: Tuple(int, int)

    """
    super().__init__()
    # A linear function and a layer normalization
    self.linear = Linear(4*dim, 2*dim, bias=False)
    self.norm = LayerNorm(4*dim)
    self.downsampling = downsampling
  
  def forward(self, x, n_patch_vert, n_patch_lat, n_patch_lon):
    """
    Forward pass of Downsample 3D.
    
    x: Tensor
        input tensor
    n_patch_vert: int
        number of patches in the vertical dimension.
    n_patch_lat: int
        number of patches in the longitude dimension.
    n_patch_lon: int
        number of patches in the latitude dimension.
    """
    # Reshape x to three dimensions for downsampling
    x = reshape(x, shape=(x.shape[0], n_patch_vert, n_patch_lat, n_patch_lon, x.shape[-1]))

    # Padding the input to facilitate downsampling
    y1_pad    = (self.downsampling[0] - (x.shape[2] % self.downsampling[0])) % self.downsampling[0] // 2
    y2_pad    = (self.downsampling[0] - (x.shape[2] % self.downsampling[0])) % self.downsampling[0] - y1_pad
    z1_pad    = (self.downsampling[1] - (x.shape[3] % self.downsampling[1])) % self.downsampling[1] // 2
    z2_pad    = (self.downsampling[1] - (x.shape[3] % self.downsampling[1])) % self.downsampling[1] - z1_pad

    x = torch.nn.functional.pad(x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
    n_patch_vert, n_patch_lat, n_patch_lon = x.shape[1:4]
    # Reshape x to facilitate downsampling
    x = reshape(x, shape=(x.shape[0], n_patch_vert, n_patch_lat//2, 2, n_patch_lon//2, 2, x.shape[-1]))
    # Change the order of x
    x = permute(x, (0,1,2,4,3,5,6))
    # Reshape to get a tensor of resolution (8, 180, 91)
    x = reshape(x, shape=(x.shape[0], n_patch_vert*(n_patch_lat//2)*(n_patch_lon//2), 4 * x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x)
    return x
  
class DownSample2D(nn.Module):
  """Downsample 2D."""

  def __init__(self, dim, downsampling=(2,2)):
    """2D Down-sampling operation."""
    super().__init__()
    # A linear function and a layer normalization
    self.linear = Linear(4*dim, 2*dim, bias=False)
    self.norm = LayerNorm(4*dim)
    self.downsampling = downsampling
  
  def forward(self, x, n_patch_lat, n_patch_lon):
    """
    Forward pass of Downsample 2D.

    x: Tensor
        input tensor
    n_patch_lat: int
        number of patches in the latitude dimension.
    n_patch_lon: int
        number of patches in the longitude dimension.

    Returns
    -------
    x: Tensor
    """
    # Reshape x to three dimensions for downsampling
    x = reshape(x, shape=(x.shape[0], n_patch_lon, n_patch_lat, x.shape[-1]))

    # Padding the input to facilitate downsampling
    y1_pad    = (self.downsampling[0] - (x.shape[1] % self.downsampling[0])) % self.downsampling[0] // 2
    y2_pad    = (self.downsampling[0] - (x.shape[1] % self.downsampling[0])) % self.downsampling[0] - y1_pad
    z1_pad    = (self.downsampling[1] - (x.shape[2] % self.downsampling[1])) % self.downsampling[1] // 2
    z2_pad    = (self.downsampling[1] - (x.shape[2] % self.downsampling[1])) % self.downsampling[1] - z1_pad

    x = torch.nn.functional.pad(x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
    n_patch_lat, n_patch_lon = x.shape[1:3]
    # Reshape x to facilitate downsampling
    x = reshape(x, shape=(x.shape[0], n_patch_lat//2, 2, n_patch_lon//2, 2, x.shape[-1]))
    # Change the order of x
    x = permute(x, (0, 1, 3, 2, 4, 5))
    # Reshape to get a tensor of resolution (8, 180, 91)
    x = reshape(x, shape=(x.shape[0], (n_patch_lat//2)*(n_patch_lon//2), 4 * x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x)
    return x

class UpSample(nn.Module):
  """Upsample 3D."""

  def __init__(self, input_dim, output_dim, n_height, n_lat, n_lon, height_crop=(0,0), lat_crop=(0,0), lon_crop=(0,0)):
    """
    Up-sampling operation.
    
    input_dim: int
        dimensions of the input data
    output_dim: int
        dimensions of output 
    n_lat: int
        number of patches in the latitude dimension
    n_lon: int
        number of patches in the longitude dimension
    height_crop: Tuple(int, int)      
        cropping applied to the top and bottom dimensions to obtain the original size of the image
        if downsampled image is (8, 181, 360), height_crop = (0, 0)
        if downsampled image is (8, 91, 180), height_crop = (0, 0)
    lat_crop: Tuple(int, int)
        cropping applied to the left and right dimensions to obtain the original size of the image
        if downsampled image is (8, 181, 360), lat_crop = (0, 1)
        if downsampled image is (8, 91, 180), lat_crop = (0, 1)
    lon_crop: Tuple(int, int)
        cropping applied to the top and bottom dimensions to obtain the original size of the image
        if downsampled image is (8, 181, 360), lon_crop= (0, 0)
        if downsampled image is (8, 91, 180), lon_crop= (0, 0)
    """
    super().__init__()
    self.n_height = n_height
    self.n_lat = n_lat
    self.n_lon = n_lon
    self.height_crop = height_crop
    self.lat_crop = lat_crop
    self.lon_crop = lon_crop

    # Linear layers without bias to increase channels of the data
    self.linear1 = Linear(input_dim, output_dim*4, bias=False)
    
    # Linear layers without bias to mix the data up
    self.linear2 = Linear(output_dim, output_dim, bias=False)

    # Normalization
    self.norm = LayerNorm(output_dim)
  
  def forward(self, x):
    """
    Forward pass of Upsample. 

    x: Tensor
        input tensor

    Returns
    -------
    x: Tensor
    """
    # Call the linear functions to increase channels of the data
    x = self.linear1(x)

    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
    # Reshape x to facilitate upsampling.
    x = reshape(x, shape=(x.shape[0], self.n_height, self.n_lat, self.n_lon, 2, 2, x.shape[-1]//4))
    # Change the order of x
    x = permute(x, (0,1,2,4,3,5,6))
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = reshape(x, shape=(x.shape[0], self.n_height, self.n_lat*2, self.n_lon*2, x.shape[-1]))    

    # Crop the output to the input shape of the network
    x = x[:, self.height_crop[0]:-self.height_crop[1] or None, self.lat_crop[0]:-self.lat_crop[1] or None, self.lon_crop[0]:-self.lon_crop[1] or None, :] 

    # Reshape x back
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x
  
class UpSample2D(nn.Module):
  """Upsample 2D."""

  def __init__(self, input_dim, output_dim, n_lat, n_lon,  lat_crop=(0,0), lon_crop=(0,0)):
    """
    Up-sampling operation.
    
    input_dim: int
        dimensions of the input data
    output_dim: int
        dimensions of output 
    n_lat: int
        number of patches in the latitude dimension
    n_lon: int
        number of patches in the longitude dimension
    lat_crop: Tuple(int, int)
        cropping applied to the left and right dimensions to obtain the original size of the image
        if downsampled image is (8, 181, 360), lat_crop = (0, 1)
        if downsampled image is (8, 91, 180), lat_crop = (0, 1)
    lon_crop: Tuple(int, int)
        cropping applied to the top and bottom dimensions to obtain the original size of the image
        if downsampled image is (8, 181, 360), lon_crop= (0, 0)
        if downsampled image is (8, 91, 180), lon_crop= (0, 0)
    """
    # TODO: should we fix the relationship between the input and output dim? esp. because the output dimension is 
    # changed (x4) here

    super().__init__()
    self.n_lat = n_lat
    self.n_lon = n_lon
    self.lat_crop = lat_crop
    self.lon_crop = lon_crop

    # Linear layers without bias to increase channels of the data
    self.linear1 = Linear(input_dim, output_dim*4, bias=False)
    
    # Linear layers without bias to mix the data up
    self.linear2 = Linear(output_dim, output_dim, bias=False)

    # Normalization
    self.norm = LayerNorm(output_dim)
  
  def forward(self, x):
    """
    Forward pass of Upsample.

    x: Tensor
        input tensor
    """
    # Call the linear functions to increase channels of the data
    x = self.linear1(x)

    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
    # Reshape x to facilitate upsampling.
    x = reshape(x, shape=(x.shape[0], self.n_lat, self.n_lon, 2, 2, x.shape[-1]//4))
    # Change the order of x
    x = permute(x, (0,1,3,2, 4, 5))
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = reshape(x, shape=(x.shape[0], self.n_lat*2, self.n_lon*2, x.shape[-1]))    

    # Crop the output to the input shape of the network
    x = x[:, self.lat_crop[0]:-self.lat_crop[1] or None, self.lon_crop[0]:-self.lon_crop[1] or None, :] 

    # Reshape x back
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x