import torch.nn as nn

from torch.nn import Linear, Conv3d, Conv2d, ConvTranspose3d, ConvTranspose2d
# Functions in the networks, namely GeLU, DropOut, DropPath, LayerNorm, and Softmax
# GeLU: the GeLU activation function, see Pytorch API or Tensorflow API
# DropOut: the dropout function, available in all deep learning libraries
# DropPath: the DropPath function, see the implementation of vision-transformer, see timm pakage of Pytorch
# A possible implementation of DropPath: from timm.models.layers import DropPath
# LayerNorm: the layer normalization function, see Pytorch API or Tensorflow API
# Softmax: Softmax function, see Pytorch API or Tensorflow API
#from torch.nn import GeLU, DropOut, DropPath, LayerNorm, Softmax
from torch.nn import GELU, Dropout, LayerNorm, Softmax
#from networks.Drop import DropPath
from torch import nn
from timm.layers import DropPath, trunc_normal_

# Common functions for roll, pad, and crop, depends on the data structure of your software environment
#from torch.nn import roll3D, pad3D, pad2D, Crop3D, Crop2D
from torch.nn.functional import pad
from torchvision.transforms.functional import crop

# Common functions for reshaping and changing the order of dimensions
# reshape: change the shape of the data with the order unchanged, see Pytorch API or Tensorflow API
# transpose: change the order of the dimensions, see Pytorch API or Tensorflow API
from torch import reshape, transpose, permute

# Common functions for creating new tensors
# ConstructTensor: create a new tensor with an arbitrary shape
# TruncatedNormalInit: Initialize the tensor with Truncate Normalization distribution
# RangeTensor: create a new tensor like arange(a, b)
#from Your_AI_Library import ConstructTensor, TruncatedNormalInit, RangeTensor
from torch import arange, zeros
# ConstructTensor, TruncatedNormalInit

# Common operations for the data, you may design it or simply use deep learning APIs default operations
# linspace: a tensor version of numpy.linspace
# MeshGrid: a tensor version of numpy.meshgrid
# Stack: a tensor version of numpy.stack
# Flatten: a tensor version of numpy.ndarray.flatten
# TensorSum: a tensor version of numpy.sum
# TensorAbs: a tensor version of numpy.abs
# Concatenate: a tensor version of numpy.concatenate
from torch import linspace, meshgrid, stack, flatten, sum

# Common functions for training models
# LoadModel and SaveModel: Load and save the model, some APIs may require further adaptation to hardwares
# Backward: Gradient backward to calculate the gratitude of each parameters
# UpdateModelParametersWithAdam: Use Adam to update parameters, e.g., torch.optim.Adam
from torch import load

# Custom functions to read your data from the disc
# LoadData: Load the ERA5 data
# LoadConstantMask: Load constant masks, e.g., soil type
# LoadStatic: Load mean and std of the ERA5 training data, every fields such as T850 is treated as an image and calculate the mean and std
# from Your_Data_Code import LoadData, LoadConstantMask, LoadStatic

from DataLoader import LoadConstantMask
from Tools import TruncatedNormalInit
from torch.nn import Parameter
import torch
import numpy as np

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
    self.layer1 = EarthSpecificLayer(2, self.C, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360))
    self.layer2 = EarthSpecificLayer(6, 2*self.C, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180))
    self.layer3 = EarthSpecificLayer(6, 2*self.C, drop_list[2:], 12, input_shape=[8, 96], device=device, input_resolution=(8, 96, 180))
    self.layer4 = EarthSpecificLayer(2, self.C, drop_list[:2], 6,  input_shape=[8, 186], device=device, input_resolution=(8, 186, 360))

    # Upsample and downsample
    self.upsample = UpSample(self.C*2, self.C)

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
    topography = torch.broadcast_to(self.topography, (input_surface_shape[0], 1, input_surface_shape[2], input_surface_shape[3]))
    
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

class DownSample(nn.Module):
  def __init__(self, dim, downsampling=(2,2)):
    '''Down-sampling operation'''
    super().__init__()
    # A linear function and a layer normalization
    self.linear = Linear(4*dim, 2*dim, bias=False)

    self.norm = LayerNorm(4*dim)
    self.downsampling = downsampling
  
  def forward(self, x, Z, H, W):
    # Reshape x to three dimensions for downsampling
    x = reshape(x, shape=(x.shape[0], Z, H, W, x.shape[-1]))

    # Padding the input to facilitate downsampling
    y1_pad    = (self.downsampling[0] - (x.shape[2] % self.downsampling[0])) % self.downsampling[0] // 2
    y2_pad    = (self.downsampling[0] - (x.shape[2] % self.downsampling[0])) % self.downsampling[0] - y1_pad
    z1_pad    = (self.downsampling[1] - (x.shape[3] % self.downsampling[1])) % self.downsampling[1] // 2
    z2_pad    = (self.downsampling[1] - (x.shape[3] % self.downsampling[1])) % self.downsampling[1] - z1_pad

    x = torch.nn.functional.pad(x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
    Z, H, W = x.shape[1:4]
    # Reshape x to facilitate downsampling
    x = reshape(x, shape=(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1]))
    # Change the order of x
    x = permute(x, (0,1,2,4,3,5,6))
    # Reshape to get a tensor of resolution (8, 180, 91)
    x = reshape(x, shape=(x.shape[0], Z*(H//2)*(W//2), 4 * x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x)
    return x

class UpSample(nn.Module):
  def __init__(self, input_dim, output_dim):
    '''Up-sampling operation'''
    super().__init__()
    # Linear layers without bias to increase channels of the data
    self.linear1 = Linear(input_dim, output_dim*4, bias=False)
    
    # Linear layers without bias to mix the data up
    self.linear2 = Linear(output_dim, output_dim, bias=False)

    # Normalization
    self.norm = LayerNorm(output_dim)
  
  def forward(self, x):
    # Call the linear functions to increase channels of the data
    x = self.linear1(x)

    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
    # Reshape x to facilitate upsampling.
    x = reshape(x, shape=(x.shape[0], 8, 91, 180, 2, 2, x.shape[-1]//4))
    # Change the order of x
    x = permute(x, (0,1,2,4,3,5,6))
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = reshape(x, shape=(x.shape[0], 8, 182, 360, x.shape[-1]))    

    # Crop the output to the input shape of the network
    x = x[:, :, 1:, :, :] # How to communicate cropping efficiently between the down/upsampling dimensions?
    # Reshape x back
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x
  
class EarthSpecificLayer(nn.Module):
  # depth = 2
  # dim   = 192
  # drop_path_ratio_list: drop_list[:2]
  # heads = 6
  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution):
    '''Basic layer of our network, contains 2 or 6 blocks'''
    super().__init__()
    self.depth = depth
    self.dim = dim  # TODO : can remove dim l8r on?
    
    # Construct basic blocks
    self.blocks = torch.nn.ModuleList([EarthSpecificBlock(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution).to(device) for i in torch.arange(depth)])    
    
  def forward(self, x, Z, H, W):
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        self.blocks[i](x, Z, H, W, roll=False)
      else:
        self.blocks[i](x, Z, H, W, roll=True)
    return x

class EarthSpecificBlock(nn.Module):
  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    super().__init__()
    # Define the window size of the neural network 
    self.window_size = (2, 6, 12)

    # Initialize serveral operations
    self.drop_path = DropPath(drop_prob=drop_path_ratio)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.linear = MLP(dim, 0)
    self.attention = EarthAttention3D(dim, heads, 0, self.window_size, input_shape)

    # Only generate masks one time @ initialization
    self.attn_mask = self._gen_mask_(Z=input_resolution[0], H=input_resolution[1], W=input_resolution[2])
    self.zero_mask = torch.zeros(self.attn_mask.shape)
    self.input_resolution = input_resolution

  def _window_partition(self, x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, Z, H, W, C = x.shape
    x = x.view(B, Z // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

  def _window_reverse(self, windows, window_size, Z, H, W):
      """
      Args:
          windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
          window_size (int): Window size
          Z (int): Number of pressure heights
          H (int): Height of image
          W (int): Width of image

      Returns:
          x: (B, Z, H, W, C)
      """
      B = int(windows.shape[0] / (Z * H * W) * (window_size[0] * window_size[1] * window_size[2]))
      x = windows.view(B, Z // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
      x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, Z, H, W, -1)
      return x

  def forward(self, x, Z, H, W, roll):
    # Z = 8,
    # H = 360
    # W = 181

    # Save the shortcut for skip-connection
    shortcut = x.clone()     

    # Reshape input to three dimensions to calculate window attention
    x = reshape(x, shape=(x.shape[0], Z, H, W, x.shape[2]))
    
    # Zero-pad input if needed
    # TODO: padding according to window size. Correct?
    x1_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] // 2
    x2_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] - x1_pad
    y1_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] // 2
    y2_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] - y1_pad
    z1_pad    = (self.window_size[2] - (x.shape[3] % self.window_size[2])) % self.window_size[2] // 2
    z2_pad    = (self.window_size[2] - (x.shape[3] % self.window_size[2])) % self.window_size[2] - z1_pad

    x = torch.nn.functional.pad(x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)
    
    # Store the shape of the input for restoration
    ori_shape = x.shape

    if roll:
      # Roll x for half of the window for 3 dimensions
      x = torch.roll(x, shifts=(self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2), dims=(1, 2, 3))
    
    # Generate mask of attention masks
    # If two pixels are not adjacent, then mask the attention between them
    # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
    # Reorganize data to calculate window attention 
    x_window = self._window_partition(x, self.window_size) # nW * B, window_size[0], window_size[1], window_size[2], C
    
    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube                   
    x_window = x_window.view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], x_window.shape[-1])# nW * B, window_size[0], window_size[1], window_size[2], C
    
    # Apply 3D window attention with Earth-Specific bias
    
    if roll:
      attn_windows  = self.attention(x_window, self.attn_mask)
    else:
      attn_windows  = self.attention(x_window, self.zero_mask)

    attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], x.shape[-1])

    x = self._window_reverse(attn_windows, self.window_size, Z=x.shape[1], H=x.shape[2], W=x.shape[3])

    # Reshape the tensor back to its original shape
    x = reshape(x, shape=ori_shape) # B, Z, H, W, C

    if roll:
      # Roll x back for half of the window
      x = torch.roll(x, shifts=(-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2), dims=(1, 2, 3))

    # Crop the zero padding
    x = x[:, x2_pad:x2_pad+Z, y2_pad:y2_pad+H, z2_pad:z2_pad+W, :]
    
    # Reshape the tensor back to the input shape
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]))

    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))

    return x
  

  # Windows that wrap around w dimension do not get masked
  def _gen_mask_(self, Z, H, W):
    img_mask = torch.zeros((1, Z, H, W, 1))  # 1 Z H W 1
    z_slices = (slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.window_size[0]//2),
                slice(-self.window_size[0]//2, None))
    h_slices = (slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.window_size[1]//2),
                slice(-self.window_size[1]//2, None))
    
    cnt = 0
    for z in z_slices:
      for h in h_slices:
          
          img_mask[:, z, h, :, :] = cnt
          cnt += 1

    mask_windows = self._window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1]* self.window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
    
class EarthAttention3D(nn.Module):
  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    super().__init__()

    # Initialize several operations
    # Should make sense to use dim*3 to generate the vectors for a qkv matrix
    self.linear1 = Linear(dim, dim*3, bias=True)
    self.linear2 = Linear(dim, dim)
    self.Softmax = Softmax(dim=-1)
    self.dropout = Dropout(dropout_rate)

    # Store several attributes
    self.head_number = heads
    self.dim = dim
    self.scale = (dim//heads)**-0.5
    self.window_size = window_size

    # TODO: input_shape
    # This is currently nonsensical
    # input_shape = [640, 144] # what does this represent? # this still doesnt make sense
    # input_shape is current shape of the self.forward function
    # You can run your code to record it, modify the code and rerun it

    # Record the number of different window types
    self.type_of_windows = (input_shape[0]//window_size[0])*(input_shape[1]//window_size[1])

    # For each type of window, we will construct a set of parameters according to the paper
    self.earth_specific_bias = zeros(size=((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads))

    # Making these tensors to be learnable parameters
    self.earth_specific_bias = Parameter(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    trunc_normal_(self.earth_specific_bias, std=.02) 

    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index()
    
    
  def _construct_index(self):
    ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
    # Index in the pressure level of query matrix
    coords_zi = arange(start=0, end=self.window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -arange(start=0, end=self.window_size[0])*self.window_size[0]

    # Index in the latitude of query matrix
    coords_hi = arange(start=0, end=self.window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -arange(start=0, end=self.window_size[1])*self.window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = arange(start=0, end=self.window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 = stack(meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = stack(meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = flatten(coords_1, start_dim=1) 
    coords_flatten_2 = flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = permute(coords, (1, 2, 0))

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += self.window_size[2] - 1
    coords[:, :, 1] *= 2 * self.window_size[2] - 1
    coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

    # Sum up the indexes in three dimensions
    self.position_index = sum(coords, dim=-1)
                                                    
    # Flatten the position index to facilitate further indexing
    self.position_index = flatten(self.position_index)
    return self.position_index
    
  def forward(self, x, mask):
    # Linear layer to create query, key and value
    # Record the original shape of the input BEFORE linear layer (correct?)

    # port mask onto device??? 
    if x.get_device() < 0: 
      device = 'cpu'
    else:
      device = x.get_device()
    mask = mask.to(device)

    # x shape: (B*nWindows, W0, W1, W2, C)
    original_shape = x.shape 
    B_ = original_shape[0]
    B  = B_ // self.type_of_windows
    
    # x shape: (B*nWindows, W0*W1*W2, C)
    x = self.linear1(x)

    # reshape the data to calculate multi-head attention
    qkv = reshape(x, shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)) 
    query, key, value = permute(qkv, (2, 0, 3, 1, 4))

    # Scale the attention
    query = query * self.scale

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    # Attention shape: nB * nWindows, nHeads, N, N (N = W0*W1*W2)
    attention = (query @ key.transpose(-2, -1)) # @ denotes matrix multiplication
  
    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    EarthSpecificBias = self.earth_specific_bias[self.position_index.repeat(attention.shape[0] // self.type_of_windows)] 

    # Reshape the learnable bias to the same shape as the attention matrix
    EarthSpecificBias = reshape(EarthSpecificBias, shape=(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], -1, self.head_number))
    EarthSpecificBias = permute(EarthSpecificBias, (2, 3, 0, 1))

    # Add the Earth-Specific bias to the attention matrix
    attention = attention + EarthSpecificBias 

    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    nW = mask.shape[0]
    N = original_shape[1]

    attention = attention.view(B_ // nW, nW, self.head_number, N, N)
    attention = attention + mask.unsqueeze(1).unsqueeze(0)
    attention = attention.view(-1, self.head_number, N, N)

    attention = self.Softmax(attention)
    attention = self.dropout(attention)

    # Calculated the tensor after spatial mixing.
    x = (attention @ value) # @ denote matrix multiplication
    x = x.transpose(1, 2)
    
    # Reshape tensor to the original shape
    x = reshape(x, shape = original_shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x
  
class MLP(nn.Module):
  def __init__(self, dim, dropout_rate):
    super().__init__()
    '''MLP layers, same as most vision transformer architectures.'''
    self.linear1 = Linear(dim, dim * 4)
    self.linear2 = Linear(dim * 4, dim)
    self.activation = GELU()
    self.drop = Dropout(p=dropout_rate)
    
  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.drop(x)
    x = self.linear2(x)
    x = self.drop(x)
    return x

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
