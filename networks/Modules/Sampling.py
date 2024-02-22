import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from torch import reshape, permute

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
  def __init__(self, input_dim, output_dim, nHeight, nLat, nLon):
    '''Up-sampling operation'''
    super().__init__()
    self.nHeight = nHeight
    self.nLat = nLat
    self.nLon = nLon
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
    x = reshape(x, shape=(x.shape[0], self.nHeight, self.nLat, self.nLon, 2, 2, x.shape[-1]//4))
    # Change the order of x
    x = permute(x, (0,1,2,4,3,5,6))
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = reshape(x, shape=(x.shape[0], self.nHeight, self.nLat*2, self.nLon*2, x.shape[-1]))    

    # Crop the output to the input shape of the network
    x = x[:, :, 1:, :, :] # How to communicate cropping efficiently between the down/upsampling dimensions?
    # Reshape x back
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x