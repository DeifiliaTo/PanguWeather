import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, Dropout, LayerNorm, Softmax, Parameter
from torch import reshape, arange, zeros, meshgrid, stack, flatten, permute, sum
from torch.nn.functional import pad

from timm.layers import DropPath, trunc_normal_
from Modules.MLP import MLP
import gc

class EarthSpecificLayerBase(nn.Module):
  def __init__(self, depth, dim):
    super().__init__()
    self.depth = depth
    self.dim = dim
    
  def forward(self, x, Z, H, W):
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        self.blocks[i](x, Z, H, W, roll=False)
      else:
        self.blocks[i](x, Z, H, W, roll=True)
    return x
  
class EarthSpecificLayerAbsolute(EarthSpecificLayerBase):
  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    '''Basic layer of our network, contains 2 or 6 blocks'''
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockAbsolute(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
class EarthSpecificLayerRelative(EarthSpecificLayerBase):
  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    '''Basic layer of our network, contains 2 or 6 blocks'''
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockRelative(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
    
class EarthSpecificLayerNoBias(EarthSpecificLayerBase):
  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    '''Basic layer of our network, contains 2 or 6 blocks'''
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockNoBias(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
  
# Assumes implementation for 3D transformer block
class EarthSpecificBlockBase(nn.Module):
  def __init__(self, dim, drop_path_ratio, input_resolution, window_size):
    super().__init__()
    # Define the window size of the neural network 
    self.window_size = window_size

    # Initialize serveral operations
    self.drop_path = DropPath(drop_prob=drop_path_ratio)
    self.norm1  = LayerNorm(dim)
    self.norm2  = LayerNorm(dim)
    self.linear = MLP(dim, 0)
    self.input_resolution = input_resolution
    
    # Only generate masks one time @ initialization
    self.attn_mask = self._gen_mask_(Z=input_resolution[0], H=input_resolution[1], W=input_resolution[2])
    self.zero_mask = torch.zeros(self.attn_mask.shape)
  
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
    # Save the shortcut for skip-connection
    shortcut = x.clone()     

    # Reshape input to three dimensions to calculate window attention
    x = reshape(x, shape=(x.shape[0], Z, H, W, x.shape[2]))
    
    # Zero-pad input accordign to window sizes
    x1_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] // 2
    x2_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] - x1_pad
    y1_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] // 2
    y2_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] - y1_pad
    z1_pad    = (self.window_size[2] - (x.shape[3] % self.window_size[2])) % self.window_size[2] // 2
    z2_pad    = (self.window_size[2] - (x.shape[3] % self.window_size[2])) % self.window_size[2] - z1_pad

    x = pad(x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)
    
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
    
# 3D, absolute
class EarthSpecificBlockAbsolute(EarthSpecificBlockBase):
  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttention3DAbsolute(dim, heads, 0, self.window_size, input_shape)

# 3D, Relative
class EarthSpecificBlockRelative(EarthSpecificBlockBase):
  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttention3DRelative(dim, heads, 0, self.window_size, input_shape)
   
class EarthSpecificBlockNoBias(EarthSpecificBlockBase):
  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttentionNoBias(dim, heads, 0, self.window_size)
     

class EarthSpecificBlock2D(EarthSpecificBlockBase):
  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    # Define the window size of the neural network 
    self.window_size = (6, 12)

    self.attention = EarthAttention2D(dim, heads, 0, self.window_size, input_shape)
    
    # Only generate masks one time @ initialization
    self.attn_mask = self._gen_mask_(H=input_resolution[1], W=input_resolution[2])
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
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
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
      B = int(windows.shape[0] / (H * W) * (window_size[0] * window_size[1]))
      x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
      x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

      return x

  def forward(self, x, Z, H, W, roll):
    # Z = 8,
    # H = 360
    # W = 181

    # Save the shortcut for skip-connection
    shortcut = x.clone()     

    # Reshape input to three dimensions to calculate window attention
    x = reshape(x, shape=(x.shape[0],  H, W, x.shape[2]))
    
    # Zero-pad input accordign to window sizes
    x1_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] // 2
    x2_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] - x1_pad
    y1_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] // 2
    y2_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] - y1_pad
    
    x = pad(x, pad=(0, 0, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0) # TODO check
    
    # Store the shape of the input for restoration
    ori_shape = x.shape

    if roll:
      # Roll x for half of the window for 3 dimensions
      x = torch.roll(x, shifts=(self.window_size[0]//2, self.window_size[1]//2), dims=(1, 2))
    
    # Generate mask of attention masks
    # If two pixels are not adjacent, then mask the attention between them
    # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
    # Reorganize data to calculate window attention 
    x_window = self._window_partition(x, self.window_size) # nW * B, window_size[0], window_size[1], window_size[2], C
    
    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube                   
    x_window = x_window.view(-1, self.window_size[0]*self.window_size[1], x_window.shape[-1])# nW * B, window_size[0], window_size[1], window_size[2], C
    
    # Apply 3D window attention with Earth-Specific bias
    
    if roll:
      attn_windows  = self.attention(x_window, self.attn_mask)
    else:
      attn_windows  = self.attention(x_window, self.zero_mask)

    attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], x.shape[-1])

    x = self._window_reverse(attn_windows, self.window_size, H=x.shape[1], W=x.shape[2])

    # Reshape the tensor back to its original shape
    x = reshape(x, shape=ori_shape) # B, H, W, C

    if roll:
      # Roll x back for half of the window
      x = torch.roll(x, shifts=(-self.window_size[0]//2, -self.window_size[1]//2), dims=(1, 2))

    # Crop the zero padding
    x = x[:, x2_pad:x2_pad+W, y2_pad:y2_pad+W, :]
    
    # Reshape the tensor back to the input shape 
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4])) # TODO modify to delete x.shape[3] and swap [4] with [3]?

    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))

    return x
  

  # Windows that wrap around w dimension do not get masked
  def _gen_mask_(self, Z, H, W):
    img_mask = torch.zeros((1, H, W, 1))  # 1 Z H W 1
    h_slices = (slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.window_size[0]//2),
                slice(-self.window_size[0]//2, None))
    
    cnt = 0
    for h in h_slices:
      img_mask[:, h, :, :] = cnt
      cnt += 1

    # TODO: check implementation with SWIN transformer archittecture
    mask_windows = self._window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
   
class EarthAttentionBase(nn.Module):
  def __init__(self, dim, heads, dropout_rate, window_size):
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
    self.total_window_size = torch.prod(window_size)
  
  def calculate_attention(self, x):
    # Linear layer to create query, key and value
    # Record the original shape of the input BEFORE linear layer (correct?)

    # x shape: (B*nWindows, W0, W1, W2, C)
    original_shape = x.shape 

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
    return attention, value

  def mask_attention(self, x, mask, attention, original_shape):
    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    if x.get_device() < 0: 
      device = 'cpu'
    else:
      device = x.get_device()
    mask = mask.to(device)

    nW = mask.shape[0]
    N = original_shape[1]
    B_ = original_shape[0]

    attention = attention.view(B_ // nW, nW, self.head_number, N, N)
    attention = attention + mask.unsqueeze(1).unsqueeze(0)
    attention = attention.view(-1, self.head_number, N, N)
    return attention

  def activate(self, attention):
    attention = self.Softmax(attention)
    attention = self.dropout(attention)
    return attention
  
  def mixing_linear_layer(self, attention, value, original_shape):
    x = (attention @ value) # @ denote matrix multiplication
    x = x.transpose(1, 2)
    
    # Reshape tensor to the original shape
    x = reshape(x, shape = original_shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x

class EarthAttention3DAbsolute(EarthAttentionBase):
  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    super().__init__()
    
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
    attention, value, original_shape = self.calculate_attention(x)

    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    EarthSpecificBias = self.earth_specific_bias[self.position_index] 

    # Reshape the learnable bias to the same shape as the attention matrix
    EarthSpecificBias = reshape(EarthSpecificBias, shape=(self.total_window_size, self.total_window_size, -1, self.head_number))
    EarthSpecificBias = permute(EarthSpecificBias, (2, 3, 0, 1))


    # Add the Earth-Specific bias to the attention matrix
    attention = attention.view(-1, EarthSpecificBias.shape[0], self.head_number, self.total_window_size, self.total_window_size) + EarthSpecificBias.unsqueeze(0)
    attention = attention.view(-1, self.head_number, self.total_window_size, self.total_window_size)
    attention = self.mask_attention(x, mask, attention, original_shape)
    attention = self.activate(attention)
    
    # Calculated the tensor after spatial mixing.
    x = self.mixing_linear_layer(attention, value, original_shape)

    return x

class EarthAttention3DRelative(EarthAttentionBase):
    def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
        super().__init__(dim, heads, dropout_rate, window_size, input_shape)

        #self.earth_specific_bias = None
        
        self.type_of_windows = 1
        # For each type of window, we will construct a set of parameters according to the paper
        # Making these tensors to be learnable parameters
        self.earth_specific_bias = Parameter(zeros(size=((2 * window_size[2] - 1) * (2 * window_size[1] - 1) * (2  * window_size[0] -1), self.type_of_windows, heads)))

        # Initialize the tensors using Truncated normal distribution
        trunc_normal_(self.earth_specific_bias, std=.02) 

        # Construct position index to reuse self.earth_specific_bias
        self.position_index = self._construct_index()

    def _construct_index(self):
        ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
        # Index in the pressure level of query matrix
        coords_z = arange(self.window_size[0])
        coords_h = arange(self.window_size[1])
        coords_w = arange(self.window_size[2])

        # Change the order of the index to calculate the index in total
        coords_1 = stack(meshgrid([coords_z, coords_h, coords_w]))
        coords_2 = stack(meshgrid([coords_z, coords_h, coords_w]))
        coords_flatten_1 = flatten(coords_1, start_dim=1) 
        coords_flatten_2 = flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = permute(coords, (1, 2, 0))

        # Shift the index for each dimension to start from 0
        coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        coords[:, :, 1] += self.window_size[1] - 1
        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 0] *= 2 * self.window_size[1] - 1

        # Sum up the indexes in three dimensions
        self.position_index = sum(coords, dim=-1)
                                                        
        # Flatten the position index to facilitate further indexing
        self.position_index = flatten(self.position_index)
        return self.position_index
    
    def forward(self, x, mask):
      attention, value, original_shape = self.calculate_attention(x)

      # self.earth_specific_bias is a set of neural network parameters to optimize. 
      EarthSpecificBias = self.earth_specific_bias[self.position_index] 

      # Reshape the learnable bias to the same shape as the attention matrix
      EarthSpecificBias = reshape(EarthSpecificBias, shape=(self.total_window_size, self.total_window_size,  self.head_number))
      EarthSpecificBias = permute(EarthSpecificBias, (2, 0, 1))

      # Add the Earth-Specific bias to the attention matrix
      attention = attention + EarthSpecificBias.unsqueeze(0)
      attention = self.mask_attention(x, mask, attention, original_shape)
      attention = self.activate(attention)
      
      # Calculated the tensor after spatial mixing.
      x = self.mixing_linear_layer(attention, value, original_shape)
      
      return x
  
class EarthAttention2D(EarthAttentionBase):
  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    super().__init__()

    # Record the number of different window types
    self.type_of_windows = (input_shape[0]//window_size[0]) # check

    # Window size: (6, 12)
    # For each type of window, we will construct a set of parameters according to the paper
    self.earth_specific_bias = zeros(size=((2 * window_size[1] - 1) * window_size[0] * window_size[0] , self.type_of_windows, heads))

    # Making these tensors to be learnable parameters
    self.earth_specific_bias = Parameter(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    trunc_normal_(self.earth_specific_bias, std=.02) 

    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index()
    
  # TO BE IMPLEMENTED for 2D
  def _construct_index(self):
    ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
        # Index in the latitude of query matrix
    coords_hi = arange(start=0, end=self.window_size[0])
    # Index in the latitude of key matrix
    coords_hj = -arange(start=0, end=self.window_size[0])*self.window_size[0]

    # Index in the longitude of the key-value pair
    coords_w = arange(start=0, end=self.window_size[1])

    # Change the order of the index to calculate the index in total
    coords_1 = stack(meshgrid([coords_hi, coords_w]))
    coords_2 = stack(meshgrid([coords_hj, coords_w]))
    coords_flatten_1 = flatten(coords_1, start_dim=1) 
    coords_flatten_2 = flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = permute(coords, (1, 2, 0))

    # Shift the index for each dimension to start from 0
    coords[:, :, 1] += self.window_size[1] - 1
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

    attention = self.activate(attention)

    # Calculated the tensor after spatial mixing.
    x = self.mixing_linear_layer(attention, value, original_shape)
    return x
  
class EarthAttentionNoBias(EarthAttentionBase):
  def __init__(self, dim, heads, dropout_rate, window_size):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    super().__init__(dim, heads, dropout_rate, window_size)
    self.type_of_windows = 0
    
    
  def forward(self, x, mask):
    attention, value, original_shape = self.calculate_attention(x)
    
    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    attention = self.mask_attention(x, mask, attention, original_shape)

    attention = self.activate(attention)
    # Calculated the tensor after spatial mixing.
    # Linear layer to post-process operated tensor
    x = self.mixing_linear_layer(attention, value, original_shape)
    return x
