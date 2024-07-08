import torch
import torch.nn as nn
from Modules.MLP import MLP
from timm.layers import DropPath, trunc_normal_
from torch import arange, flatten, meshgrid, permute, reshape, stack, sum, zeros
from torch.nn import Dropout, LayerNorm, Linear, Parameter, Softmax
from torch.nn.functional import pad
from torch.utils.checkpoint import checkpoint


class EarthSpecificLayerBase(nn.Module):
  """Base class for defining a series of attention blocks."""

  def __init__(self, depth, dim):
    """
    Initialize the EarthSpecificLayer.

    depth: int
        Number of identical sequential layers
    dim: int
        Hidden dimension of the attention mechanism.  
    """
    super().__init__()
    self.depth = depth
    self.dim = dim
    
  def forward(self, x, n_patch_vert, n_patch_lat, n_patch_lon):
    """
    Forward pass of the EarthSpecificLayer. The input is rolled every other layer. Checkpoint activation is used to reduce memory requirements.

    x: Tensor
      Input of shape (n_batch, n_patch_vert*n_patch_lat*n_patch_lon, hidden_dim)
    n_patch_vert: int
      Number of patches in the vertical (first) dimension
    n_patch_lat: int
      Number of patches in the latitude (second) dimension
    n_patch_lon: int
      Number of patches in the longitude (third) dimension

    Returns
    -------
    x: Tensor
    """
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        x = checkpoint(self.blocks[i], x, n_patch_vert, n_patch_lat, n_patch_lon, False, use_reentrant=False)
        torch.cuda.empty_cache()
      else:
        x = checkpoint(self.blocks[i], x, n_patch_vert, n_patch_lat, n_patch_lon, True, use_reentrant=False)
        torch.cuda.empty_cache()
    return x
  
class EarthSpecificLayerAbsolute(EarthSpecificLayerBase):
  """Earth-Specific Layer for absolute positional bias term within attention mechanism."""

  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    """
    Initialize layer of network that calls EarthSpecificBlockAbsolute.
    
    depth: int
        Number of layers
    dim: int
        Size of hidden dimension
    drop_path_ratio_list: List[float]
        List of length = layers specifying drop_path_ratio for each layer.
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockAbsolute(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
class EarthSpecificLayerRelative(EarthSpecificLayerBase):
  """Earth-Specific Layer for absolute positional bias term within attention mechanism."""

  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    """
    Initialize layer of network that calls EarthSpecificBlockRelative.
    
    depth: int
        Number of layers
    dim: int
        Size of hidden dimension
    drop_path_ratio_list: List[float]
        List of length = layers specifying drop_path_ratio for each layer.
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockRelative(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
    
class EarthSpecificLayerNoBias(EarthSpecificLayerBase):
  """Earth-Specific Layer for attention mechanism with no bias term."""

  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([2, 6, 12])):
    """
    Initialize layer of network that calls EarthSpecificBlockNoBias.

    depth: int
        Number of layers
    dim: int
        Size of hidden dimension
    drop_path_ratio_list: List[float]
        List of length = layers specifying drop_path_ratio for each layer.
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlockNoBias(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    

class EarthSpecificLayer2D(EarthSpecificLayerBase):
  """Earth-Specific Layer for 2D attention mechanism with absolute positional bais term within attention mechanism."""

  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([6, 12])):
    """
    Initialize layer of network that calls EarthSpecificBlock2D.
    
    depth: int
        Number of layers
    dim: int
        Size of hidden dimension
    drop_path_ratio_list: List[float]
        List of length = layers specifying drop_path_ratio for each layer.
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlock2D(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
  def forward(self, x, n_patch_lat, n_patch_lon):
    """
    Forward pass of Earth-Specific layers for 2D attention mechanism with absolute positional bias term.
    
    x: Tensor
        Input to model of shape (n_batch, n_patch_lat*n_patch_lon, hidden_dim)
    n_patch_lat: int
        Number of patches in the latitude dimension.
    n_patch_lon: int
        Number of patches in the longitude dimension.

    Returns
    -------
    x: Tensor
    """
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        x = checkpoint(self.blocks[i], x, n_patch_lat, n_patch_lon, False, use_reentrant=False)
        torch.cuda.empty_cache()
      else:
        x = checkpoint(self.blocks[i], x, n_patch_lat, n_patch_lon, True, use_reentrant=False)
        torch.cuda.empty_cache()
    return x
  
class EarthSpecificLayer2DNoBias(EarthSpecificLayerBase):
  """Earth-Specific Layer for 2D attention mechanism without bias term."""

  def __init__(self, depth, dim, drop_path_ratio_list, heads, input_shape, device, input_resolution, window_size=torch.tensor([6, 12])):
    """
    Initialize layer of network that calls EarthSpecificBlock2DNoBias.

    depth: int
        Number of layers
    dim: int
        Size of hidden dimension
    drop_path_ratio_list: List[float]
        List of length = layers specifying drop_path_ratio for each layer.
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(depth, dim)
    
    # Construct basic blocks
    self.blocks = nn.ModuleList([EarthSpecificBlock2DNoBias(dim, drop_path_ratio_list[i], heads, input_shape=input_shape,device=device, input_resolution=input_resolution, window_size=window_size).to(device) for i in torch.arange(depth)])    
    
  def forward(self, x, n_patch_lat, n_patch_lon):
    """
    Forward pass of Earth-Specific layers for 2D attention mechanism with absolute positional bias term.
    
    x: Tensor
        Input to model of shape (B, n_patch_lat, n_patch_lon, n_variables*n_patch_vert) #TODO correct?
    n_patch_lat: int
        Number of patches in the latitude dimension.
    n_patch_lon: int
        Number of patches in the longitude dimension.

    Returns
    -------
    x: Tensor
    """
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        x = checkpoint(self.blocks[i], x, n_patch_lat, n_patch_lon, False, use_reentrant=False)
        torch.cuda.empty_cache()
      else:
        x = checkpoint(self.blocks[i], x, n_patch_lat, n_patch_lon, True, use_reentrant=False)
        torch.cuda.empty_cache()
    return x
  
# Provides base for EarthSpecificBlock
class EarthSpecificBlockBase(nn.Module):
  """Base class for one block of the attention mechanism."""

  def __init__(self, dim, drop_path_ratio, input_resolution, window_size):
    """
    Initialize attention block.

    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that sample will be dropped.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__()
    # Define the window size of the neural network 
    self.window_size = window_size

    # Initialize several operations
    self.drop_path = DropPath(drop_prob=drop_path_ratio)
    self.norm1  = LayerNorm(dim)
    self.norm2  = LayerNorm(dim)
    self.linear = MLP(dim, 0)
    self.input_resolution = input_resolution
    
  # Windows that wrap around w dimension do not get masked
  def _gen_mask_(self, n_patch_vert, n_patch_lat, n_patch_lon):
    """
    Generate mask for attention mechanism when fields are rolled. 

    n_patch_vert: int
        Number of patches in the vertical (first) dimension
    n_patch_lat: int
        Number of patches in the latitude (second) dimension
    n_patch_lon: int
        Number of patches in the longitude (third) dimension

    Returns
    -------
    attention_mask: Tensor 
        of shape (n_windows, n_patch_in_window, n_patch_in_window) #TODO: verify
        attention mask with value of 0 when attention should not be masked and -10000 when masked.
    """
    img_mask = torch.zeros((1, n_patch_vert, n_patch_lat, n_patch_lon, 1))  # 1 n_patch_vert n_patch_lat n_patch_lon 1
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

    mask_windows = self._window_partition(img_mask, self.window_size)  # n_windows, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1]* self.window_size[2])
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-10000.0)).masked_fill(attention_mask == 0, float(0.0))
    return attention_mask
  
  def _window_partition(self, x, window_size):
    """
    Partitions input tensor into windows over which attention will be calculated.
    
    x: Tensor
        of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
    window_size: Tensor
        of length 3, describing window size in (vert, lat, lon) dimensions
        = (window_size_vert, window_size_lat, window_size_lon)

    Returns
    -------
    windows: Tensor
        of shape(num_windows*n_batch, window_size_vert, window_size_lat, window_size_lon, C)
    """
    n_batch, n_patch_vert, n_patch_lat, n_patch_lon, hidden_dim = x.shape
    x = x.view(n_batch, n_patch_vert // window_size[0], window_size[0], n_patch_lat // window_size[1], window_size[1], n_patch_lon // window_size[2], window_size[2], hidden_dim)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], hidden_dim)
    return windows

  def _window_reverse(self, windows, window_size, n_patch_vert, n_patch_lat, n_patch_lon):
      """
      Reverse window operation to retrieve x Tensor in original shape.

      windows: Tensor
          of shape (num_windows*n_batch, window_size_vert, window_size_lat, window_size_lon, C) == nB,W0,W1,C
      window_size: Tensor
        of length 3, describing window size in (vert, lat, lon) dimensions

      n_patch_vert: int
        Number of patches in the vertical (first) dimension
      n_patch_lat: int
          Number of patches in the latitude (second) dimension
      n_patch_lon: int
          Number of patches in the longitude (third) dimension

      Returns
      -------
      x: Tensor
          of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
      """
      n_batch = int(windows.shape[0] / (n_patch_vert * n_patch_lat * n_patch_lon) * (window_size[0] * window_size[1] * window_size[2]))
      x = windows.view(n_batch, n_patch_vert // window_size[0], n_patch_lat // window_size[1], n_patch_lon // window_size[2], window_size[0], window_size[1], window_size[2], -1)
      x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
      return x
  
  def forward(self, x, n_patch_vert, n_patch_lat, n_patch_lon, roll):
    """
    Forward pass of a single attention block.

    n_patch_vert: int
        Number of patches in the vertical (first) dimension
    n_patch_lat: int
        Number of patches in the latitude (second) dimension
    n_patch_lon: int
        Number of patches in the longitude (third) dimension
    roll: bool
        Specifying if the layer is rolled
    
    Returns
    -------
    x: Tensor
        of shape (B, n_patch_vert*n_patch_lat*n_patch_lon, C)
    """
    # Save the shortcut for skip-connection
    shortcut = x.clone()     

    # Reshape input to three dimensions to calculate window attention
    x = reshape(x, shape=(x.shape[0], n_patch_vert, n_patch_lat, n_patch_lon, x.shape[2]))
    
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
    x_window = self._window_partition(x, self.window_size) # n_windows * B, window_size[0], window_size[1], window_size[2], C
    
    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube                   
    x_window = x_window.view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], x_window.shape[-1])# n_windows * B, window_size[0], window_size[1], window_size[2], C
    
    # Apply 3D window attention with Earth-Specific bias
    
    mask = self._gen_mask_(n_patch_vert=ori_shape[1], n_patch_lat=ori_shape[2], n_patch_lon=ori_shape[3]).to(torch.float32)  
    
    if roll:
      attention_windows  = self.attention(x_window, mask)
    else:
      zero_mask = torch.zeros(mask.shape).to(torch.float32)
      attention_windows  = self.attention(x_window, zero_mask)

    attention_windows = attention_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], x.shape[-1])

    x = self._window_reverse(attention_windows, self.window_size, n_patch_vert=x.shape[1], n_patch_lat=x.shape[2], n_patch_lon=x.shape[3])
    # Reshape the tensor back to its original shape
    x = reshape(x, shape=ori_shape) # B, n_patch_vert, n_patch_lat, n_patch_lon, C 

    if roll:
      # Roll x back for half of the window
      x = torch.roll(x, shifts=(-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2), dims=(1, 2, 3))

    # Crop the zero padding
    x = x[:, x1_pad:x1_pad+n_patch_vert, y1_pad:y1_pad+n_patch_lat, z1_pad:z1_pad+n_patch_lon, :]
    
    # Reshape the tensor back to the input shape
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]))

    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))

    return x
    
class EarthSpecificBlockAbsolute(EarthSpecificBlockBase):
  """Class for one block of the 3D attention mechanism with absolute positional bias term and window attention."""

  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    """
    Initialize 3D Transformer block with absolute positional bias term.
    
    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that a sample will be dropped during training
    heads: int
        Number of attention heads
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttention3DAbsolute(dim, heads, 0, self.window_size, input_shape)

# 3D, Relative
class EarthSpecificBlockRelative(EarthSpecificBlockBase):
  """Class for one block of the 3D attention mechanism with relative positional bias term and window attention."""

  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    """
    Initialize 3D Transformer block with relative positional bias term.
    
    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that a sample will be dropped during training
    heads: int
        Number of attention heads
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttention3DRelative(dim, heads, 0, self.window_size, input_shape, device)
   
class EarthSpecificBlockNoBias(EarthSpecificBlockBase):
  """Class for one block of the 3D attention mechanism with no positional bias term and window attention."""

  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    """
    Initialize 3D Transformer block with no bias term.
    
    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that a sample will be dropped during training
    heads: int
        Number of attention heads
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (8, 186, 360)
        For Pangu-Weather-Lite, input resolution is (8, 93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    
    self.attention = EarthAttentionNoBias(dim, heads, 0, self.window_size)
     

class EarthSpecificBlock2D(EarthSpecificBlockBase):
  """Class for one block of the 2D attention mechanism with absolute positional bias term."""

  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    """
    Initialize 2D Transformer block with absolute positional bias term.
    
    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that a sample will be dropped during training
    heads: int
        Number of attention heads
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 2 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (186, 360)
        For Pangu-Weather-Lite, input resolution is (93, 180)
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(dim, drop_path_ratio, input_resolution, window_size)
    # Define the window size of the neural network 
    self.attention = EarthAttention2D(dim, heads, 0, self.window_size, input_shape)
    
    # Only generate masks one time @ initialization
    self.input_resolution = input_resolution

  def _window_partition(self, x, window_size):
    """
    Partitions input tensor into windows over which attention will be calculated.
    
    x: Tensor
        of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
    window_size: Tensor
        of length 2, describing window size in (vert, lat, lon) dimensions
        = (window_size_lat, window_size_lon)

    Returns
    -------
    windows: Tensor
        of shape(num_windows*n_batch, window_size_lat, window_size_lon, C)
    """
    n_batch, n_patch_lat, n_patch_lon, hidden_dim = x.shape
    x = x.view(n_batch, n_patch_lat // window_size[0], window_size[0], n_patch_lon // window_size[1], window_size[1], hidden_dim)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], hidden_dim)
    return windows

  def _window_reverse(self, windows, window_size, n_patch_lat, n_patch_lon):
      """
      Reverse window operation to retrieve x Tensor in original shape.

      windows: Tensor
          of shape (num_windows*B, window_size_lat, window_size_lon, C) == nB,W0,W1,C
      window_size: Tensor
        of length 3, describing window size in (vert, lat, lon) dimensions
      n_patch_lat: int
          Number of patches in the latitude dimension
      n_patch_lon: int
          Number of patches in the longitude dimension

      Returns
      -------
      x: Tensor
          of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
      """
      n_batch = int(windows.shape[0] / (n_patch_lat * n_patch_lon) * (window_size[0] * window_size[1]))
      x = windows.view(n_batch, n_patch_lat // window_size[0], n_patch_lon // window_size[1], window_size[0], window_size[1], -1)
      x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(n_batch, n_patch_lat, n_patch_lon, -1)

      return x

  def forward(self, x, n_patch_lat, n_patch_lon, roll):
    """
    Forward pass of a single 2D attention block.

    n_patch_lat: int
        Number of patches in the latitude dimension
    n_patch_lon: int
        Number of patches in the longitude dimension
    roll: bool
        Specifying if the layer is rolled
    
    Returns
    -------
    x: Tensor
        of shape (B, n_patch_lat*n_patch_lon, C)
    """
    # Save the shortcut for skip-connection
    shortcut = x.clone()     
    
    # Reshape input to three dimensions to calculate window attention
    x = reshape(x, shape=(x.shape[0], n_patch_lat, n_patch_lon, x.shape[2]))
    
    # Zero-pad input accordign to window sizes
    x1_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] // 2
    x2_pad    = (self.window_size[0] - (x.shape[1] % self.window_size[0])) % self.window_size[0] - x1_pad
    y1_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] // 2
    y2_pad    = (self.window_size[1] - (x.shape[2] % self.window_size[1])) % self.window_size[1] - y1_pad
    
    x = pad(x, pad=(0, 0, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0) 
    
    # Store the shape of the input for restoration
    ori_shape = x.shape

    if roll:
      # Roll x for half of the window for 3 dimensions
      x = torch.roll(x, shifts=(self.window_size[0]//2, self.window_size[1]//2), dims=(1, 2))
    
    # Generate mask of attention masks
    # If two pixels are not adjacent, then mask the attention between them
    # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
    # Reorganize data to calculate window attention 
    x_window = self._window_partition(x, self.window_size) # n_windows * B, window_size[0], window_size[1], window_size[2], C
    
    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube                   
    x_window = x_window.view(-1, self.window_size[0]*self.window_size[1], x_window.shape[-1])# n_windows * B, window_size[0], window_size[1], window_size[2], C
    
    mask = self._gen_mask_(n_patch_lat=ori_shape[1], n_patch_lon=ori_shape[2]).to(torch.float32)  

    # Apply 3D window attention with Earth-Specific bias
    if roll:      
      attention_windows  = self.attention(x_window, mask)
    else:
      zero_mask = torch.zeros(mask.shape).to(torch.float32)
      attention_windows  = self.attention(x_window, zero_mask)

    attention_windows = attention_windows.view(-1, self.window_size[0], self.window_size[1], x.shape[-1])

    x = self._window_reverse(attention_windows, self.window_size, n_patch_lat=x.shape[1], n_patch_lon=x.shape[2])

    # Reshape the tensor back to its original shape
    x = reshape(x, shape=ori_shape) # B, n_patch_lat, n_patch_lon, C

    if roll:
      # Roll x back for half of the window
      x = torch.roll(x, shifts=(-self.window_size[0]//2, -self.window_size[1]//2), dims=(1, 2))

    # Crop the zero padding
    x = x[:, x2_pad:x2_pad+n_patch_lat, y2_pad:y2_pad+n_patch_lon, :]
    
    # Reshape the tensor back to the input shape 
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))

    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))

    return x
  
  def _gen_mask_(self, n_patch_lat, n_patch_lon):
    """
    Generate mask for attention mechanism when fields are rolled. 

    n_patch_vert: int
        Number of patches in the vertical (first) dimension
    n_patch_lat: int
        Number of patches in the latitude (second) dimension
    n_patch_lon: int
        Number of patches in the longitude (third) dimension

    Returns
    -------
    attention_mask: Tensor 
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
        attention mask with value of 0 when attention should not be masked and -10000 when masked.
    """
    img_mask = torch.zeros((1, n_patch_lat, n_patch_lon, 1))  # 1 n_patch_lat n_patch_lon 1
    h_slices = (slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.window_size[0]//2),
                slice(-self.window_size[0]//2, None))
    
    cnt = 0
    for h in h_slices:
      img_mask[:, h, :, :] = cnt
      cnt += 1

    # TODO: check implementation with SWIN transformer archittecture
    mask_windows = self._window_partition(img_mask, self.window_size)  # n_windows, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-10000.0)).masked_fill(attention_mask == 0, float(0.0))
    return attention_mask
  
class EarthSpecificBlock2DNoBias(EarthSpecificBlock2D):
  """Class for one block of the 2D attention mechanism with no bias term."""

  def __init__(self, dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size):
    """
    Initialize class for 2D attention mechanism with no bias.

    dim: int
        Size of hidden dimension
    drop_path_ratio: float
        Probability that a sample will be dropped during training
    heads: int
        Number of attention heads
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    device: String
        device that the code is being run on.
    input_resolution: List[int]
        List of length 2 that specifies the dimensions after the patch embedding step incl. padding.
        For Pangu-Weather, input resolution is (186, 360)
        For Pangu-Weather-Lite, input resolution is (93, 180)
    window_size: Tensor
        Tensor of length(2) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
    super().__init__(dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size)
    # Define the window size of the neural network 
    
    self.attention = EarthAttention2DNoBias(dim, heads, 0, self.window_size, input_shape)
    
    # Only generate masks one time @ initialization
    self.input_resolution = input_resolution

   
class EarthAttentionBase(nn.Module):
  """Base class for attention mechanism."""

  def __init__(self, dim, heads, dropout_rate, window_size):
    """
    Initialize base class for attention.

    dim: int
        Size of hidden dimension
    heads: int
        Number of attention heads
    dropout_rate: float
        probability that value is dropped during training
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    """
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
    """
    Calculate attention.

    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)

    Returns
    -------
    attention: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
    value: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
    original_shape: torch.Size
        of x at input
    """
    # Linear layer to create query, key and value
    # Record the original shape of the input BEFORE linear layer (correct?)
    original_shape = x.shape # x shape: (B*n_windows, W0*W1*W2, dim)
    
    x = self.linear1(x)      # x shape: (B*n_windows, W0*W1*W2, 3*dim)
    # reshape the data to calculate multi-head attention
    # q shape: (B*n_windows, W0*W1*W2, 3, nHead, dim/nHead)
    qkv = reshape(x, shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)) 
    # query after permute: (nB, nHead, W0*W1*W2, C/nHead)
    query, key, value = permute(qkv, (2, 0, 3, 1, 4))
    # Scale the attention
    query = query * self.scale

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    # Attention shape: B*n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
    attention = (query @ key.transpose(-2, -1)) # @ denotes matrix multiplication
    return attention, value, original_shape

  def mask_attention(self, x, mask, attention, original_shape):
    """
    Masks attention between non-adjacent pixels. -10000 is added to masked elements.

    x: Tensor
    mask: Tensor
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
    attention: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
    value: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
    original_shape: torch.Size
        of x at input = (n_batch * n_windows, n_patch_in_window, hidden_dim)

    Returns
    -------
    attention: Tensor
        masked attention of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
    """
    if attention.get_device() < 0: 
      device = 'cpu'
    else:
      device = attention.get_device()
    mask = mask.to(device)

    # attention shape: n_windows * B, nHeads, W[0]*W[1]*W[2], W[0]*W[1]*W[2]
    # mask shape: n_windows, W[0]*W[1]*W[2], W[0]*W[1]*W[2]
    n_windows = mask.shape[0]  # n_windows
    n_patch_in_window = original_shape[1] # W0*W1*W2
    n_windows_total = original_shape[0] # B*n_windows

    # original attention shape: (n_windows * B, nHead, W0*W1*W2, W0*W1*W2)
    attention = attention.view(n_windows_total // n_windows, n_windows, self.head_number, n_patch_in_window, n_patch_in_window) # (B, n_windows, nHead, W0*W1*W2, W0*W1*W2)
    attention = attention + mask.unsqueeze(1).unsqueeze(0)           # (B, n_windows, nHead, W0*W1*W2, W0*W1*W2) + (1,  n_windows, 1, W0*W1*W2, W0*W1*W2)
    attention = attention.view(n_windows_total, self.head_number, n_patch_in_window, n_patch_in_window)           # (B*n_windows, nHead, W0*W1*W2, W0*W1*W2)
    return attention

  def activate(self, attention):
    """
    Apply Softmax activation to attention.

    attention: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)

    Returns
    -------
    attention: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
    """
    attention = self.Softmax(attention)
    attention = self.dropout(attention)
    return attention
  
  def mixing_linear_layer(self, attention, value, original_shape): # attention.view(batches*n_windows, nHeads, N, N)
    """
    Compute (Q@K^T)@V and linear layer to allow mixing.

    attention: Tensor
        = (Q@K^T)/(sqrt(D)) + bias + masked_attention + Softmax activation.
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
    value: Tensor
        of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
    original_shape: torch.Size
        of x at input = (n_batch * n_windows, n_patch_in_window, hidden_dim)        

    Returns
    -------
    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)        
    """    
    # x shape after mat mul: (nB, nHeads, W0*W1*W2, C/nHeads)
    x = (attention @ value)
    # x shape after transpose: (B*n_windows, W0*W1*W2, nHeads, C/nHeads)
    x = torch.transpose(x, 1, 2)
    
    # original shape: (B*n_windows, W0*W1*W2, dim)
    # Reshape tensor to the original shape
    x = reshape(x, shape = original_shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x

class EarthAttention3DAbsolute(EarthAttentionBase):
  """
  3D Attention mechanism with absolute position bias term.

  Attention(Q, K, V) = SoftMax(Q@K^T/sqrt(D) + B @ V,
  where B=f(height, latitude)
  """

  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    """
    Initialize 3D Attention with absolute positional bias term. 

    dim: int
        Size of hidden dimension
    heads: int
        Number of attention heads
    dropout_rate: float
        probability that value is dropped during training
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    input_shape: torch.Shape
        of shape 3 describing the number of patches in the (vert, lat, lon) dimensions 
    """
    super().__init__(dim, heads, dropout_rate, window_size)
    
    # Record the number of different windows
    # negative signs = "upside-down floor division": used to mimic ceiling function
    self.type_of_windows = int(-(input_shape[0]//-window_size[0]) * -(input_shape[1]//-window_size[1]))
    
    # For each type of window, we will construct a set of parameters according to the paper
     # (size of each window, number of windows, heads)
    self.earth_specific_bias = zeros(size=((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads))

    # Making these tensors to be learnable parameters
    self.earth_specific_bias = Parameter(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    trunc_normal_(self.earth_specific_bias, std=.02) 

    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index()
    
    
  def _construct_index(self):
    """
    Construct the position index to reuse symmetrical parameters of the absolute position bias.
    
    Returns
    -------
    position_index: Tensor 
        of shape (n_patch_in_window)
    """
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
    coords_1 = stack(meshgrid([coords_zi, coords_hi, coords_w], indexing='ij'))
    coords_2 = stack(meshgrid([coords_zj, coords_hj, coords_w], indexing='ij'))
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
    """
    Forward pass of 3D attention mechanism with absolute positional bias term.

    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    mask: Tensor
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
    
    Returns
    -------
    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    """
    attention, value, original_shape = self.calculate_attention(x)

    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    earth_specific_bias = self.earth_specific_bias[self.position_index] 
    
    # Reshape the learnable bias to the same shape as the attention matrix
    earth_specific_bias = reshape(earth_specific_bias, shape=(self.total_window_size, self.total_window_size, self.type_of_windows, self.head_number))
    # ESB shape after permute: n_windows, n_heads, window size, window size
    earth_specific_bias = permute(earth_specific_bias, (2, 3, 0, 1))
    
    # Add the Earth-Specific bias to the attention matrix
    # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
    # attention.view(batches, n_windows, nHeads, N, N) + (1, n_windows, nHeads, N, N)
    attention = attention.view(-1, earth_specific_bias.shape[0], self.head_number, self.total_window_size, self.total_window_size) + earth_specific_bias.unsqueeze(0)
    # attention.view(batches*n_windows, nHeads, N, N)
    attention = attention.view(-1, self.head_number, self.total_window_size, self.total_window_size)

    attention = self.mask_attention(x, mask, attention, original_shape)
    attention = self.activate(attention)
    
    # Calculated the tensor after spatial mixing.
    x = self.mixing_linear_layer(attention, value, original_shape)

    return x

class EarthAttention3DRelative(EarthAttentionBase):
    """
    3D Attention mechanism with relative position bias term.

    Attention(Q, K, V) = SoftMax(Q@K^T/sqrt(D) + B @ V
    where B is not a function of (height, latitude)
    """

    def __init__(self, dim, heads, dropout_rate, window_size, input_shape, device):
        super().__init__( dim, heads, dropout_rate, window_size)

        #self.earth_specific_bias = None
        
        self.type_of_windows = 1
        # For each type of window, we will construct a set of parameters according to the paper
        # Making these tensors to be learnable parameters
        self.earth_specific_bias = Parameter(zeros(size=((2 * window_size[2] - 1) * (2 * window_size[1] - 1) * (2  * window_size[0] -1), self.type_of_windows, heads), device=device).to(torch.float32))

        # Initialize the tensors using Truncated normal distribution
        trunc_normal_(self.earth_specific_bias, std=.02) 

        # Construct position index to reuse self.earth_specific_bias
        self.position_index = self._construct_index()

    def _construct_index(self):
        """
        Construct the position index to reuse symmetrical parameters of the relative position bias.

        Returns
        -------
        position_index: Tensor 
            of shape (n_patch_in_window)
        """
        # Index in the pressure level of query matrix
        coords_z = arange(self.window_size[0])
        coords_h = arange(self.window_size[1])
        coords_w = arange(self.window_size[2])

        # Change the order of the index to calculate the index in total
        coords_1 = stack(meshgrid([coords_z, coords_h, coords_w], indexing='ij'))
        coords_flatten = flatten(coords_1, start_dim=1) 

        coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        coords = permute(coords, (1, 2, 0))

        # Shift the index for each dimension to start from 0
        coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        coords[:, :, 1] += self.window_size[1] - 1
        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        # Sum up the indexes in three dimensions
        self.position_index = sum(coords, dim=-1)
                                                        
        # Flatten the position index to facilitate further indexing
        self.position_index = flatten(self.position_index)
        return self.position_index
    
    def forward(self, x, mask):
      """
      Forward pass of 3D attention mechanism with relative positional bias term.

      x: Tensor
          of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
      mask: Tensor
          of shape (n_windows, n_patch_in_window, n_patch_in_window)
      
      Returns
      -------
      x: Tensor
          of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
      """
      attention, value, original_shape = self.calculate_attention(x)

      # self.earth_specific_bias is a set of neural network parameters to optimize. 
      earth_specific_bias = self.earth_specific_bias[self.position_index] 

      # Reshape the learnable bias to the same shape as the attention matrix
      earth_specific_bias = reshape(earth_specific_bias, shape=(self.total_window_size, self.total_window_size,  self.head_number))
      earth_specific_bias = permute(earth_specific_bias, (2, 0, 1))
              
      # Add the Earth-Specific bias to the attention matrix
      attention = attention + earth_specific_bias.unsqueeze(0)
      attention = self.mask_attention(x, mask, attention, original_shape)
      attention = self.activate(attention)
      
      # Calculated the tensor after spatial mixing.
      x = self.mixing_linear_layer(attention, value, original_shape)
      
      return x
  
class EarthAttention2D(EarthAttentionBase):
  """
  2D Attention mechanism with absolute position bias term.

  Attention(Q, K, V) = (Softmax(Q@K^T/sqrt(D) + B) @ V
  where B=f(latitude)
  """

  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    """
    Initialize 2D Attention with absolute positional bias term.

    dim: int
        Size of hidden dimension
    heads: int
        Number of attention heads
    dropout_rate: float
        probability that value is dropped during training
    window_size: Tensor
        Tensor of length(2) describing the window size in (lat, long) dimensions to be used in the attention mechanism.
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    """
    super().__init__(dim, heads, dropout_rate, window_size)

    # Record the number of different window types
    self.type_of_windows = (input_shape[0]//window_size[0]) 

    # Window size: (6, 12)
    # For each type of window, we will construct a set of parameters according to the paper
    self.earth_specific_bias = zeros(size=((2 * window_size[1] - 1) * window_size[0] * window_size[0] , self.type_of_windows, heads))

    # Making these tensors to be learnable parameters
    self.earth_specific_bias = Parameter(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    trunc_normal_(self.earth_specific_bias, std=.02) 

    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index()
    
  def _construct_index(self):
    """
    Construct the position index to reuse symmetrical parameters of the absolute position bias.
    
    Returns
    -------
    position_index: Tensor 
         of shape (n_patch_in_window)
    """
        # Index in the latitude of query matrix
    coords_hi = arange(start=0, end=self.window_size[0])
    # Index in the latitude of key matrix
    coords_hj = -arange(start=0, end=self.window_size[0])*self.window_size[0]

    # Index in the longitude of the key-value pair
    coords_w = arange(start=0, end=self.window_size[1])

    # Change the order of the index to calculate the index in total
    coords_1 = stack(meshgrid([coords_hi, coords_w], indexing='ij'))
    coords_2 = stack(meshgrid([coords_hj, coords_w], indexing='ij'))
    coords_flatten_1 = flatten(coords_1, start_dim=1) 
    coords_flatten_2 = flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = permute(coords, (1, 2, 0))

    # Shift the index for each dimension to start from 0
    coords[:, :, 1] += self.window_size[1] - 1
    coords[:, :, 0] *= (2 * self.window_size[1] - 1)

    # Sum up the indexes in three dimensions
    self.position_index = sum(coords, dim=-1)
                                                    
    # Flatten the position index to facilitate further indexing
    self.position_index = flatten(self.position_index)
    return self.position_index
    
  def forward(self, x, mask):
    """
    Forward pass of 2D attention mechanism with absolute positional bias term.

    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    mask: Tensor
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
    
    Returns
    -------
    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    """
    if x.get_device() < 0: 
      device = 'cpu'
    else:
      device = x.get_device()
    mask = mask.to(device)

    # x shape: (B*n_windows, W0, W1, W2, C)
    original_shape = x.shape 
    n_windows_total = original_shape[0]
    
    # x shape: (B*n_windows, W0*W1*W2, C)
    x = self.linear1(x)

    # reshape the data to calculate multi-head attention
    qkv = reshape(x, shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)) 
    query, key, value = permute(qkv, (2, 0, 3, 1, 4))

    # Scale the attention
    query = query * self.scale

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
    attention = (query @ key.transpose(-2, -1)) # @ denotes matrix multiplication
  
    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    earth_specific_bias = self.earth_specific_bias[self.position_index.repeat(attention.shape[0] // self.type_of_windows)] 
    # Reshape the learnable bias to the same shape as the attention matrix
    earth_specific_bias = reshape(earth_specific_bias, shape=(self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1, self.head_number))
    earth_specific_bias = permute(earth_specific_bias, (2, 3, 0, 1))

    # Add the Earth-Specific bias to the attention matrix
    attention = attention + earth_specific_bias 

    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    n_windows = mask.shape[0]
    n_patch_in_window = original_shape[1]

    attention = attention.view(n_windows_total // n_windows, n_windows, self.head_number, n_patch_in_window, n_patch_in_window)
    attention = attention + mask.unsqueeze(1).unsqueeze(0)
    attention = attention.view(-1, self.head_number, n_patch_in_window, n_patch_in_window)

    attention = self.activate(attention)

    # Calculated the tensor after spatial mixing.
    x = self.mixing_linear_layer(attention, value, original_shape)
    return x
  
class EarthAttention2DNoBias(EarthAttentionBase):
  """
  2D Attention mechanism with absolute position bias term.

  Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
  """

  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    """
    Initialize 2D Attention with no positional bias term.

    dim: int
        Size of hidden dimension
    heads: int
        Number of attention heads
    dropout_rate: float
        probability that value is dropped during training
    window_size: Tensor
        Tensor of length(2) describing the window size in (lat, long) dimensions to be used in the attention mechanism.
    input_shape: torch.Shape
        of shape 2 describing the number of patches in the (lat, lon) dimensions 
    """
    super().__init__(dim, heads, dropout_rate, window_size)

  def forward(self, x, mask):
    """
    Forward pass of 2D attention mechanism with absolute positional bias term.

    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    mask: Tensor
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
    
    Returns
    -------
    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    """
    # Linear layer to create query, key and value
    # Record the original shape of the input BEFORE linear layer (correct?)
    if x.get_device() < 0: 
      device = 'cpu'
    else:
      device = x.get_device()
    mask = mask.to(device)

    # x shape: (n_batch*n_windows, W0, W1, W2, C)
    original_shape = x.shape 
    n_windows_total = original_shape[0]
    
    # x shape: (B*n_windows, W0*W1*W2, C)
    x = self.linear1(x)

    # reshape the data to calculate multi-head attention
    qkv = reshape(x, shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)) 
    query, key, value = permute(qkv, (2, 0, 3, 1, 4))

    # Scale the attention
    query = query * self.scale

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
    attention = (query @ key.transpose(-2, -1)) # @ denotes matrix multiplication
  

    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    n_windows = mask.shape[0]
    n_patch_in_window = original_shape[1]

    attention = attention.view(n_windows_total // n_windows, n_windows, self.head_number, n_patch_in_window, n_patch_in_window)
    attention = attention + mask.unsqueeze(1).unsqueeze(0)
    attention = attention.view(-1, self.head_number, n_patch_in_window, n_patch_in_window)

    attention = self.activate(attention)

    # Calculated the tensor after spatial mixing.
    x = self.mixing_linear_layer(attention, value, original_shape)
    return x
  
class EarthAttentionNoBias(EarthAttentionBase):
  """
  3D Attention mechanism with no bias term.

  Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
  """

  def __init__(self, dim, heads, dropout_rate, window_size):
    """
    Initialize 3D Attention with no positional bias term.

    dim: int
        Size of hidden dimension
    heads: int
        Number of attention heads
    dropout_rate: float
        probability that value is dropped during training
    window_size: Tensor
        Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
    input_shape: torch.Shape
        of shape 3 describing the number of patches in the (vert, lat, lon) dimensions 
    """
    super().__init__(dim, heads, dropout_rate, window_size)
    self.type_of_windows = 0
    
    
  def forward(self, x, mask):
    """
    Forward pass of 2D attention mechanism with absolute positional bias term.

    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    mask: Tensor
        of shape (n_windows, n_patch_in_window, n_patch_in_window)
    
    Returns
    -------
    x: Tensor
        of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
    """
    attention, value, original_shape = self.calculate_attention(x)
    
    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    # from SWIN paper
    attention = self.mask_attention(x, mask, attention, original_shape)

    attention = self.activate(attention)
    # Calculated the tensor after spatial mixing.
    # Linear layer to post-process operated tensor
    # attention shape: (B*n_windows, nHead, W0*W1*W2, W0*W1*W2)
    x = self.mixing_linear_layer(attention, value, original_shape)
    
    return x