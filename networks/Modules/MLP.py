import torch.nn as nn
from torch.nn import GELU, Dropout, Linear


class MLP(nn.Module):
  """MLP."""

  def __init__(self, dim, dropout_rate):
    """
    Initialize MLP.
    
    dim: int
    dropout_rate: float
    """
    super().__init__()
    '''MLP layers, same as most vision transformer architectures.'''
    self.linear1 = Linear(dim, dim * 4)
    self.linear2 = Linear(dim * 4, dim)
    self.activation = GELU()
    self.drop = Dropout(p=dropout_rate)
    
  def forward(self, x):
    """Forward function of MLP."""
    x = self.linear1(x)
    x = self.activation(x)
    x = self.drop(x)
    x = self.linear2(x)
    x = self.drop(x)
    return x
