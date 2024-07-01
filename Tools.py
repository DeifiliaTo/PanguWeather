import numpy as np
import torch


def TruncatedNormalInit(tensor, mean=0, std=0.02):
    """
    Initialize a tensor using a truncated normal distribution.
    """
    size = tensor.shape
    tmp = np.random.normal(mean, std, size).astype(np.float32)
    valid = (tmp < 2 * std) & (tmp > -2 * std)
    ind = np.where(~valid)[0]

    while len(ind):
        tmp[ind] = np.random.normal(mean, std, len(ind)).astype(np.float32)
        valid = (tmp < 2 * std) & (tmp > -2 * std)
        ind = np.where(~valid)[0]

    with torch.no_grad():
        tensor.copy_(torch.from_numpy(tmp))
    return tensor