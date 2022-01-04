import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 0, 1)
    
class ZNormalize(nn.Module):
    """
    Layer for point centering and sphere normalizing
    """
    def __init__(self, eps=0.01):
        super(ZNormalize, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        # Point center
        Y = X - torch.reshape(torch.mean(X, axis=0), (1, -1))
        # Sphere normalize
        norm = torch.sqrt(torch.sum(Y*Y, axis=1))
        norm = torch.reshape(norm, (-1, 1))
        Y = Y/(norm+self.eps)
        return Y

class SlidingWindowLayer(nn.Module):
    """
    A good old fashion sliding window via fixed convolutional kernels
    """
    def __init__(self, win, dim, dT):
        """
        Parameters
        ----------
        win: int
            Window length
        dim: int
            Dimension of embedding (come up with
            an integer Tau that gets as close as possible
            to the window length to match this dimension)
        dT: int
            Length between windows
        """
        super(SlidingWindowLayer, self).__init__()
        self.win = win
        self.dim = dim
        self.dT = dT
        kernels = np.zeros((dim, 1, win))
        Tau = int(np.floor((win-1)/dim))
        for k in range(dim):
            kernels[k, 0, k*Tau] = 1
        self.kernels = torch.from_numpy(kernels).float()
    
    def forward(self, x):
        return torch.conv1d(x, self.kernels, stride=self.dT)
