import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from gph import ripser_parallel

class RipsPersistenceDistance(nn.Module):
    """
    pytorch module that takes a distance matrix and outputs a list of persistence diagrams
    of the associated Rips filtration
    """
    def __init__(self, hom_dims):
        """
        hom_dims: a tuple of degrees in which to compute persistent homology
        """
        super(RipsPersistenceDistance, self).__init__()
        self.hom_dims = hom_dims

    def forward(self, input):
        # Compute persistence from distance matrix.
        idx = ripser_parallel(input.detach().numpy(),
                              metric="precomputed",
                              maxdim=max(self.hom_dims),
                              return_generators=True)["gens"]
        dgms = []
        for hom_dim in self.hom_dims:
            if hom_dim == 0:
                if idx[0].shape[0] == 0:
                    dgms.append(torch.zeros((0, 2), requires_grad=True))
                else:
                    verts = torch.from_numpy(idx[0]).type(torch.LongTensor)
                    dgm = torch.stack((input[verts[:, 0], verts[:, 0]], input[verts[:, 1], verts[:, 2]]), 1)
                    dgms.append(dgm)
            if hom_dim != 0:
                if len(idx[1]) == 0:
                    dgms.append(torch.zeros((0, 2), requires_grad=True))
                else:
                    verts = torch.from_numpy(idx[1][hom_dim - 1]).type(torch.LongTensor)
                    dgm = torch.stack((input[verts[:, 0], verts[:, 1]], input[verts[:, 2], verts[:, 3]]), 1)
                    dgms.append(dgm)
        return dgms


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


class SlidingVideoDistanceMatrixLayer(nn.Module):
    """
    Compute a distance matrix for the sliding window 
    embedding of a video with an integer step size, as per
    [1] "(Quasi) Periodicity Quantification in Video Data, Using Topology"
        by Chris Tralie and Jose Perea
    """
    def __init__(self, win, device):
        """
        Parameters
        ----------
        win: int
            Length of window
        """
        super(SlidingVideoDistanceMatrixLayer, self).__init__()
        self.win = win
        self.device = device
    
    def forward(self, x):
        win = self.win
        eps = 1e-12
        x = torch.reshape(x, (x.shape[0], np.prod(x.shape[1::])))
        n_pixels = x.shape[1]
        xsqr = x.pow(2).sum(1).view(-1, 1)
        dist = xsqr + xsqr.t().contiguous() - 2*torch.mm(x, x.t().contiguous())
        dist = torch.clamp(dist, eps, np.inf)
        dist = torch.sqrt(dist)
        N = dist.shape[0]
        dist = dist / torch.sqrt(torch.tensor([n_pixels*win]))
        dist_stack = torch.zeros((N-win+1, N-win+1), device=self.device)
        for i in range(0, win):
            dist_stack += dist[i:i+N-win+1, i:i+N-win+1]
        for i in range(N-win+1):
            dist_stack[i, i] = 0
        return dist_stack

