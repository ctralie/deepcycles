
import numpy as np

def get_sliding_window(X, Win):
    """
    Perform a multivariate sliding window embedding with no
    interpolation

    Parameters
    ----------
    X: ndarray(N, d)
        A multivariate time series
    Win: int 
        Window length to use

    Returns
    -------
    Y: ndarray(N-win+1, d*Win)
        A sliding window embedding
    """
    d = X.shape[1]
    M = X.shape[0]-Win+1
    Y = np.zeros((M, d*Win))
    for k in range(Win):
        Y[:, k*d:(k+1)*d] = X[k:k+M, :]
    return Y


def normalize_sliding_window(X, d, Win):
    """
    Do point centering and sphere normalizing to each window
    to control for linear drift and global amplitude.  Do point
    centering within each dimension but amplitude normalization
    across all dimensions

    Parameters
    ----------
    X: ndarray(M, d*Win)
        A sliding window embedding
    d: int
        Dimension of original time series
    Win: int
        Length of sliding window embedding
    
    Returns
    -------
    XRet: ndarray(M, d*Win)
        An array in which the mean of each row is zero
        and the norm of each row is 1
    """
    M = X.shape[0]
    X = np.reshape(X, (M, Win, d))
    for j in range(d):
        X[:, :, j] -= np.mean(X[:, :, j], 1)[:, None]
    X = np.reshape(X, (M, Win*d))
    XRet = X - np.mean(X, 1)[:, None]
    Norm = np.sqrt(np.sum(XRet**2, 1))
    Norm[Norm == 0] = 1
    XRet /= Norm[:, None]
    return XRet

def get_sliding_window_L2_inverse(X, d, Win):
    """
    Given a sliding window X with no interpolation, 
    devise a time series x whose non interpolated
    sliding window is as close as possible to X
    under the L2 norm.
    Note that if X is actually a sliding window embedding,
    it should be an exact inverse

    Parameters
    ----------
    X: ndarray(M, d*Win)
        A sliding window embedding
    d: int
        Dimension of original time series
    Win: int
        Length of sliding window embedding
    
    Returns
    -------
    x: ndarray(M+d-1)
        The resulting time series
    """
    M = X.shape[0]
    X = np.reshape(X, (M, Win, d))
    N = M+Win-1
    ret = np.zeros((N, d))
    for j in range(d):
        Y = np.fliplr(X[:, :, j])
        for i in range(N):
            ret[i, j] = np.mean(np.diag(Y, Win-i-1))
    return ret


def get_random_walk_curve(res, N, dim):
    """
    Synthesize a random walk curve
    
    Parameters
    ----------
    res: int
        Resolution of the grid used to generate the curve
    N: int
        Number of points on the curve
    dim: int
        Ambient dimension of curve
    
    Returns
    -------
    ndarray(N, d): The synthesized curve
    """
    #Enumerate all neighbors in hypercube via base 3 counting between [-1, 0, 1]
    Neighbs = np.zeros((3**dim, dim))
    Neighbs[0, :] = -np.ones((1, dim))
    idx = 1
    for ii in range(1, 3**dim):
        NCopy = np.copy(Neighbs[idx-1, :])
        NCopy[0] += 1
        for kk in range(dim):
            if NCopy[kk] > 1:
                NCopy[kk] = -1
                NCopy[kk+1] += 1
        Neighbs[idx, :] = NCopy
        idx += 1
    #Exclude the neighbor that's in the same place
    Neighbs = Neighbs[np.sum(np.abs(Neighbs), 1) > 0, :]

    #Pick a random starting point
    X = np.zeros((N, dim))
    X[0, :] = np.random.choice(res, dim)
    
    #Trace out a random path
    for ii in range(1, N):
        prev = np.copy(X[ii-1, :])
        N = np.tile(prev, (Neighbs.shape[0], 1)) + Neighbs
        #Pick a random next point that is in bounds
        idx = np.sum(N > 0, 1) + np.sum(N < res, 1)
        N = N[idx == 2*dim, :]
        X[ii, :] = N[np.random.choice(N.shape[0], 1), :]
    return X

def smooth_curve(X, fac):
    """
    Smooth out a curve

    Parameters
    ----------
    X: ndarray(N, d)
        The original curve
    fac: int
        Smoothing window
    
    Returns
    -------
    Y: ndarray(N, d)
        The smoothed curve
    """
    import scipy.interpolate as interp
    N = X.shape[0]
    dim = X.shape[1]
    idx = range(N)
    idxx = np.linspace(0, N, N*fac)
    Y = np.zeros((N*fac, dim))
    NOut = 0
    for ii in range(dim):
        Y[:, ii] = np.interp(idxx, idx, X[:, ii])
        #Smooth with box filter
        y = (0.5/fac)*np.convolve(Y[:, ii], np.ones(fac*2), mode='same')
        Y[0:len(y), ii] = y
        NOut = len(y)
    Y = Y[0:NOut-1, :]
    Y = Y[2*fac:-2*fac, :]
    return Y
