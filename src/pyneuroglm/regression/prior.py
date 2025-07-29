import numpy as np


def none(rho, nx):
    """No prior inverse covariance matrix"""
    return np.zeros((nx, nx))


def ridge(rho, nx, add_constant=False):
    """Ridge inverse covariance matrix"""
    if add_constant:
        d = np.ones(1 + nx)
        d[0] = 0
    else:
        d = np.ones(nx)
    return np.diag(d * rho)


def gaussian_zero_mean_inv(w, Cinv):
    """
    Evaluate negative log gaussian prior with mean zero and covariance Cinv
    [p, dp, ddp] = gaussian_zero_mean_inv(w, Cinv)

    Evaluate a Gaussian negative log-prior at parameter vector w.

    Inputs:
        w [n] - parameter vector (last element can be DC)
        Cinv [m x m] - gaussian inverse covariance

    Outputs:
        p [1 x 1] - log-prior
        dp [n x 1] - grad
        ddp [n x n] - Hessian
    """
    # check intercept
    if len(w) == Cinv.shape[0] + 1:
        w = w[1:]  # assume the 1st column is the intercept

    dp = Cinv @ w
    p = 0.5 * np.inner(w, dp)
    return p, dp, Cinv
