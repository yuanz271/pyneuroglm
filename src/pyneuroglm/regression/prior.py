import numpy as np


def none(rho, nx):
    """
    Return a zero (no prior) inverse covariance matrix.

    Parameters
    ----------
    rho : float
        Regularization parameter (ignored).
    nx : int
        Number of parameters.

    Returns
    -------
    numpy.ndarray
        Zero matrix of shape (nx, nx).
    """
    return np.zeros((nx, nx))


def ridge(rho, nx, add_constant=False):
    """
    Return a ridge (L2) prior inverse covariance matrix.

    Parameters
    ----------
    rho : float
        Regularization parameter.
    nx : int
        Number of parameters.
    add_constant : bool, optional
        If True, do not regularize the first parameter (intercept).

    Returns
    -------
    numpy.ndarray
        Diagonal matrix of shape (nx, nx) or (nx+1, nx+1) if add_constant is True.
    """
    if add_constant:
        d = np.ones(1 + nx)
        d[0] = 0
    else:
        d = np.ones(nx)
    return np.diag(d * rho)


def gaussian_zero_mean_inv(w, Cinv):
    """
    Evaluate the negative log Gaussian prior with mean zero and inverse covariance.

    Parameters
    ----------
    w : array-like of shape (n,) or (n+1,)
        Parameter vector (last element can be DC/intercept).
    Cinv : array-like of shape (m, m)
        Gaussian inverse covariance matrix.

    Returns
    -------
    p : float
        Negative log-prior.
    dp : numpy.ndarray
        Gradient of the negative log-prior.
    ddp : numpy.ndarray
        Hessian (inverse covariance matrix).

    Notes
    -----
    If `w` has one more element than `Cinv` (i.e., includes an intercept), the first element is ignored.
    """
    # check intercept
    if len(w) == Cinv.shape[0] + 1:
        w = w[1:]  # assume the 1st column is the intercept

    dp = Cinv @ w
    p = 0.5 * np.inner(w, dp)
    return p, dp, Cinv
