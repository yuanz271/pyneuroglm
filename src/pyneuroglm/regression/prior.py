import numpy as np


def none_Cinv(rho, nx):
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


def ridge_Cinv(rho, nx, add_constant=False):
    """
    Return a ridge (L2) prior inverse covariance matrix.

    Parameters
    ----------
    rho : float
        Regularization parameter.
    nx : int
        Number of weights.
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
    Evaluate the log Gaussian prior with mean zero and inverse covariance.

    Parameters
    ----------
    w : array-like of shape (n,) or (n+1,)
        Parameter vector (last element can be DC/intercept).
    Cinv : array-like of shape (m, m)
        Gaussian inverse covariance matrix.

    Returns
    -------
    p : float
        log-prior.
    dp : numpy.ndarray
        Gradient of the log-prior.
    ddp : numpy.ndarray
        Hessian (inverse covariance matrix).

    Notes
    -----
    If `w` has more or fewer elements than `Cinv` (i.e., includes an intercept), the first element is ignored.
    """
    # check intercept
    assert len(w) == Cinv.shape[0], (
        "Unmatched size of weights and covariance. Exclude the intercept if it is in the weights vector."
    )
    # log p(w) = -0.5*log|S| - 0.5*w' Sinv w
    ddP = -Cinv
    dP = ddP @ w
    P = 0.5 * np.inner(w, dP)
    return P, dP, ddP
