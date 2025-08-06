import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy


def poisson_negloglik(w, X, y, nlfun, subset_inds=None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the negative log-likelihood, gradient, and Hessian for a Poisson GLM.

    Parameters
    ----------
    w : array-like of shape (p,)
        Regression weights.
    X : array-like of shape (n, p)
        Design matrix.
    y : array-like of shape (n,)
        Observed counts.
    nlfun : callable
        Nonlinearity function that returns (f, df, ddf) for input X @ w.
    subset_inds : array-like or None, optional
        Indices to subset the data. If None, use all data.

    Returns
    -------
    L : float
        Negative log-likelihood.
    dL : numpy.ndarray
        Gradient of the negative log-likelihood with respect to `w`.
    H : numpy.ndarray
        Hessian of the negative log-likelihood with respect to `w`.

    Notes
    -----
    The function handles zero counts and applies the nonlinearity to the linear predictor.
    """
    if subset_inds is not None:
        X = X[subset_inds]
        y = y[subset_inds]

    Xproj = X @ w
    dL = 0
    H = 0
    
    f, df, ddf = nlfun(Xproj)

    nz = f > 0
    
    L = - np.sum(xlogy(y, np.log(f))) + np.sum(f)  # 0 * log(0) = 0

    y = y[nz]
    f = f[nz]
    X = X[nz]
    df = df[nz]
    ddf = ddf[nz]

    yf = y / f  # (n,)
    dL = X.T @ ((1 - yf) * df)  # (p, n) (n,) -> (p,)
    d = ddf * (1 - yf) + y * (df / f) ** 2  # (n,) (n,) + (n,) (n,) -> (n,)
    H = X.T @ (d[:, None] * X)  # (p ,n) (n, p) -> (p, p)

    return L, dL, H
