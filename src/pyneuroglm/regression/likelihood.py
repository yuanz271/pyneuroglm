"""
Likelihood functions for GLMs.

This module implements distribution-specific log-likelihoods and their
first and second derivatives for optimization-based fitting.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy


def poisson(w, X, y, inverse_link, subset_inds=None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the log-likelihood, gradient, and Hessian for a Poisson GLM.

    Parameters
    ----------
    w : array-like of shape (p,)
        Regression weights.
    X : array-like of shape (n, p)
        Design matrix.
    y : array-like of shape (n,)
        Observed counts.
    inverse_link : callable
        Inverse link function that returns (f, df, ddf) for input X @ w.
    subset_inds : array-like or None, optional
        Indices to subset the data. If None, use all data.

    Returns
    -------
    L : float
        Log-likelihood.
    dL : numpy.ndarray
        Gradient of the log-likelihood with respect to `w`.
    H : numpy.ndarray
        Hessian of the log-likelihood with respect to `w`.

    Notes
    -----
    The function handles zero counts and applies the nonlinearity to the linear predictor.
    """
    if subset_inds is not None:
        X = X[subset_inds]
        y = y[subset_inds]

    eta = X @ w
    dL = 0
    ddL = 0

    lam, dlam, ddlam = inverse_link(eta)  # rate (lambda) and derivatives

    nz = lam > 0

    L = np.sum(xlogy(y, lam)) - np.sum(lam)  # Compute x*log(y) so that the result is 0 if x = 0.

    y = y[nz]
    lam = lam[nz]
    X = X[nz]
    dlam = dlam[nz]
    ddlam = ddlam[nz]

    # dL = sum_i (y[i]/lam[i] - 1) d(lam[i])/d(eta[i]) x[i]
    ylam = y / lam  # (n,)
    d = (ylam - 1) * dlam
    dL = X.T @ d  # (p, n) (n,) -> (p,)  gradient

    # ddL = X'HX
    h = d * ddlam - (ylam + 1) / lam * np.square(dlam)  # (n,) (n,) + (n,) (n,) -> (n,)  diagonal
    ddL = np.einsum("ij,i,ik->jk", X, h, X)  # (p ,n) (n,n) (n, p) -> (p, p)  # Hessian

    return L, dL, ddL
