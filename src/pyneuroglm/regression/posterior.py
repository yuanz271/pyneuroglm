"""
Posterior objectives and fitting routines.

Provides helpers to compose log-likelihoods with Gaussian priors and
fit GLMs via Newton-CG, returning MAP weights and uncertainty.
"""

from collections.abc import Callable
from typing import Any
import warnings

import numpy as np
from scipy.optimize import minimize
from . import nonlinearity, likelihood, prior, optim


def poisson(w, X, y, Cinv, nlfun, inds):
    """
    Compute the log-posterior, gradient, and Hessian for a Poisson GLM with Gaussian prior.

    Parameters
    ----------
    w : array-like
        Regression weights.
    X : array-like
        Design matrix.
    y : array-like
        Observed counts.
    Cinv : array-like
        Inverse covariance matrix for the Gaussian prior.
    nlfun : callable
        Nonlinearity function.
    inds : array-like
        Array of indices to subset the data for computation.

    Returns
    -------
    L : float
        Log-posterior.
    dL : numpy.ndarray
        Gradient of the log-posterior.
    ddL : numpy.ndarray
        Hessian of the log-posterior.
    """
    L, dL, ddL = likelihood.poisson(w, X, y, nlfun, inds)
    P, dP, ddP = prior.gaussian_zero_mean_inv(w, Cinv)
    return L + P, dL + dP, ddL + ddP


def bernoulli(w, X, y, Cinv, inds):
    """
    Raise NotImplementedError for Bernoulli log-posterior.

    Raises
    ------
    NotImplementedError
        This function is not implemented.
    """
    raise NotImplementedError


def get_posterior_function(dist) -> Callable:
    """
    Get a posterior function for the specified distribution.

    Parameters
    ----------
    dist : str
        Distribution name ("poisson", ...).

    Returns
    -------
    callable
        Posterior function for the specified distribution.

    Raises
    ------
    NotImplementedError
        If the distribution is not supported.
    """
    match dist:
        case "poisson":
            nlfun = nonlinearity.exp
            return lambda w, X, y, Cinv, inds: poisson(w, X, y, Cinv, nlfun, inds)
        case _:
            raise NotImplementedError(f"{dist=}")
        # case "bernoulli":
        # return lambda w, X, y, Cinv, inds: bernoulli_neg_log_posterior(
        #     w, X, y, Cinv, inds
        # )


def get_likelihood_function(dist, **kwargs):
    """
    Get a likelihood function for the specified distribution.

    Parameters
    ----------
    dist : str
        Distribution name ("poisson", ...).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    callable
        Likelihood function for the specified distribution.

    Raises
    ------
    NotImplementedError
        If the distribution is not supported.
    """
    match dist:
        case "poisson":
            nlfun = nonlinearity.exp
            return lambda w, X, y, Cinv, inds: likelihood.poisson(w, X, y, nlfun, inds)
        case _:
            raise NotImplementedError(f"{dist=}")


def initialize_lstsq(X, y, Cinv, cvfolds=None):
    """
    Initialize weights using least squares.

    Parameters
    ----------
    X : array-like
        Design matrix.
    y : array-like
        Response vector.
    Cinv : array-like
        Inverse covariance matrix for the prior.
    cvfolds : any, optional
        Cross-validation folds (not implemented).

    Returns
    -------
    numpy.ndarray
        Initial weights.

    Raises
    ------
    NotImplementedError
        If cross-validation is requested.
    """
    if cvfolds is None:
        w0 = np.linalg.lstsq(X.T @ X + Cinv, X.T @ y)[0]
    else:
        raise NotImplementedError
    return w0


def initialize_zero(X, y, Cinv, cvfolds=None, bias=True, nlin=None):
    """
    Initialize weights using zeros or mean for bias.

    Parameters
    ----------
    X : array-like
        Design matrix.
    y : array-like
        Response vector.
    Cinv : array-like
        Inverse covariance matrix for the prior.
    cvfolds : any, optional
        Cross-validation folds (not implemented).
    bias : bool, optional
        If True, initialize the first weight as the mean of y.
    nlin : callable or None, optional
        Nonlinearity to apply to the bias.

    Returns
    -------
    numpy.ndarray
        Initial weights.
    """
    w = np.zeros(X.shape[1])
    if bias:
        w0 = np.mean(y)
        if nlin is not None:
            w0 = nlin(w0)
        w[0] = w0
    return w


def get_posterior_weights(
    X,
    y,
    Cinv,
    dist="poisson",
    cvfolds=None,
    initialize=initialize_zero,
    init_kwargs=None,
):
    """
    Fit a GLM by maximizing the posterior and return weights, standard deviations, and Hessian.

    Parameters
    ----------
    X : array-like
        Design matrix.
    y : array-like
        Response vector.
    Cinv : array-like
        Inverse covariance matrix for the prior.
    dist : str, optional
        Distribution name ("poisson", ...).
    cvfolds : any, optional
        Cross-validation folds (not implemented).
    initialize : callable, optional
        Initialization function.
    init_kwargs : dict or None, optional
        Additional keyword arguments for initialization.

    Returns
    -------
    w : numpy.ndarray
        MAP weights.
    sd : numpy.ndarray
        Standard deviations of the weights.
    invH : numpy.ndarray
        Inverse Hessian at the MAP weights (approximate covariance).

    Raises
    ------
    NotImplementedError
        If cross-validation is requested.
    """
    if cvfolds is None:
        if init_kwargs is None:
            init_kwargs = {}
        w0 = initialize(X, y, Cinv, cvfolds, **init_kwargs)
        args = (X, y, Cinv, np.arange(len(y)))
        obj = get_posterior_function(dist)
        obj = optim.Objective(obj, flip_sign=True)
        opt: Any = minimize(
            obj.function,
            w0,
            args,
            method="Newton-CG",
            jac=obj.gradient,
            hess=obj.hessian,
        )
        if not opt.success:
            warnings.warn("Optimization not succeed")
        # print(opt)
        w = opt.x
        H: Any = obj.hessian(w, *args)  # negative Hessian
        # Ensure H is a floating-point array for np.linalg.inv
        try:
            invH = np.linalg.inv(H)
            sd = np.sqrt(invH.diagonal())
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Hessian inversion failed: {e}")
            invH = np.full(H.shape, np.nan)
            sd = np.full(H.shape[0], np.nan)
    else:
        raise NotImplementedError

    return w, sd, invH
