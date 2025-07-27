from collections.abc import Callable
import warnings

import numpy as np
from scipy.optimize import minimize
from . import nonlinearity, negloglik, prior, optim


def poisson_neg_log_posterior(w, X, y, Cinv, nlfun, inds):
    L, dL, ddL = negloglik.poisson(w, X, y, nlfun, inds)
    p, dp, ddp = prior.gaussian_zero_mean_inv(w, Cinv)
    return L + p, dL + dp, ddL + ddp


def bernoulli_neg_log_posterior(w, X, y, Cinv, inds):
    raise NotImplementedError


def get_posterior_function(dist) -> Callable:
    match dist:
        case "poisson":
            nlfun = nonlinearity.exp
            return lambda w, X, y, Cinv, inds: poisson_neg_log_posterior(
                w, X, y, Cinv, nlfun, inds
            )
        case _:
            raise NotImplementedError(f"{dist=}")
        # case "bernoulli":
            # return lambda w, X, y, Cinv, inds: bernoulli_neg_log_posterior(
            #     w, X, y, Cinv, inds
            # )


def get_likelihood_function(dist, **kwargs):
    match dist:
        case "poisson":
            nlfun = nonlinearity.exp
            return lambda w, X, y, Cinv, inds: negloglik.poisson(w, X, y, nlfun, inds)


def initialize_lstsq(X, y, Cinv, cvfolds=None):
    """Initialize weights using least squares"""
    if cvfolds is None:
        w0 = np.linalg.lstsq(X.T @ X + Cinv, X.T @ y)[0]
    else:
        raise NotImplementedError
    return w0


def get_posterior_weights(X, y, Cinv, dist="poisson", cvfolds=None):
    if cvfolds is None:
        w0 = initialize_lstsq(X, y, Cinv)
        obj = get_posterior_function(dist)
        obj = optim.Objective(obj)  # pyright: ignore[reportArgumentType]
        opt = minimize(
            obj.function,
            w0,
            (X, y, Cinv, np.arange(len(y))),
            method="trust-ncg",
            jac=obj.gradient,
            hess=obj.hessian,
        )
        if not opt.success:
            warnings.warn("Optimization not succeed")
        w = opt.x
        H = opt.hess
        invH = np.linalg.inv(H)
        sd = np.sqrt(np.diag(invH))
    else:
        raise NotImplementedError

    return w, sd, H