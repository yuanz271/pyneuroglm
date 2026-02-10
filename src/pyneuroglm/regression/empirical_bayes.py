"""Empirical Bayes utilities for GLM evidence approximation."""

import numpy as np


def log_evidence(param, hyperparam, loglik, llargs, logprior, lpargs):
    """
    Compute the log-evidence under a Generalized Linear Model (GLM) using the Laplace approximation.

    Parameters
    ----------
    param : numpy.ndarray of shape (m,)
        Regression weights (MAP estimate).
    hyperparam : Any
        Hyperparameters for the prior.
    loglik : callable
        Function that returns ``(L, dL, ddL)``, the log-likelihood and its first and second derivatives at the given parameters.
    llargs : tuple
        Extra arguments to pass to the log-likelihood function.
    logprior : callable
        Function that returns ``(P, dP, ddP)``, the log-prior and its first and second derivatives at the given parameters.
    lpargs : tuple
        Extra arguments to pass to the log-prior function.

    Returns
    -------
    float
        Log-evidence computed via Laplace's method at the MAP weights.

    Raises
    ------
    ValueError
        If the posterior Hessian is not positive definite.
    """
    L, _, ddL = loglik(param, *llargs)
    P, _, ddP = logprior(param, hyperparam, *lpargs)

    # Laplace: log Z ≈ L + P - 0.5*log|Sinv| where Sinv = -(ddL + ddP)
    Sinv = -(ddL + ddP)
    sign, logdetSinv = np.linalg.slogdet(Sinv)

    if sign <= 0:
        raise ValueError("Negative Hessian must be positive definite at MAP.")

    return L + P - 0.5 * logdetSinv
