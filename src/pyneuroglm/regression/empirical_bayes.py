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
    # Evaluate log-likelihood and its Hessian
    L, _, ddL = loglik(param, *llargs)

    # Evaluate log-prior and its Hessian
    P, _, ddP = logprior(param, hyperparam, *lpargs)

    # logjoint = loglik + logprior
    # logdet = - log|Hessian logjoint|
    # logZ = logjoint + 0.5 logdet

    # logZ = logp(y|x, b) + logp(b) + 0.5 * logdetS
    # Sinv = - (dd logp(y|x, b) + dd logp(b))
    # logZ = L + P + 0.5 * log|S|

    Sinv = ddL + ddP  # -Hession joint
    sign, logdetSinv = np.linalg.slogdet(Sinv)

    if sign <= 0:
        raise ValueError("Posterior Hessian must be positive definite.")

    # Laplace log-evidence
    logZ = L + P - 0.5 * logdetSinv  # log|S| = -log|invS|
    return logZ


# def hessian_log_posterior_mvnorm(beta, X, y, Sigma_prior_inv):
#     eta = X @ beta
#     lambda_ = np.exp(eta)
#     W = np.diag(lambda_)
#     H = X.T @ W @ X + Sigma_prior_inv
#     return H


# ddL = -X.T @ diag(lambda) @ X
# ddp = -Cinv
