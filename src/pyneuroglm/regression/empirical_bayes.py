import numpy as np


def log_evidence(param, hyperparam, mstruct):
    """
    Compute the log-evidence under a GLM using the Laplace approximation.

    :param wts: Regression weights (MAP estimate).
    :type wts: numpy.ndarray, shape (m,)
    :param hprs: Hyperparameters for the prior.
    :type hprs: numpy.ndarray, shape (p,)
    :param mstruct: Model structure object with the following attributes:
        - neglogli (callable): Returns (L, grad, Hessian) = negative log-likelihood and its derivatives at wts.
        - logprior (callable): Returns (p, dp, negCinv, logdetCinv) = log-prior, gradient, negative inverse covariance, and log-det of C⁻¹.
        - liargs (list or tuple): Extra args for neglogli.
        - priargs (list or tuple): Extra args for logprior.
    :type mstruct: object

    :returns: logEv (float): Log-evidence via Laplace's method at the MAP weights.
    :rtype: float

    :raises ValueError: If the posterior Hessian is not positive definite.
    """
    # Evaluate negative log-likelihood and its Hessian
    L, _, ddL = mstruct.neglogli(param, *mstruct.liargs)

    # Evaluate log-prior, its negative inverse covariance, and its log-det
    p, _, negCinv, logdetCinv = mstruct.logprior(param, hyperparam, *mstruct.priargs)

    # Posterior Hessian = (negative log-likelihood Hessian) − (neg inverse prior)
    H_post = ddL - negCinv

    # Compute log-determinant of the posterior Hessian
    sign, logdet_H = np.linalg.slogdet(H_post)
    if sign <= 0:
        raise ValueError("Posterior Hessian must be positive definite.")

    # Laplace log-evidence
    logEv = -L + p + 0.5 * logdetCinv - 0.5 * logdet_H
    return logEv
