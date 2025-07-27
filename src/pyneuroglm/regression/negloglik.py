import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy


def poisson(w, X, y, nlfun, subset_inds=None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
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


def posterior(
    wts: ArrayLike,
    mstruct: dict,
    indices=None
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute negative log-posterior of data under GLM regression model,
    plus gradient and Hessian.

    Parameters
    ----------
    wts : array-like, shape (m,)
        Regression weights.
    mstruct : dict
        Model structure with fields:
            - 'neglogli': function for negative log-likelihood
            - 'neglogpr': function for negative log-prior
            - 'liargs': list of arguments for neglogli
            - 'priargs': list of arguments for neglogpr
    indices : array-like or None, optional
        Indices to use for likelihood (default: all).

    Returns
    -------
    P : float
        Negative log-posterior.
    dP : np.ndarray
        Gradient.
    ddP : np.ndarray
        Hessian.
    """
    if indices is None:
        indices = np.arange(len(mstruct['liargs'][1]))

    L, dL, ddL = mstruct['neglogli'](wts, *mstruct['liargs'], indices)
    p, dp, ddp = mstruct['neglogpri'](wts, *mstruct['priargs'])
    P = L + p
    dP = dL + dp
    ddP = ddL + ddp
    
    return P, dP, ddP
