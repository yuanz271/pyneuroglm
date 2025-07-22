from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix


@dataclass
class Basis:
    """
    Represents a basis for modeling functions, with constructor function, arguments,
    basis matrix B, and effective dimension edim.
    """

    name: str
    func: Callable
    kwargs: dict
    B: np.ndarray
    edim: int
    tr: np.ndarray
    centers: np.ndarray


def make_smooth_temporal_basis(shape, duration, n_bases, binfun):
    """
    :param shape: 'raised cosine' or 'boxcar'
    :param duration: time to be covered
    :param nbasis: number of basis
    :param binfun:
    :return:
    """

    def rcos(x, period):
        return np.where(np.abs(x / period) < 0.5, np.cos(x * 2 * np.pi / period) * 0.5 + 0.5, 0)

    n_bins = binfun(duration)  # total number of bins
    
    tt = np.arange(1, n_bins + 1, dtype=float)
    ttb = np.tile(tt[:, None], (1, n_bases))

    if shape == "raised cosine":
        dbcenter = n_bins / (3.0 + n_bases)
        width = 4 * dbcenter
        bcenters = 2 * dbcenter + dbcenter * np.arange(n_bases)
        BBstm = rcos(ttb - bcenters[None, :], width)
    elif shape == "boxcar":
        width = n_bins / n_bases
        BBstm = np.zeros_like(ttb)
        bcenters = width * np.arange(1, n_bases + 1) - width / 2
        for k in range(n_bases):
            mask = np.logical_and(
                ttb[:, k] > ceil(width * k), ttb[:, k] <= ceil(width * (k + 1))
            )
            BBstm[mask, k] = 1.0 / sum(mask)
    else:
        raise ValueError(f"Unknown basis shape: {shape}")

    basis = Basis(
        name=shape,
        func=make_smooth_temporal_basis,
        kwargs=dict(shape=shape, duration=duration, n_bases=n_bases, binfun=binfun),
        B=BBstm,
        edim=BBstm.shape[1],
        tr=tt - 1,
        centers=bcenters
    )

    return basis


def temporal_bases(x, B, mask=None, addDC=False):
    x = np.asarray(x)
    B = np.asarray(B)

    T, dx = x.shape
    TB, M = B.shape

    if mask is None:
        mask = np.ones((dx, M), dtype=bool)
    
    sI = np.sum(mask, 1)  # bases oer covariate
    BX = np.zeros((T, np.sum(sI) + addDC))
    sI = np.cumsum(sI)
    col = 0
    for k in range(dx):
        A = convolve2d(x[:, [k]], B[:, mask[k, :]])
        BX[:, col:sI[k]] = A[:T, :]
        col = sI[k]

    if addDC:
        BX[:, -1] = 1
    return BX


def conv_basis(x, basis, offset=0):
    """
    :param x: [T, dx]
    :param basis: basis
    :param offset: scalar
    :return:
    """
    x = np.asarray(x)
    n, ndim = x.shape
    if offset < 0:  # anti-causal
        x = np.concatenate((x, np.zeros((-offset, ndim))), axis=0)
    elif offset > 0:  # causal
        x = np.concatenate((np.zeros((offset, ndim)), x), axis=0)
    else:
        pass
    X = temporal_bases(x, basis.B)

    if offset < 0:
        X = X[-offset:, :]
    elif offset > 0:
        X = X[:-offset, :]

    return X


def delta_stim(bt, n_bins, v:np.ndarray|None=None):
    bidx = bt < n_bins
    bt = bt[bidx]
    o = np.zeros_like(bt, dtype=np.int_)

    v = np.ones_like(bt)  if v is None else v[bidx]

    assert len(o) == len(v)

    stim = coo_matrix((v, (bt, o)), shape=(n_bins, 1))
    
    return stim.toarray()


def boxcar_stim(start_bin, end_bin, nbin, v=1.0):
    x = np.zeros((nbin, 1))
    x[start_bin:end_bin, :] = v
    return x


# Raised-cosine function
def raised_cos(x, c, dc):
    arg = 0.5 * (x - c) * np.pi / dc
    arg = np.clip(arg, -np.pi, np.pi)
    return (np.cos(arg) + 1) * 0.5


def make_nonlinear_raised_cos(n_bases, binsize, end_points, nl_offset):
    """
    Create a nonlinearly stretched raised-cosine temporal basis.

    Parameters
    ----------
    n_bases : int
        Number of basis vectors.
    binsize : float
        Time bin size.
    end_points : array-like, shape (2,)
        [first_peak, last_peak] for the centers of the cosines.
    nl_offset : float
        Offset for nonlinear stretching in the same unit of binsize (must be > 0).

    Returns
    -------
    basis : Basis
        Basis object with fields:
        name : str
            Function name.
        func : callable
            Constructor function.
        args : tuple
            Input parameters (n_bases, binsize, end_points, nl_offset).
        B : ndarray
            Basis matrix (time x bases).
        edim : int
            Number of basis vectors.
        tr : ndarray
            Time lattice (in bins).
        centers : ndarray
            Centers of each basis function (in ms).
    """
    if nl_offset <= 0:
        raise ValueError("nl_offset must be greater than 0")

    # Nonlinearity and its inverse
    def nlin(x):
        return np.log(x + 1e-20)

    def invnl(y):
        return np.exp(y) - 1e-20

    # Map end points through log-stretch
    end_points = np.array(end_points)
    y_range = nlin(end_points + nl_offset)

    # Spacing in the transformed domain
    db = (y_range[1] - y_range[0]) / (n_bases - 1)

    # Centers in transformed space
    # ctrs = np.arange(y_range[0], y_range[1] + db, db)
    ctrs = np.linspace(y_range[0], y_range[1], n_bases, endpoint=True)

    # Maximum time (in ms) before mapping back
    mxt = invnl(y_range[1] + 2 * db) - nl_offset

    # Time lattice in units of bins
    iht = np.arange(0, mxt + np.finfo(mxt.dtype).eps, binsize) / binsize

    # Compute basis matrix: shape (len(tr), n_bases)
    phi = nlin(iht + nl_offset)
    ihbasis = raised_cos(phi[:, None], ctrs[None, :], db)

    # Map centers back to original time axis
    ihctrs = invnl(ctrs)

    basis = Basis(
        name=make_nonlinear_raised_cos.__name__,
        func=make_nonlinear_raised_cos,
        kwargs=dict(n_bases=n_bases, binsize=binsize, end_points=end_points, nl_offset=nl_offset),
        B=ihbasis,
        edim=np.size(ihbasis, 1),
        tr=iht,
        centers=ihctrs,
    )
    return basis
