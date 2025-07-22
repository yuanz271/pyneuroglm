from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix


@dataclass
class Basis:
    """
    Data structure representing a basis for modeling functions.

    :param name: Name of the basis (e.g., 'raised cosine', 'boxcar').
    :type name: str
    :param func: Constructor function used to create the basis.
    :type func: Callable
    :param kwargs: Dictionary of arguments used to construct the basis.
    :type kwargs: dict
    :param B: Basis matrix of shape (n_bins, n_bases).
    :type B: numpy.ndarray
    :param edim: Effective dimension (number of basis vectors).
    :type edim: int
    :param tr: Time lattice or bin centers (1D array).
    :type tr: numpy.ndarray
    :param centers: Centers of each basis function (1D array).
    :type centers: numpy.ndarray
    """

    name: str
    func: Callable
    kwargs: dict
    B: NDArray
    edim: int
    tr: NDArray
    centers: NDArray


def make_smooth_temporal_basis(
    shape: str,
    duration: float,
    n_bases: int,
    binfun: Callable[[float, bool], int],
) -> Basis:
    """
    Construct a smooth temporal basis matrix using either raised cosine or boxcar functions.

    :param shape: Type of basis function to use. Must be either 'raised cosine' or 'boxcar'.
    :type shape: str
    :param duration: Total time to be covered by the basis (in the same units expected by binfun).
    :type duration: float
    :param n_bases: Number of basis functions to generate.
    :type n_bases: int
    :param binfun: Function that converts duration to number of bins. Should accept (duration, True) and return an integer.
    :type binfun: Callable
    :returns: A Basis object containing the basis matrix (B), centers, and related metadata.
    :rtype: Basis
    """

    def rcos(x, period):
        return np.where(
            np.abs(x / period) < 0.5, np.cos(x * 2 * np.pi / period) * 0.5 + 0.5, 0
        )

    n_bins = binfun(duration, True)  # total number of bins

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
        centers=bcenters,
    )

    return basis


def temporal_bases(
    x: ArrayLike, B: ArrayLike, mask: NDArray | None = None, addDC: bool = False
) -> NDArray:
    """
    Apply a temporal basis to input data, optionally masking basis functions and adding a DC component.

    :param x: Input data array of shape (T, dx), where T is the number of time bins and dx is the number of covariates.
    :type x: numpy.ndarray
    :param B: Basis matrix of shape (TB, M), where TB is the number of basis time bins and M is the number of basis functions.
    :type B: numpy.ndarray
    :param mask: Optional boolean mask of shape (dx, M) indicating which basis functions to use for each covariate. If None, all are used.
    :type mask: numpy.ndarray or None
    :param addDC: If True, adds a DC (constant) column to the output.
    :type addDC: bool
    :returns: Output array of shape (T, sum(mask) + int(addDC)), where sum(mask) is the total number of active basis functions.
    :rtype: numpy.ndarray
    """
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
        BX[:, col : sI[k]] = A[:T, :]
        col = sI[k]

    if addDC:
        BX[:, -1] = 1
    return BX


def conv_basis(x: ArrayLike, basis: Basis, offset: int = 0) -> NDArray:
    """
    Convolve input data with a temporal basis, with optional causal or anti-causal offset.

    :param x: Input data array of shape (T, dx), where T is the number of time bins and dx is the number of covariates.
    :type x: array-like
    :param basis: Basis object containing the basis matrix to convolve with.
    :type basis: Basis
    :param offset: Number of bins to offset the convolution. Positive for causal, negative for anti-causal, zero for centered.
    :type offset: int
    :returns: Output array of shape (T, n_bases), where n_bases is the number of basis functions in the basis.
    :rtype: numpy.ndarray
    """
    x = np.asarray(x)

    assert x.ndim == 2

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


def delta_stim(bt: ArrayLike, n_bins: int, v: NDArray | None = None) -> NDArray:
    """
    Create a sparse stimulus vector with delta (impulse) events at specified time bins.

    :param bt: Array of time bin indices where events occur.
    :type bt: array-like
    :param n_bins: Total number of time bins.
    :type n_bins: int
    :param v: Optional array of values for each event (default is 1 for all).
    :type v: numpy.ndarray or None
    :returns: Stimulus array of shape (n_bins, 1) with impulses at specified bins.
    :rtype: numpy.ndarray
    """
    bt = np.asarray(bt)
    bidx = bt < n_bins
    bt = bt[bidx]
    o = np.zeros_like(bt, dtype=np.int_)

    v = np.ones_like(bt) if v is None else v[bidx]

    assert len(o) == len(v)

    stim = coo_matrix((v, (bt, o)), shape=(n_bins, 1))
    # print(stim)
    return stim.toarray()


def boxcar_stim(start_bin: int, end_bin: int, nbin: int, v: float = 1.0) -> NDArray:
    """
    Create a boxcar (rectangular) stimulus vector with constant value over a specified interval.

    :param start_bin: Start index of the boxcar (inclusive).
    :type start_bin: int
    :param end_bin: End index of the boxcar (exclusive).
    :type end_bin: int
    :param nbin: Total number of time bins.
    :type nbin: int
    :param v: Value to assign within the boxcar interval (default 1.0).
    :type v: float
    :returns: Stimulus array of shape (nbin, 1) with value v in [start_bin:end_bin], zeros elsewhere.
    :rtype: numpy.ndarray
    """
    x = np.zeros((nbin, 1))
    x[start_bin:end_bin, :] = v
    return x


def raised_cos(x: ArrayLike, c: ArrayLike, dc: float) -> NDArray:
    x = np.asarray(x)
    c = np.asarray(c)
    d = 0.5 * (x - c) * np.pi / dc
    d = np.clip(d, -np.pi, np.pi)
    return (np.cos(d) + 1) * 0.5


def make_nonlinear_raised_cos(
    n_bases, binsize_in_ms, end_points_in_ms, nl_offset_in_ms
) -> Basis:
    """
    Create a nonlinearly stretched raised-cosine temporal basis.

    :param n_bases: Number of basis vectors to generate.
    :type n_bases: int
    :param binsize_in_ms: Time bin size in milliseconds.
    :type binsize_in_ms: float
    :param end_points_in_ms: Sequence of two floats, [first_peak, last_peak], specifying the centers of the first and last basis functions in milliseconds.
    :type end_points_in_ms: array-like
    :param nl_offset_in_ms: Offset for nonlinear stretching in milliseconds (must be > 0).
    :type nl_offset_in_ms: float
    :returns: A Basis object containing the nonlinearly stretched raised-cosine basis matrix and related metadata.
    :rtype: Basis
    """
    if nl_offset_in_ms <= 0:
        raise ValueError("nl_offset must be greater than 0")

    # Nonlinearity and its inverse
    def nlin(x):
        return np.log(x + 1e-20)

    def invnl(y):
        return np.exp(y) - 1e-20

    # Map end points through log-stretch
    end_points_in_ms = np.asarray(end_points_in_ms)
    y_range = nlin(end_points_in_ms + nl_offset_in_ms)

    # Spacing in the transformed domain
    db = (y_range[1] - y_range[0]) / (n_bases - 1)

    # Centers in transformed space
    ctrs = np.linspace(y_range[0], y_range[1], n_bases, endpoint=True)

    # Maximum time before mapping back
    mxt = invnl(y_range[1] + 2 * db) - nl_offset_in_ms

    # Time lattice in units of bins
    min_time_interval = 1.0
    iht = np.arange(0, mxt + min_time_interval, binsize_in_ms) / binsize_in_ms

    # Compute basis matrix: shape (len(tr), n_bases)
    phi = nlin(iht + nl_offset_in_ms)  # column vector 281, 1, ctrs is row vector 1, 10
    ihbasis = raised_cos(phi[:, None], ctrs[None, :], db)
    # Map centers back to original time axis
    ihctrs = invnl(ctrs)

    basis = Basis(
        name=make_nonlinear_raised_cos.__name__,
        func=make_nonlinear_raised_cos,
        kwargs=dict(
            n_bases=n_bases,
            binsize_in_ms=binsize_in_ms,
            end_points_in_ms=end_points_in_ms,
            nl_offset_in_ms=nl_offset_in_ms,
        ),
        B=ihbasis,
        edim=np.size(ihbasis, 1),
        tr=iht,
        centers=ihctrs,
    )
    return basis
