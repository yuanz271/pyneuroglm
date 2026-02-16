"""
Temporal basis functions and stimulus utilities for GLM design.

Provides helpers to construct smooth temporal bases (raised cosine, boxcar),
apply them to covariates, and generate simple stimulus arrays used throughout
the design matrix construction.
"""

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

    Parameters
    ----------
    name : str
        Name of the basis (e.g., 'raised cosine', 'boxcar').
    func : Callable
        Constructor function used to create the basis.
    kwargs : dict
        Dictionary of arguments used to construct the basis.
    B : numpy.ndarray
        Basis matrix of shape (n_bins, n_bases).
    edim : int
        Effective dimension (number of basis vectors).
    tr : numpy.ndarray
        Time lattice or bin centers (1D array).
    centers : numpy.ndarray
        Centers of each basis function (1D array).
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

    Parameters
    ----------
    shape : str
        Type of basis function to use. Must be either 'raised cosine' or 'boxcar'.
    duration : float
        Total time to be covered by the basis (in the same units expected by binfun).
    n_bases : int
        Number of basis functions to generate.
    binfun : Callable[[float, bool], int]
        Function that converts duration to number of bins. Should accept (duration, True) and return an integer.

    Returns
    -------
    Basis
        A Basis object containing the basis matrix (B), centers, and related metadata.

    Raises
    ------
    ValueError
        If the shape is not recognized.
    """

    def rcos(x, period):
        """
        Evaluate a single raised-cosine bump at locations `x`.

        Parameters
        ----------
        x : ndarray
            Sample locations where the basis is evaluated.
        period : float
            Width (period) of the raised cosine.

        Returns
        -------
        ndarray
            Raised-cosine values with the same shape as `x`.
        """
        return np.where(np.abs(x / period) < 0.5, np.cos(x * 2 * np.pi / period) * 0.5 + 0.5, 0)

    n_bins = binfun(duration, True)  # total number of bins

    tt = np.arange(1, n_bins + 1, dtype=float)  # matlab index
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
            mask = np.logical_and(ttb[:, k] > ceil(width * k), ttb[:, k] <= ceil(width * (k + 1)))
            BBstm[mask, k] = 1.0 / np.count_nonzero(mask)
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

    Parameters
    ----------
    x : array-like of shape (T, dx)
        Input data array, where T is the number of time bins and dx is the number of covariates.
    B : array-like of shape (TB, M)
        Basis matrix, where TB is the number of basis time bins and M is the number of basis functions.
    mask : numpy.ndarray of bool, shape (dx, M), optional
        Boolean mask indicating which basis functions to use for each covariate. If None, all are used.
    addDC : bool, default=False
        If True, adds a DC (constant) column to the output.

    Returns
    -------
    numpy.ndarray
        Output array of shape (T, sum(mask) + int(addDC)), where sum(mask) is the total number of active basis functions.
    """
    x = np.asarray(x)
    B = np.asarray(B)

    T, dx = x.shape
    TB, M = B.shape

    if mask is None:
        mask = np.ones((dx, M), dtype=bool)

    sI = np.sum(mask, 1)  # bases per covariate
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

    Parameters
    ----------
    x : array-like of shape (T, dx)
        Input data array, where T is the number of time bins and dx is the number of covariates.
    basis : Basis
        Basis object containing the basis matrix to convolve with.
    offset : int, default=0
        Number of bins to offset the convolution. Positive for causal, negative for anti-causal, zero for centered.

    Returns
    -------
    numpy.ndarray
        Output array of shape (T, dx * n_bases), where `dx` is the number of
        input covariates (columns of `x`) and `n_bases` is the number of basis
        functions in `basis`.
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

    Parameters
    ----------
    bt : array-like
        Array of time bin indices where events occur.
    n_bins : int
        Total number of time bins.
    v : numpy.ndarray or None, optional
        Array of values for each event (default is 1 for all).

    Returns
    -------
    numpy.ndarray
        Stimulus array of shape (n_bins, 1) with impulses at specified bins.
    """
    bt = np.asarray(bt)
    bidx = (bt >= 0) & (bt < n_bins)
    bt = bt[bidx]
    o = np.zeros_like(bt, dtype=np.int_)

    v = np.ones_like(bt) if v is None else v[bidx]

    assert len(o) == len(v)

    stim = coo_matrix((v, (bt, o)), shape=(n_bins, 1))
    # print(stim)
    return stim.toarray()


def boxcar_stim(start_bin: int, end_bin: int, n_bins: int, v: ArrayLike = 1.0) -> NDArray:
    """
    Create a boxcar (rectangular) stimulus vector with constant value over a specified interval.

    Parameters
    ----------
    start_bin : int
        Start index of the boxcar (inclusive).
    end_bin : int
        End index of the boxcar (exclusive).
    n_bins : int
        Total number of time bins.
    v : array-like or float, default=1.0
        Value to assign within the boxcar interval.

    Returns
    -------
    numpy.ndarray
        Stimulus array of shape (n_bins, d) with value `v` in
        [start_bin:end_bin], zeros elsewhere; `d` is 1 for scalar `v` or
        equals `len(v)` for a 1D array.

    Raises
    ------
    ValueError
        If v is a multi-dimensional array.
    TypeError
        If v is not a scalar or 1D numpy array.
    """
    if isinstance(v, np.ndarray):
        if v.ndim > 1:
            raise ValueError("v must be a 1D array or scalar")
        else:
            d = v.shape[0]
    elif isinstance(v, (int, float)):
        d = 1
    else:
        raise TypeError("v must be a scalar or 1D Numpy array")
    # print(f"{v=}")
    x = np.zeros((n_bins, d))
    x[start_bin:end_bin, :] = (
        v  # NOTE: neuroGLM effectively uses the right bin edge, but we use the left bin edge.
    )
    # print(f"{x=}")
    return x


def raised_cos(x: ArrayLike, c: ArrayLike, dc: float) -> NDArray:
    """
    Compute a raised cosine basis function.

    Parameters
    ----------
    x : array-like
        Input values.
    c : array-like
        Centers of the raised cosine functions.
    dc : float
        Width parameter for the raised cosine.

    Returns
    -------
    numpy.ndarray
        Evaluated raised cosine basis functions.
    """
    x = np.asarray(x)
    c = np.asarray(c)
    d = 0.5 * (x - c) * np.pi / dc
    d = np.clip(d, -np.pi, np.pi)
    return (np.cos(d) + 1) * 0.5


def make_nonlinear_raised_cos(n_bases, binsize_in_ms, end_points_in_ms, nl_offset_in_ms) -> Basis:
    """
    Create a nonlinearly stretched raised-cosine temporal basis.

    Parameters
    ----------
    n_bases : int
        Number of basis vectors to generate.
    binsize_in_ms : float
        Time bin size in milliseconds.
    end_points_in_ms : array-like
        Sequence of two floats, [first_peak, last_peak], specifying the centers of the first and last basis functions in milliseconds.
    nl_offset_in_ms : float
        Offset for nonlinear stretching in milliseconds (must be > 0).

    Returns
    -------
    Basis
        A Basis object containing the nonlinearly stretched raised-cosine basis matrix and related metadata.

    Raises
    ------
    ValueError
        If nl_offset_in_ms is not greater than 0.
    """
    if nl_offset_in_ms <= 0:
        raise ValueError("nl_offset must be greater than 0")

    # Nonlinearity and its inverse
    def nlin(x):
        """
        Apply the log-based nonlinear warping used for spacing basis centers.

        Parameters
        ----------
        x : ndarray or float
            Input values in milliseconds.

        Returns
        -------
        ndarray or float
            Log-transformed values on the warped axis.
        """
        return np.log(x + 1e-20)

    def invnl(y):
        """
        Invert the nonlinear warping applied by `nlin`.

        Parameters
        ----------
        y : ndarray or float
            Warped values produced by `nlin`.

        Returns
        -------
        ndarray or float
            Values in the original (millisecond) domain.
        """
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
