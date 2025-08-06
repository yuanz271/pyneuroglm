import numpy as np
from numpy.typing import ArrayLike, NDArray


def zscore(x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute the z-score of an array along the first axis.

    Parameters
    ----------
    x : array-like
        Input data array.

    Returns
    -------
    z : numpy.ndarray
        Z-scored array with the same shape as `x`.
    m : numpy.ndarray
        Mean of `x` along the first axis, shape matches `x` with axis 0 reduced.
    s : numpy.ndarray
        Standard deviation of `x` along the first axis, shape matches `x` with axis 0 reduced.

    Notes
    -----
    The z-score is computed as ``(x - m) / s``, where `m` is the mean and `s` is the unbiased standard deviation (ddof=1).
    """
    x = np.asarray(x)
    m = np.mean(x, axis=0, keepdims=True)
    s = np.std(x, axis=0, keepdims=True, ddof=1)  # unbiased

    z = (x - m) / s
    return z, m, s
