"""
Nonlinearity functions for GLMs.

This module provides nonlinearity functions that return the value, first, and second derivatives.
"""

import numpy as np


def exp(x):
    """
    Exponential nonlinearity and its first and second derivatives.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    f : numpy.ndarray
        Exponential of `x`.
    df : numpy.ndarray
        First derivative of the exponential (same as `f`).
    ddf : numpy.ndarray
        Second derivative of the exponential (same as `f`).
    """
    f = np.exp(x)
    df = f
    ddf = f
    return f, df, ddf
