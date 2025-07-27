"""
Nonlinearity functions returning value, 1st and 2nd order derivatives
"""
import numpy as np


def exp(x):
    f = np.exp(x)
    df = f
    ddf = f
    return f, df, ddf
