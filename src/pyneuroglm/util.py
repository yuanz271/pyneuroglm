import numpy as np
from numpy.typing import ArrayLike


def zscore(x: ArrayLike):
    x = np.asarray(x)
    m = np.mean(x, axis=0, keepdims=True)
    s = np.std(x, axis=0, keepdims=True, ddof=1)  # unbiased

    z = (x - m) / s
    return z, m, s
