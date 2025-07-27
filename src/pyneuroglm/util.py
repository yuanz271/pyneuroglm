import numpy as np
from numpy.typing import ArrayLike, NDArray


def zscore(x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    x = np.asarray(x)
    m = np.mean(x, axis=0, keepdims=True)
    s = np.std(x, axis=0, keepdims=True, ddof=1)  # unbiased

    z = (x - m) / s
    return z, m, s
