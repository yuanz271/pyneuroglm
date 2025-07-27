from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Objective:
    """Objectiove function wrapper
    cache gradient and Hessian
    """
    def __init__(self, fun: Callable[..., tuple[ArrayLike, ArrayLike, ArrayLike]]) -> None:
        self._fun = fun
        self._ret = None
        self._x = None

    def _compute(self, x, *args) -> tuple[float, NDArray, NDArray]:
        if self._x is None or self._ret is None or not np.array_equal(x, self._x):
            self._x = x
            self._ret = self._fun(x, *args)
        return self._ret # type: ignore
    
    def function(self, x, *args) -> float:
        ret = self._compute(x, *args)
        return ret[0]
    
    def gradient(self, x, *args) -> NDArray:
        ret = self._compute(x, *args)
        return ret[1]
    
    def hessian(self, x, *args) -> NDArray:
        ret = self._compute(x, *args)
        return ret[2]
