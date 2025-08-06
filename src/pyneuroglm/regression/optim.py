from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Objective:
    """
    Objective function wrapper for optimization.

    This class caches the gradient and Hessian computations for a given function,
    reducing redundant calculations during optimization routines.

    Parameters
    ----------
    fun : callable
        Function that returns a tuple (value, gradient, Hessian) for given parameters.

    Attributes
    ----------
    _fun : callable
        The objective function.
    _ret : tuple or None
        Cached return value from the last evaluation.
    _x : array-like or None
        Parameters at which the function was last evaluated.
    """

    def __init__(self, fun: Callable[..., tuple[ArrayLike, ArrayLike, ArrayLike]]) -> None:
        """
        Initialize the Objective wrapper.

        Parameters
        ----------
        fun : callable
            Function that returns (value, gradient, Hessian).
        """
        self._fun = fun
        self._ret = None
        self._x = None

    def _compute(self, x, *args) -> tuple[float, NDArray, NDArray]:
        """
        Compute or retrieve cached function, gradient, and Hessian values.

        Parameters
        ----------
        x : array-like
            Parameters at which to evaluate the function.
        *args
            Additional arguments to pass to the function.

        Returns
        -------
        tuple
            Tuple containing (value, gradient, Hessian).
        """
        if self._x is None or self._ret is None or not np.array_equal(x, self._x):
            self._x = x
            self._ret = self._fun(x, *args)
        return self._ret # type: ignore
    
    def function(self, x, *args) -> float:
        """
        Evaluate and return the objective function value.

        Parameters
        ----------
        x : array-like
            Parameters at which to evaluate the function.
        *args
            Additional arguments to pass to the function.

        Returns
        -------
        float
            Value of the objective function.
        """
        ret = self._compute(x, *args)
        return ret[0]
    
    def gradient(self, x, *args) -> NDArray:
        """
        Evaluate and return the gradient of the objective function.

        Parameters
        ----------
        x : array-like
            Parameters at which to evaluate the gradient.
        *args
            Additional arguments to pass to the function.

        Returns
        -------
        numpy.ndarray
            Gradient of the objective function.
        """
        ret = self._compute(x, *args)
        return ret[1]
    
    def hessian(self, x, *args) -> NDArray:
        """
        Evaluate and return the Hessian of the objective function.

        Parameters
        ----------
        x : array-like
            Parameters at which to evaluate the Hessian.
        *args
            Additional arguments to pass to the function.

        Returns
        -------
        numpy.ndarray
            Hessian of the objective function.
        """
        ret = self._compute(x, *args)
        return ret[2]
