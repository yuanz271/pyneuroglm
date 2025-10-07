"""Public package API exports for pyneuroglm."""

from .design import DesignMatrix, Covariate
from .experiment import Experiment, Trial, Variable

__all__ = [
    "DesignMatrix",
    "Covariate",
    "Experiment",
    "Trial",
    "Variable",
]
