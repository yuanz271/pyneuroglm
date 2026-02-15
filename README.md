# pyneuroglm

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/github/license/yuanz271/pyneuroglm.svg?style=flat-square)](https://choosealicense.com/licenses/mit/)

`pyneuroglm` is a native Python port of [neuroGLM](https://github.com/pillowlab/neuroGLM) for building and fitting generalized linear models of neuronal spike trains.

## Highlights

- Temporal basis utilities for raised cosine and nonlinear time warping constructions.
- Experiment/Trial abstractions that keep stimulus, covariate, and spike data tidy.
- Design matrix helpers to assemble covariates and align them with experimental bins.
- MAP estimation for Poisson GLMs with ridge prior and Laplace posterior uncertainty (scikit-learn API).

## Installation

Install the latest source build directly from GitHub:

```bash
pip install git+https://github.com/yuanz271/pyneuroglm.git
```

## Quickstart

```python
import numpy as np

from pyneuroglm import Experiment, Trial
from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis

# Define an experiment that bins data at 10 ms resolution
expt = Experiment(time_unit="ms", binsize=10, eid=1)

# Populate a trial with 1 second of spike counts
duration_ms = 1000
n_bins = expt.binfun(duration_ms, right_edge=True)
trial = Trial(tid=1, duration=duration_ms)
trial["spikes"] = np.random.poisson(0.1, size=n_bins)

# Build a raised-cosine temporal basis and convolve the spikes
basis = make_smooth_temporal_basis("raised cosine", 100, 5, expt.binfun)
design_matrix = conv_basis(trial["spikes"][:, None], basis, offset=5)
# design_matrix has shape (n_bins, basis.edim)
```

Explore the [`tutorial.md`](./tutorial.md) notebook-style walkthrough and the `example/` directory for MATLAB reference data and outputs.

## License

`pyneuroglm` is distributed under the [MIT License](https://choosealicense.com/licenses/mit/).
