# Differences from MATLAB neuroGLM

This package (`pyneuroglm`) is a Python port of the MATLAB [neuroGLM](https://github.com/pillowlab/neuroGLM) toolbox by Pillow et al.

While the **core mathematical framework and modeling philosophy are preserved**, there are **important differences** in implementation, supported features, and performance characteristics.

This document enumerates those differences explicitly.

---

## 1. Scope of the Python Port

`pyneuroglm` aims to provide:

**Implemented:**
- Experiment / trial structure
- Design matrix construction
- Temporal basis functions (raised cosine, boxcar)
- Poisson GLMs
- Empirical Bayes estimation for ridge-like priors

**Not implemented:**
- Full feature parity with MATLAB neuroGLM (see details below)
- Bernoulli GLM fitting (posterior stub is not implemented)

---

## 2. Design Matrix Construction

### 2.1 Sparse vs Dense Matrices

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| Default storage | Sparse | Dense (NumPy) |

**Implications:**
- Memory usage may be substantially higher for large experiments
- Very large datasets may not fit in memory

### 2.2 Constant Column Removal

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| Constant column removal | Automatic (`removeConstantCols.m`) | Not automatic |

**Implications:**
- Potential identifiability issues
- Numerical instability in regression
- Differences in posterior estimates compared to MATLAB

**Workaround:** Users should manually ensure covariates vary over time.

### 2.3 Column and Group Index Metadata

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| Column indices per covariate | Explicit | Partial |
| Group indices for priors | Explicit | Partial |

**Implications:**
- Posterior weight interpretation may require additional bookkeeping
- Structured priors are harder to implement without extensions

### 2.4 Z-score Weight Inversion (Bug Fix)

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| `combineWeights` z-score undo | `w .* sigma + mu` | `w / sigma` |

MATLAB `combineWeights.m` (line 22) uses `w .* sigma + mu` to undo
z-scoring when reconstructing weights. This formula inverts the data
transform (`X = Z * sigma + mu`) rather than the weight transform.
For z-scored data `Z = (X - mu) / sigma`, original-space weights are
`w_orig = w_z / sigma`, not `w_z * sigma + mu`. pyneuroglm corrects
this bug.

### 2.5 `delta_stim` Negative Index Filtering

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| `deltaStim` index filter | `bt <= nT` (upper bound only) | `(bt >= 0) & (bt < n_bins)` |

MATLAB `deltaStim.m` filters only the upper bound (`bidx = bt <= nT`).
Negative indices would cause an error in MATLAB's `sparse()` just as they
would in scipy's `coo_matrix`. pyneuroglm adds a lower bound check as
defensive hardening. In normal usage negative indices cannot occur because
`binfun` enforces `t >= 0`.

---

## 3. Supported Priors

### Ridge Prior Intercept Handling

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| `ridge` / `ridge_Cinv` | `speye(nx) * rho` (penalizes all weights) | `intercept_prepended` option to zero out intercept penalty |

MATLAB `+gpriors/ridge.m` returns `speye(nx) * rho` with no special
intercept handling -- the intercept is regularized like any other weight.
pyneuroglm's `ridge_Cinv` adds an `intercept_prepended` parameter that
zeros out `d[0]` to leave the intercept unpenalized, which is standard
practice in regularized regression.

### Implemented

| Prior | MATLAB | Python |
|-------|--------|--------|
| Ridge / Gaussian | Yes | Yes |

### Not Implemented

| Prior | MATLAB | Python |
|-------|--------|--------|
| AR(1) temporal prior | Yes | No |
| Block-diagonal priors | Yes | No |
| Pairwise difference priors | Yes | No |

**Implications:**
- Temporal smoothness must currently be enforced via basis choice
- Some MATLAB models cannot be replicated exactly

---

## 4. Nonlinearities

### Implemented

| Nonlinearity | MATLAB | Python |
|--------------|--------|--------|
| Exponential (`exp`) | Yes | Yes |

### Not Implemented

| Nonlinearity | MATLAB | Python |
|--------------|--------|--------|
| `logexp*` variants | Yes | No |
| `quadnl` | Yes | No |
| `threshLinear` | Yes | No |

**Implications:**
- Python models assume canonical exponential link
- Alternative link functions require custom extensions

---

## 5. Empirical Bayes and Hyperparameter Optimization

| Feature | MATLAB | Python |
|---------|--------|--------|
| Empirical Bayes (ridge) | Yes | Yes |
| Grid search | Yes | No |
| Active learning | Yes | No |

**Implementation Difference:**
- **MATLAB**: Uses **cross-validation** test likelihood for hyperparameter selection
- **Python**: Uses **Laplace approximation** log-evidence for hyperparameter selection

Both approaches are valid for selecting the regularization strength (alpha), but they will produce different optimal hyperparameters. The Laplace approximation is analytical and faster; cross-validation is empirical and may be more robust for small datasets.

---

## 6. Testing and Numerical Parity

| Test Type | MATLAB | Python |
|-----------|--------|--------|
| Basis function parity | Yes | Yes |
| Design matrix parity | Yes | Yes |
| Likelihood parity (log-lik, grad, Hessian) | Yes | Yes |
| Prior parity | Yes | Yes |
| MAP regression parity | Yes | Yes |
| End-to-end reproduction | Yes | Partial |

**Current State:**
- Numerical equivalence is verified for: basis functions, design matrix, Poisson likelihood, ridge prior, and MAP weights
- Tests use MATLAB `.mat` fixtures (`exampleDM.mat`, `exampleData.mat`) for validation
- Tolerances: log-likelihood ~1e-8, gradient ~1e-8, Hessian ~1e-7

---

## 7. API and Structural Differences

### Object-Oriented API

| Aspect | MATLAB neuroGLM | pyneuroglm |
|--------|-----------------|------------|
| Paradigm | Procedural (functions + structs) | Object-oriented (classes) |
| Core types | `designSpec`, `expt` structs | `Experiment`, `Trial`, `DesignMatrix`, `Covariate` |

**Implications:**
- Python API is more explicit and extensible
- MATLAB scripts cannot be ported line-by-line

### Import Structure

```python
from pyneuroglm import Experiment, Trial, DesignMatrix
from pyneuroglm.basis import make_smooth_temporal_basis
from pyneuroglm.regression.sklearn import BayesianGLMRegressor
```

Note: `regression/` is a namespace package (no `__init__.py`).

---

## 8. Performance Considerations

Python implementation prioritizes **clarity and correctness** over raw performance.

MATLAB implementation is more optimized for:
- Sparse operations
- Large-scale experiments

**Recommendations for large datasets:**
- Use appropriate bin sizes
- Limit covariate dimensionality
- Monitor memory usage

---

## 9. Summary Table

| Feature | MATLAB neuroGLM | pyneuroglm |
|---------|-----------------|------------|
| Design matrix sparsity | Yes | No |
| Constant column removal | Yes | No |
| Structured priors (AR1, blkdiag) | Yes | No |
| Multiple nonlinearities | Yes | No |
| Empirical Bayes | Yes | Yes |
| Basis parity tests | Yes | Yes |
| Design matrix parity tests | Yes | Yes |
| Likelihood/prior parity tests | Yes | Yes |
| MAP regression parity tests | Yes | Yes |
| End-to-end parity tests | Yes | Partial |

---

## 10. When to Use Each

### Use pyneuroglm if you:

- Want a Python-native GLM framework
- Prefer scikit-learn style APIs
- Use raised-cosine temporal bases
- Do not require AR1 or structured priors
- Work with moderately-sized datasets

### Use MATLAB neuroGLM if you:

- Require strict numerical equivalence to published results
- Depend on structured priors (AR1, block-diagonal)
- Run very large sparse experiments
- Need features not yet ported

---

## 11. Completed Parity Improvements

The following parity tests have been implemented:

- [x] Design matrix parity (`test_design_matrix_parity.py`)
- [x] Poisson likelihood parity (log-lik, gradient, Hessian)
- [x] Ridge prior parity
- [x] MAP regression parity
- [x] Laplace log-evidence validation (mathematical correctness)

**Bug Fixes (discovered during parity audit):**
- Hessian formula in `likelihood.py` (commit `8ea3fde`)
- `Cinv` dimension mismatch when `fit_intercept=True` (commit `05b270e`)
- Optimizer tolerance settings for Newton-CG (commit `05b270e`)
- Sign error in `empirical_bayes.py` - Sinv should be `-(ddL + ddP)`

### 11.1 MAP Parity Check Details

MAP parity is validated using a MATLAB fixture (`exampleMAP.mat`) when it is available.

**Key findings:**

1. Same objective: neg log-posterior = neg log-likelihood + neg log-prior
2. Same ridge prior construction: `Cinv[0,0]=0` (no intercept regularization)
3. Same optimum: MATLAB and Python reach the same objective value within tolerance

**Optimizer note:** MATLAB typically uses `fminunc` (quasi-Newton). Python uses `scipy.optimize` with `trust-ncg`, which can converge to a tighter gradient tolerance.

**Test coverage:**
- `test_map_parity_with_matlab` in `tests/test_poisson_likelihood_parity.py`
- Skips automatically if `exampleMAP.mat` is not present

## 12. Planned Improvements

The following improvements are planned to increase parity:

- [ ] Sparse design matrix support (`scipy.sparse`)
- [ ] Constant column removal
- [ ] AR(1) temporal priors
- [ ] Explicit group index bookkeeping
- [ ] Empirical Bayes parity test
- [ ] End-to-end reproduction test (full tutorial workflow)

---

## 13. Contributing

Contributions toward MATLAB parity are welcome. Priority areas:

1. Sparse matrix support
2. Structured priors
3. Parity test infrastructure

See the main README for contribution guidelines.
