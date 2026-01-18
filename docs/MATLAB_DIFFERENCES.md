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
- Poisson and Bernoulli GLMs
- Empirical Bayes estimation for ridge-like priors

**Not implemented:**
- Full feature parity with MATLAB neuroGLM (see details below)

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

---

## 3. Supported Priors

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

---

## 6. Testing and Numerical Parity

| Test Type | MATLAB | Python |
|-----------|--------|--------|
| Basis function parity | Yes | Yes |
| Design matrix parity | Yes | No |
| Posterior equivalence | Yes | No |
| End-to-end reproduction | Yes | No |

**Implications:**
- Numerical equivalence is guaranteed for basis functions
- Full model equivalence is **not yet guaranteed**

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
| End-to-end parity tests | Yes | No |

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

## 11. Planned Improvements

The following improvements are planned to increase parity:

- [ ] Sparse design matrix support (`scipy.sparse`)
- [ ] Constant column removal
- [ ] AR(1) temporal priors
- [ ] Explicit group index bookkeeping
- [ ] MATLAB `.mat` parity regression tests

---

## 12. Contributing

Contributions toward MATLAB parity are welcome. Priority areas:

1. Sparse matrix support
2. Structured priors
3. Parity test infrastructure

See the main README for contribution guidelines.
