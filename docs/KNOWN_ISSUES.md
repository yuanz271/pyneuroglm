# Known Issues

Bugs identified in `src/pyneuroglm/` as of 2026-02-15. Ordered by severity.

---

## Summary

| # | Severity | File | Line | Summary |
|---|----------|------|------|---------|
| 1 | FIXED | `regression/optim.py` | 62 | `Objective` cache stores array reference — stale gradient/Hessian |
| 2 | FIXED | `design.py` | 453 | `combine_weights` z-score inversion formula is wrong |
| 3 | FIXED | `regression/sklearn.py` | 109 | `log_evidence_scorer` always adds intercept |
| 4 | FIXED | `design.py` | 170 | `add_covariate_constant` ignores `stim_label` parameter |
| 5 | FIXED | `design.py` | 407 | `compile_design_matrix` crashes when `condition` excludes a covariate |
| 6 | FIXED | `basis.py` | 243 | `delta_stim` doesn't filter negative bin indices |
| 7 | FIXED | `regression/prior.py` | 45 | `ridge_Cinv` docstring claims wrong return shape |
| 8 | Low | `regression/posterior.py` | 165 | `initialize_zero` sets intercept to `mean(y)` not `log(mean(y))` |

---

## Details

### 1. `Objective` cache stores array reference, not copy

**Location:** `src/pyneuroglm/regression/optim.py:62`

```python
self._x = x  # stores a reference, not a copy
```

The cache key comparison `np.array_equal(x, self._x)` always returns `True`
when scipy reuses the same array object with different values, because both
names point to the same memory. The cache never invalidates, so `gradient()`
and `hessian()` can return results computed at a previous point.

**Impact:** In practice, scipy's `trust-ncg` reuses the same array object
with a different value only once (at the first iteration transition). The
optimizer self-corrects, so convergence is not affected in observed tests.
The bug is still real: the cache returns one stale value per run.

**Fixed:** Changed `self._x = x` to `self._x = x.copy()` in `optim.py:62`.
For a 10k-element vector, `copy()` costs ~1 µs and `np.array_equal` costs
~3 µs — both negligible compared to the likelihood/Hessian computation the
cache avoids. Regression test in `tests/test_objective_cache.py`.

---

### 2. `combine_weights` z-score inversion formula is wrong

**Location:** `src/pyneuroglm/design.py:453`

```python
w = w * self.zstats["s"].squeeze() + self.zstats["m"].squeeze()
```

If the design matrix was z-scored as `X_z = (X - m) / s`, the original-space
weights should be `w_orig = w_z / s`. The current formula `w * s + m` inverts
the *data* transform, not the *weight* transform. The intercept also needs
adjustment: `intercept_orig = intercept_z - (m / s) @ w_z`.

**Impact:** Any user who calls `zscore_columns()` then `combine_weights()`
gets incorrect weight reconstructions.

**Fixed:** Replaced `w * s + m` with `w / s` in `design.py:444`. This is an
inherited bug from MATLAB neuroGLM (`+buildGLM/combineWeights.m` line 22),
which uses the same wrong formula `w .* sigma + mu`. Regression test in
`tests/test_pyneuroglm.py::test_combine_weights_zscore_inversion`.

---

### 3. `log_evidence_scorer` always prepends intercept

**Location:** `src/pyneuroglm/regression/sklearn.py:109`

```python
if hasattr(estimator, "intercept_"):
```

`BayesianGLMRegressor` always sets `self.intercept_ = 0.0`, even when
`fit_intercept=False`. So `hasattr` is always `True`, and
`log_evidence_scorer` always prepends a ones column and intercept weight.
This computes evidence for a model *with* intercept when the model was fit
without one.

**Impact:** `score()` and `log_evidence_scorer()` return wrong values when
`fit_intercept=False`.

**Fixed:** Replaced `hasattr(estimator, "intercept_")` with
`getattr(estimator, "fit_intercept", False)` in `sklearn.py:109`. This
follows sklearn convention where all linear estimators set `intercept_`
(to `0.0` when disabled) but only those with `fit_intercept=True` actually
model an intercept. Regression test in
`tests/test_pyneuroglm.py::test_log_evidence_scorer_respects_fit_intercept`.

---

### 4. `add_covariate_constant` ignores `stim_label`

**Location:** `src/pyneuroglm/design.py:170`

```python
constant_stim(label, binfun),  # should be stim_label, not label
```

The method accepts a `stim_label` parameter (defaulting to `label`) but
always passes `label` to `constant_stim()`. When a user provides a different
`stim_label`, the wrong trial key is used to look up data.

**Fixed:** Changed `constant_stim(label, binfun)` to
`constant_stim(stim_label, binfun)` in `design.py:170`. Regression test in
`tests/test_pyneuroglm.py::test_add_covariate_constant_uses_stim_label`.

---

### 5. `compile_design_matrix` crashes when `condition` excludes a covariate

**Location:** `src/pyneuroglm/design.py:407–415`

When a covariate has a `condition` callable that returns `False` for a
particular trial, the covariate's columns are skipped entirely in the
per-trial block `Xt`. This produces fewer columns than `self.edim`, and the
assertion `Xt.shape == (n_bins, self.edim)` fails with `AssertionError`.

**Impact:** The `condition` parameter on any `add_covariate_*` method is
unusable — it always crashes if any trial is excluded.

**Fixed:** Replaced `append`/`column_stack` with pre-allocated
`np.zeros((n_bins, self.edim))` and column-index assignment, matching the
MATLAB `compileSparseDesignMatrix.m` approach. Excluded covariates leave
their columns as zeros. Regression test in
`tests/test_pyneuroglm.py::test_compile_design_matrix_with_condition`.

---

### 6. `delta_stim` doesn't filter negative bin indices

**Location:** `src/pyneuroglm/basis.py:243`

```python
bidx = bt < n_bins  # only filters upper bound
```

The filter removes bins `>= n_bins` but not negative values. If a negative
index reaches `coo_matrix`, scipy raises
`ValueError: negative axis 0 index`. In normal usage `binfun` enforces
`t >= 0`, so negative indices are unlikely but not impossible (e.g., if
`delta_stim` is called directly with raw data).

**Fixed:** Changed `bidx = bt < n_bins` to `bidx = (bt >= 0) & (bt < n_bins)`
in `basis.py:243`. Regression test in
`tests/test_pyneuroglm.py::test_delta_stim_filters_negative_indices`.

---

### 7. `ridge_Cinv` docstring claims wrong return shape

**Location:** `src/pyneuroglm/regression/prior.py:45`

The docstring says the return shape is `(nx+1, nx+1) if
intercept_prepended is True`, but the code always returns `(nx, nx)` — it
zeros out `d[0]` rather than expanding the matrix. The caller
(`BayesianGLMRegressor.fit`) already passes `X_.shape[1]` which includes the
intercept column, so the code is correct; only the docstring is wrong.

**Fixed:** Changed docstring to `Diagonal matrix of shape (nx, nx)` in
`prior.py:45`.

---

### 8. `initialize_zero` sets intercept to `mean(y)` instead of `log(mean(y))`

**Location:** `src/pyneuroglm/regression/posterior.py:165`

```python
w[0] = w0  # w0 = np.mean(y)
```

For a Poisson GLM with exp link, the intercept-only model predicts
`lambda = exp(w0)`, so the correct initialization is `w0 = log(mean(y))`.
The `nlin` parameter on `initialize_zero` exists to handle this, but
`get_posterior_weights` never passes it.

**Impact:** Only affects `initialize='zero'`; the default `initialize='lstsq'`
is not affected. The optimizer can recover from a bad starting point, but
convergence may be slower.

**Fix:** In `get_posterior_weights`, pass `nlin=np.log` when constructing
`init_kwargs` for the zero initializer, or change the default in
`initialize_zero` to apply `log` when the distribution is Poisson.
