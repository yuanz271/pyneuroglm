# Test Reorganization Plan

**Created:** 2026-02-15
**Branch:** `refactor/test-layout`

---

## Goal

Mirror `src/pyneuroglm/` layout in `tests/`, one test file per module,
with minimal redundancy.

## Current layout

```
tests/
├── basis.npy                           # MATLAB fixture
├── test_design_matrix_parity.py        # 3 tests (slow, MATLAB parity)
├── test_empirical_bayes_quadrature.py  # 6 tests
├── test_objective_cache.py             # 5 tests
├── test_poisson_likelihood_parity.py   # 13 tests
└── test_pyneuroglm.py                 # 13 tests (mixed modules)
```

## Target layout

```
tests/
├── basis.npy                           # MATLAB fixture (keep)
├── test_basis.py                       # basis.py
├── test_design.py                      # design.py
├── test_experiment.py                  # experiment.py
├── regression/
│   ├── test_empirical_bayes.py         # regression/empirical_bayes.py
│   ├── test_likelihood.py             # regression/likelihood.py
│   ├── test_optim.py                  # regression/optim.py
│   ├── test_prior.py                  # regression/prior.py
│   └── test_sklearn.py               # regression/sklearn.py
```

No test files for:
- `util.py` -- zscore is trivial, tested indirectly via `test_design.py`
- `regression/nonlinearity.py` -- trivial exp(), tested via likelihood tests
- `regression/posterior.py` -- tested through `test_sklearn.py`

## Migration map

### test_pyneuroglm.py (13 tests)

| Test | Destination | Action |
|------|-------------|--------|
| `test_experiment` | `test_experiment.py` | move |
| `test_variable` | `test_experiment.py` | move |
| `test_trial` | `test_experiment.py` | move |
| `test_make_smooth_temporal_basis` | `test_basis.py` | move |
| `test_conv_basis` | `test_basis.py` | move |
| `test_combine_weights` | -- | remove (tests namedtuple, not our code) |
| `test_make_nonlinear_raised_cos` | `test_basis.py` | move |
| `test_combine_weights_zscore_inversion` | `test_design.py` | move |
| `test_log_evidence_scorer_respects_fit_intercept` | `regression/test_sklearn.py` | move |
| `test_add_covariate_constant_uses_stim_label` | `test_design.py` | move |
| `test_compile_design_matrix_with_condition` | `test_design.py` | move |
| `test_delta_stim_filters_negative_indices` | `test_basis.py` | move |
| `test_initialize_zero_poisson_uses_log` | `regression/test_sklearn.py` | move |

### test_objective_cache.py (5 tests)

| Test | Destination | Action |
|------|-------------|--------|
| `test_cache_invalidates_on_inplace_mutation` | `regression/test_optim.py` | move |
| `test_cache_hit_same_values` | `regression/test_optim.py` | merge with `test_cache_miss_different_values` |
| `test_cache_miss_different_values` | `regression/test_optim.py` | merge into above |
| `test_flip_sign` | `regression/test_optim.py` | move |
| `test_optimizer_convergence_with_cache` | `regression/test_optim.py` | move |

### test_poisson_likelihood_parity.py (13 tests)

| Test | Destination | Action |
|------|-------------|--------|
| `TestPoissonLikelihoodParity` (4 tests) | `regression/test_likelihood.py` | move |
| `TestPriorParity` (3 tests) | `regression/test_prior.py` | move |
| `TestMAPRegression` (6 tests) | `regression/test_sklearn.py` | move |

### test_empirical_bayes_quadrature.py (6 tests)

| Test | Destination | Action |
|------|-------------|--------|
| `TestLaplaceExactGaussian` (3 tests) | `regression/test_empirical_bayes.py` | parametrize into 1 |
| `TestLaplaceQuadraturePoisson1D` (3 tests) | `regression/test_empirical_bayes.py` | parametrize into 1 |

### test_design_matrix_parity.py (3 tests)

| Test | Destination | Action |
|------|-------------|--------|
| `TestDesignMatrixParity` (3 tests) | `test_design.py` | move |

## Shared test infrastructure

### tests/regression/conftest.py

`test_poisson_likelihood_parity.py` contains shared helpers used across
`TestPoissonLikelihoodParity`, `TestPriorParity`, and `TestMAPRegression`:
- `reference_poisson_negloglik()` -- reference implementation
- `load_design_matrix_and_spikes()` -- MATLAB fixture loader

These split into three destination files (`test_likelihood.py`,
`test_prior.py`, `test_sklearn.py`). Shared helpers and fixtures must live
in `tests/regression/conftest.py` to avoid duplication.

### tests/regression/ directory

No `__init__.py` needed. pytest discovers tests without it, consistent with
the namespace package convention used in `src/pyneuroglm/regression/`.

### test_design.py will be the largest file

`test_design_matrix_parity.py` carries ~200 lines of parity infrastructure
(`load_matlab_design_matrix`, `load_matlab_trial_data`,
`build_python_experiment`, `build_design_matrix_like_matlab`,
`compare_shapes`, `compare_values`, `find_mismatched_columns`,
`ColumnIdentity`, `ParityFailure`). All move into `test_design.py`.

### posterior.py coverage

`regression/posterior.py` has no dedicated test file. It is covered by:
- `regression/test_sklearn.py`: `test_first_order_optimality`,
  `test_hessian_negative_definite` call `poisson_posterior()` directly
- `regression/test_sklearn.py`: `test_initialization_parity_with_matlab`
  tests `initialize_lstsq()`

## Simplifications

- **Remove** `test_combine_weights`: tests Python's `namedtuple`, not pyneuroglm
- **Merge** `test_cache_hit_same_values` + `test_cache_miss_different_values` into
  one test with both assertions
- **Parametrize** `TestLaplaceExactGaussian` (3 tests differ only in matrix
  size and alpha) into 1 test with `@pytest.mark.parametrize`
- **Parametrize** `TestLaplaceQuadraturePoisson1D` (3 tests differ only in
  alpha) into 1 test with `@pytest.mark.parametrize`; include per-case
  tolerance in params (0.05 for standard/strong prior, 0.1 for weak prior)

## Files to delete after migration

- `tests/test_pyneuroglm.py`
- `tests/test_objective_cache.py`
- `tests/test_poisson_likelihood_parity.py`
- `tests/test_empirical_bayes_quadrature.py`
- `tests/test_design_matrix_parity.py`

## Execution order

1. Create branch `refactor/test-layout`
2. Create `tests/regression/` directory
3. Create `tests/regression/conftest.py` with shared helpers and fixtures
4. Write new test files, migrating and simplifying
5. Verify all tests pass with `uv run pytest -v`
6. Delete old test files
7. Final verification
