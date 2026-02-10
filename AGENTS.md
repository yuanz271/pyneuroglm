# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-17
**Commit:** 5e0d778
**Branch:** main

## OVERVIEW

Python port of [neuroGLM](https://github.com/pillowlab/neuroGLM) for building GLMs of neuronal spike trains. Core stack: NumPy, SciPy, scikit-learn. Python ≥3.11.

## STRUCTURE

```
pyneuroglm/
├── src/pyneuroglm/        # Core library
│   ├── __init__.py        # Public API: Experiment, Trial, Variable, DesignMatrix, Covariate
│   ├── basis.py           # Temporal basis functions (raised cosine, boxcar)
│   ├── design.py          # Design matrix construction (681 lines, largest module)
│   ├── experiment.py      # Trial/experiment abstractions
│   ├── util.py            # zscore helper
│   └── regression/        # Bayesian GLM fitting (NO __init__.py - namespace pkg)
│       ├── sklearn.py     # BayesianGLMRegressor (scikit-learn API)
│       ├── posterior.py   # Posterior computation (poisson, bernoulli)
│       ├── likelihood.py  # Log-likelihood functions
│       ├── prior.py       # Prior specification (ridge, gaussian)
│       ├── optim.py       # Objective class for optimization
│       ├── empirical_bayes.py  # Log evidence computation
│       └── nonlinearity.py     # exp() activation
├── tests/                 # Single test file + MATLAB fixture
├── example/               # tutorial.ipynb + MATLAB reference .mat files
└── tutorial.md            # Walkthrough documentation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Define experiment structure | `experiment.py` | `Experiment`, `Trial`, `Variable` classes |
| Build design matrix | `design.py` | `DesignMatrix.add_covariate_*` methods |
| Create temporal basis | `basis.py` | `make_smooth_temporal_basis()`, `make_nonlinear_raised_cos()` |
| Fit GLM | `regression/sklearn.py` | `BayesianGLMRegressor` (sklearn-compatible) |
| Compute posterior | `regression/posterior.py` | `get_posterior_weights()` |
| Run tests | `tests/test_pyneuroglm.py` | 7 smoke tests, MATLAB validation |
| Usage examples | `tutorial.md`, `example/tutorial.ipynb` | End-to-end walkthrough |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Experiment` | Class | experiment.py:60 | Container for trials + variable registry |
| `Trial` | Class | experiment.py:254 | Per-trial data storage |
| `DesignMatrix` | Class | design.py:25 | Builds feature matrices from covariates |
| `Covariate` | Class | design.py:556 | Defines how variables map to features |
| `Basis` | Dataclass | basis.py:19 | Stores basis matrix + reconstruction params |
| `BayesianGLMRegressor` | Class | regression/sklearn.py:134 | sklearn-compatible GLM with empirical Bayes |
| `make_smooth_temporal_basis` | Func | basis.py:51 | Creates raised-cosine basis |
| `conv_basis` | Func | basis.py:178 | Convolves signal with basis |

## CONVENTIONS

### Deviations from Standard
- **No `__init__.py` in `regression/`**: Namespace package. Import modules directly: `from pyneuroglm.regression.sklearn import BayesianGLMRegressor`
- **Build backend**: Uses `uv_build` (not setuptools/hatch). Prefer `uv sync --group dev` over pip
- **Type imports**: Modern style `from collections.abc import Callable` (not `typing.Callable`)

### Coding Style
- PEP 8 with 100-char lines (`ruff` enforced)
- NumPy docstrings (`Parameters`, `Returns`, `Raises`, `Examples` sections)
- 4-space indent, `snake_case` functions, `CapWords` classes
- Absolute imports: `from pyneuroglm.basis import ...`

### Testing
- Single test file `tests/test_pyneuroglm.py` (smoke tests)
- MATLAB reference fixture: `tests/basis.npy`
- Tests NOT seeded - uses `np.random.randn()` directly
- Run: `uv run pytest` or `pytest tests/test_pyneuroglm.py -q`

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** add `__init__.py` to `regression/` without updating all imports
- **DO NOT** use `pip install` for dev setup - use `uv sync --group dev`
- **DO NOT** commit large binaries - stash reference data in `example/`
- **DO NOT** commit without asking the user first
- **NEVER** embed machine-specific paths or credentials
- **NEVER** suppress type errors with `as any` or `# type: ignore`

## UNIQUE STYLES

- **Basis reconstruction**: `Basis` dataclass stores `func` + `kwargs` to regenerate itself: `basis.func(**basis.kwargs)`
- **MATLAB compatibility**: `basis.npy` fixture loaded with `allow_pickle=True` for dtype preservation
- **Design matrix**: Lazy compilation via `DesignMatrix.compile_design_matrix()`
- **Variable types**: Enum-like `VariableType` with CONTINUOUS, TIMING, VALUE, SPIKE

## COMMANDS

```bash
# Setup
uv sync --group dev              # Install all dependencies

# Test
uv run pytest                    # Run full test suite
uv run pytest -q                 # Quiet mode
pytest tests/test_pyneuroglm.py::test_experiment  # Single test

# Build
uv build                         # Create distribution

# Lint
ruff check .                     # Check linting
ruff format .                    # Format code
```

## NOTES

- **Port of MATLAB neuroGLM**: Some naming/patterns mirror original. See `tutorial.md` for mapping.
- **Binsize matters**: `Experiment.binfun()` converts time→bins. Ensure consistent time units.
- **Pre-commit hooks**: `ruff`, `ruff-format`, `pydocstyle --convention=numpy` configured in `.pre-commit-config.yaml`
- **Test fixture**: `basis.npy` validates `make_nonlinear_raised_cos()` against MATLAB output. Note: `assert np.allclose(basis.B[:-1], B)` - size differs by 1.
- **Stale bytecode**: `__pycache__` may contain orphaned `.pyc` for deleted modules (`glm.py`, `negloglik.py`). Safe to delete `__pycache__` dirs if issues arise.
