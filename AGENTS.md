# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/pyneuroglm/` (core modules: `basis.py`, `design.py`, `experiment.py`, `util.py`, and `regression/` for likelihoods, priors, optimization).
- Tests: `tests/` (e.g., `tests/test_pyneuroglm.py` and assets like `basis.npy`).
- Examples: `example/` (MATLAB data and result PDFs) and `tutorial.md` for a walkthrough.
- Build/metadata: `pyproject.toml` (Python 3.11+, `uv_build` backend), `uv.lock`.

## Build, Test, and Development Commands
- Install (uv): `uv sync --group dev` — create a local env and install runtime + dev tools.
- Install (pip): `pip install -e .` then `pip install pytest matplotlib` for dev extras.
- Test all: `uv run pytest` (add `-q` for quiet) — runs unit tests in `tests/`. Alternatively: `pytest` in a pip venv.
- Run a single test: `pytest tests/test_pyneuroglm.py::test_experiment -q`.
- Build wheel/sdist: `uv build` (or `python -m build` if you prefer the generic tool).

## Coding Style & Naming Conventions
- Python: follow PEP 8; 4‑space indentation; line length ~88–100 chars.
- Names: `snake_case` for functions/variables, `CapWords` for classes, modules remain lower_snake.
- Imports: prefer absolute within `pyneuroglm` (e.g., `from pyneuroglm.basis import ...`).
- Types/docs: add type hints for new/edited public functions; use NumPy docstring style with `Parameters`, `Returns`, `Raises`, `Examples`.

```python
from numpy.typing import NDArray

def conv_basis(x: NDArray, basis: Basis, offset: int = 0) -> NDArray:
    """
    Convolve input data with a temporal basis.

    Parameters
    ----------
    x : ndarray of shape (T, dx)
        Input data matrix.
    basis : Basis
        Basis object providing the basis matrix `B`.
    offset : int, default=0
        Positive = causal, negative = anti-causal, 0 = centered.

    Returns
    -------
    ndarray
        Convolved design matrix of shape (T, n_bases).

    Raises
    ------
    AssertionError
        If `x` is not 2D.

    Examples
    --------
    >>> X = conv_basis(np.random.randn(100, 2), basis, offset=5)
    >>> X.shape
    (100, basis.edim)
    """
```

## Testing Guidelines
- Framework: `pytest` with test files named `test_*.py` under `tests/`.
- Keep tests fast and deterministic (seed NumPy when randomness matters).
- Add tests alongside new features; mirror file structure when practical.
- Optional coverage: if using `coverage`, run `coverage run -m pytest && coverage html`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope-first subject (e.g., `basis: fix raised-cosine edge case`).
- Include context in the body: problem, approach, and any tradeoffs.
- PRs: link related issues, describe changes and validation (commands run, datasets used), and add before/after plots if behavior changes.
- Keep PRs focused; add/update tests and docs (`tutorial.md`/docstrings) with the code change.

## Security & Data Tips
- Avoid committing large binaries; keep example data under `example/`. Use small, anonymized samples for tests.
- Do not embed credentials or paths specific to your machine.
