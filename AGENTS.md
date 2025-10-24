# Agent Operating Guide

## 1. Project Orientation
- Core library lives in `src/pyneuroglm/` with key modules (`basis.py`, `design.py`, `experiment.py`, `util.py`, `regression/`).
- Test suite resides in `tests/`; fixtures such as `basis.npy` sit alongside their tests.
- Examples and walkthroughs are under `example/` and `tutorial.md`.
- Build metadata is managed by `pyproject.toml` (Python ≥3.11, `uv_build` backend) and `uv.lock`.

## 2. Environment & Command Rules
- Prefer `uv sync --group dev` for local setup; `pip install -e .` plus `pytest matplotlib` is an alternative.
- Use `uv run pytest` (optionally `-q`) or plain `pytest` for test runs; target a single test with `pytest tests/test_pyneuroglm.py::test_experiment -q`.
- Build artifacts with `uv build` (or `python -m build`).
- Shell invocations must go through `["bash", "-lc", "<cmd>"]` with `workdir` set explicitly.
- Reach for `rg`/`rg --files` when searching; avoid destructive commands (`git reset --hard`, etc.) unless the user demands them.
- Sandbox is `workspace-write`, network is restricted, approval policy is `on-request`; escalate only with a clear justification.

## 3. Workflow Expectations
- Skip formal plans for trivial edits, otherwise note every multi-step effort via the planner (no single-step plans).
- Never assume; ask if instructions conflict. Stop immediately if unexpected repo changes appear.
- Use `apply_patch` for focused edits; reserve scripted replacements or generated files for bulk updates.
- Default to ASCII, add comments sparingly, and maintain user edits you did not originate.

## 4. Coding Standards
- Follow PEP 8 conventions: 4-space indentation, ~88–100 char lines.
- Naming: functions/variables in `snake_case`, classes in `CapWords`, modules lower_snake.
- Favor absolute imports (`from pyneuroglm.basis import ...`).
- Add type hints for public APIs and document them with NumPy-style docstrings (`Parameters`, `Returns`, `Raises`, `Examples`).

## 5. Testing Discipline
- Keep tests deterministic; seed randomness where applicable.
- Mirror source layout for new test modules and ensure they run quickly.
- Optional coverage command: `coverage run -m pytest && coverage html`.

## 6. Change Management
- Commit messages: imperative, scope-first (e.g., `basis: fix raised-cosine edge case`).
- Commit body should capture problem, approach, and trade-offs.
- PRs: stay scoped, link issues, list validation commands/data, attach before/after visuals for behavioral changes.
- Update docs (`tutorial.md`, docstrings) and tests alongside feature work.

## 7. Security & Data Hygiene
- Keep large binaries out of version control; stash reference data in `example/`.
- Never embed credentials or machine-specific paths.
