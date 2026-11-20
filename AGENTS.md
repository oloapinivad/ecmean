# AGENTS.md

Guidance for AI coding agents (and humans) working on ecmean. Principles: **reuse the existing structure,
build on the `Diagnostic` class, deduplicate instead of copying, change as little as possible, and document
every change.**

**Never push to `main` and never merge a pull request.** Work on a branch, open a PR, and leave
reviewing and merging to the maintainers.

## What ecmean is

A lightweight evaluation tool for climate model output with two diagnostics, both exposed as a class, a
function and a command-line entry point:

| Diagnostic | Class / function | CLI | Output |
|---|---|---|---|
| Global means | `GlobalMean` / `global_mean()` | `global_mean` | YAML + text table, optional plots |
| Performance indices | `PerformanceIndices` / `performance_indices()` | `performance_indices` | YAML, optional heatmap |

## Layout

```
ecmean/
  global_mean.py, performance_indices.py   # the two diagnostics: class + static worker + entry point
  libs/
    diagnostic.py   # Diagnostic: configuration container shared by both diagnostics
    support.py      # Supporter: areas, masks, interpolation weights
    files.py        # interface/config YAML loading, filename building, open_input_files(), var_is_there()
    formula.py      # parser + evaluator for "derived" variables of the interface files
    units.py        # UnitsHandler, apply_units_conversion()
    general.py      # Timer, weight_split, check_* helpers, dict utilities
    masks.py, areas.py, ncfixers.py, climatology.py, ecplotter.py, loggy.py, parser.py
  interfaces/       # interface_*.yml: map model variable names/files to CMOR-like names
  reference/        # gm_reference_*.yml: global-mean references
  climatology/      # PI climatologies per name and resolution
  utils/            # offline scripts that build references/climatologies (ruff-excluded)
tests/              # pytest, run from the repository root (paths are relative)
```

## Architecture in one paragraph

A diagnostic is built around a `Diagnostic` instance. `Diagnostic.__init__` reads the config (path or dict),
resolves directories, interface file and model name, and dispatches on `funcname` to `cfg_global_mean()` or
`cfg_performance_indices()` to set variables, regions, seasons and the reference/climatology file.
`prepare()` loads the interface and reference files, builds a `Supporter` and calls
`Diagnostic.configure_amip_omip_cpld()` to drop variables for which no atmosphere/ocean grid is available.
`run()` splits the variable list with `weight_split`, starts one `Process` per chunk running a **static**
worker (`gm_worker` / `pi_worker`) and collects results in `Manager` dictionaries. `store()` and `plot()`
write the outputs.

## Prerequisite: ruff

ruff is mandatory. CI runs `ruff check .` and `ruff format --check .` as a gate before the tests, so a change
that does not pass them cannot be merged.

- Install the pinned version: `pip install "ruff==0.17.1"` (or `pip install -e .[style]`), and optionally enable
  the hooks with `pre-commit install` (`.pre-commit-config.yaml` runs the same ruff version).
- **Before starting**, run `ruff check . && ruff format --check .` to confirm a clean baseline.
- **Before every commit**, run `ruff check . && ruff format .` and commit the result. Never commit code that
  fails them, and never silence a rule with `# noqa` or by editing `[tool.ruff]` in `pyproject.toml` to get a
  change through; fix the code. Format only the files you touched, so the diff stays minimal.
- Configuration lives in `pyproject.toml` (line length 127, double quotes, isort with `ecmean` as first party;
  `ecmean/utils/**` and `__init__.py` files are excluded).

## Rules for changes

### 1. Build on `Diagnostic`, do not parallel it
- Anything that is configuration, a path, a filename, a default or a deprecation check belongs in
  `libs/diagnostic.py`, not in the diagnostics or in a new class.
- `Diagnostic` already branches on `funcname` in several places (`__init__`, `filenames`,
  `configure_amip_omip_cpld`). Extend those branches consistently; if a third diagnostic or a repeated
  branch appears, refactor the shared part into one method instead of adding another copy.
- Use the existing attribute names (`var_all`/`field_all`, `reffile`/`climfile`, `tabdir`, `figdir`,
  `years_joined`, ...). Do not introduce synonyms.

### 2. Deduplicate before you add
`GlobalMean` and `PerformanceIndices` are siblings with much common code. Before writing something in one,
check the other and `libs/`:
- If the same code would exist in both, put it once in `libs/` as a plain function (or in `Diagnostic` if it
  is configuration). **No base-class hierarchy**: keep the two classes flat and independent.
- Existing shared helpers to reuse: `files.open_input_files()` (the only way to open input data),
  `formula._eval_formula()` / `formula_variables()` (the only formula parser), `units.apply_units_conversion()`
  (the only place that fixes the offset/factor order), `general.Timer`, `general.weight_split`,
  `files.make_input_filename`, `files.var_is_there`.
- Known remaining duplication (candidates for a future shared helper): the `prepare()` setup, the
  `Manager`/`Process`/`join` loop in `run()`, the worker preamble. Do not fix these opportunistically inside
  an unrelated change.

### 3. Minimal intervention
- Make the smallest change that solves the problem; no drive-by refactors, renames, reformatting or
  dependency changes. If you spot something unrelated, mention it in the PR or open an issue instead.
- Preserve public API: `GlobalMean`, `global_mean`, `PerformanceIndices`, `performance_indices`,
  `Diagnostic`, `Supporter`, `UnitsHandler` (exported in `ecmean/__init__.py`), the CLI options, the
  `toc()`/`final_toc()` methods, and the YAML/interface/config file formats. If a change must break one of
  these, say so explicitly in the changelog.
- Do not leave commented-out code; delete it (git keeps history).
- Do not change reference values, climatologies or test reference files (`tests/table/*.ref`) unless that is
  the purpose of the change, and then state why and by how much they move.

### 4. Document every modification
- Add a line to `CHANGELOG.md` under `## [Unreleased]`, with the PR/issue number, in the existing style.
  Flag anything that changes numerical results with `*IMPORTANT*`.
- Update docstrings of the functions you touch (Google style `Args:`/`Returns:`, as in the existing code) and the
  Sphinx docs in `docs/sphinx` if user-facing behaviour or options change.
- Explain non-obvious assumptions where they are made. Example: `apply_units_conversion` documents that
  `units_converter()` never returns both a non-zero offset and a factor other than one.
- Describe in the PR what changed, why, what was deliberately left out, and which tests were run.

## Domain gotchas

- **Multiprocessing:** workers are `@staticmethod`s run in separate processes, so everything they receive must
  be picklable and shared results go through `Manager` dictionaries. Per-process initialisation (e.g.
  `units_extra_definition()`) must happen inside the worker (required from Python 3.14).
- **Units:** conversions return `(offset, factor)`; never both are non-trivial. Flux direction (`up`/`down`)
  is handled through the sign of the factor. Do not hand-roll conversions in a diagnostic.
- **Derived variables:** the `derived:` strings in interface files support names (also starting with a digit,
  e.g. `2t`, parsed through placeholders), numbers (decimals, scientific notation), `+ - * /`, unary minus,
  parentheses and spaces. Unsupported syntax raises `ValueError`. Formulas are parsed with Python's `ast` module restricted to these node types (nothing is
  executed); extend that whitelist in `formula.py` rather than adding another parser.
- **Opening data:** use `open_input_files()` so both diagnostics share `combine`/`join` behaviour and react
  identically to xarray default changes.
- **3D PIs:** pressure-level weights depend on level ordering and bounds (see #150); take care when touching
  `atm3d` fields.
- **Deprecations:** deprecated climatologies/references are listed in `libs/climatology.py` and warned about in
  `Diagnostic`; follow that pattern for new deprecations.

## Development workflow

- Use the dev environment (`environment-dev.yml`); the test suite imports `ecmean` from the checkout.
- Run tests from the repository root: `pytest tests`. Fast single-area runs: `pytest tests/test_formula.py
  tests/test_units.py`. The GM/PI tests run the CLIs and take a few minutes.
- Tests write into `tests/table` and `tests/pluto`; they must clean up what they generate (`CLEANUP`
  flags) and must not rely on the environment being activated (CLI tests use the running interpreter's
  executables).
- Add or extend a test for each behaviour change; prefer small unit tests (see `tests/test_formula.py`,
  `tests/test_units.py`) next to the end-to-end ones. Keep PI/GM comparisons tight: do not loosen
  `TOLERANCE` to make a test pass.
- Ruff must pass (see the prerequisite above) before tests and commits.
- Never push to `main` and never merge a PR (yours or anyone's); push a branch and open a PR.
- Branch from `main` (or the branch your change depends on), keep PRs focused on one concern, and reference
  the issue number in commits and the changelog.
