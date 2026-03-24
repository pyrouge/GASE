# Contributing to GASE

## Ground rules

- Keep changes focused. Avoid bundling unrelated refactors with bug fixes or features.
- Prefer tests with every behavioral change.
- Update documentation when public behavior, setup steps, or interfaces change.
- Do not commit generated caches, local datasets, virtual environments, or benchmark reports.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if you need local configuration overrides.

## Development workflow

1. Create a topic branch from `main`.
2. Make the smallest change that solves the problem.
3. Run the relevant tests locally.
4. Update documentation when needed.
5. Open a pull request with a clear summary, rationale, and test evidence.

## Suggested local checks

```bash
python -m pytest tests/unit
python -m pytest tests/integration -m "not slow"
python -m ruff check .
python -m black --check .
```

Run only the checks that are relevant to your change if external services or large datasets are not available.

## Pull request checklist

- The change is scoped to one problem.
- Tests were added or updated when behavior changed.
- Existing tests still pass locally, or any known failures are explained.
- Documentation was updated if setup, configuration, or behavior changed.
- Sensitive data, generated files, and local-only outputs are not included.

## Commit guidance

Use short, descriptive commit messages written in the imperative mood. Example: `Add graph cache ignore rules`.
