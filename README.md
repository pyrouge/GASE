# GASE

Graph-Augmented Structural Ensemble (GASE) is a structure-first retrieval pipeline for document-heavy retrieval-augmented generation workloads. It combines semantic retrieval, lexical retrieval, and structural graph expansion so that documents can be searched by meaning, exact phrasing, and document hierarchy at the same time.

## Status

This repository is in alpha. Interfaces, data formats, and evaluation scripts may change while the core indexing and retrieval pipeline is still being refined.

## What the project does

- Parses source documents into a hierarchy-aware chunk tree.
- Indexes the same corpus into vector, BM25, and graph backends.
- Retrieves candidates from multiple signals in parallel.
- Fuses those signals into a single ranked result set with provenance.
- Includes benchmark scripts for quick evaluation and FinanceBench-style experiments.

## Repository layout

- `src/gase/`: library code for configuration, indexing, parsing, and retrieval.
- `tests/`: unit and integration coverage.
- `benchmarks/`: evaluation scripts, datasets, and reports.
- `data/sample_documents/`: small local sample inputs for development.
- `docs/`: project documentation.
- `examples/`: usage examples and experiments.

## Development setup

### Prerequisites

- Python 3.10 or 3.11
- Git

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Run tests

```bash
python -m pytest tests/unit
```

### Run a benchmark script

```bash
python benchmarks/run_quick_eval.py
```

## Version control conventions

- Generated caches and benchmark reports are ignored by Git.
- Local environment files stay untracked except for `.env.example`.
- Sample development assets can stay in the repository when they are small, redistributable, and required for tests or examples.

## Contributing

Contributor workflow, pull request expectations, and local setup guidance are documented in `CONTRIBUTING.md`.

## Security

Report security concerns privately using the process in `SECURITY.md`.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
