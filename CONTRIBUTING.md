# Contributing to CreditLens

Thanks for your interest! This is primarily a portfolio project, but issues and PRs are welcome.

## Development setup

```bash
python -m venv .venv-ml
source .venv-ml/bin/activate      # Windows: .venv-ml\Scripts\activate
pip install -r requirements-ml.txt
pip install -e .
make synth                        # generate offline synthetic data
make test                         # run the fast test suite
make lint                         # ruff
```

The RAG workstream uses a second environment (`requirements-rag.txt`) and, for the
full stack, a local [Ollama](https://ollama.com) runtime. Everything also runs on
offline fallbacks, so tests and `make rag-eval` work without it.

## Guidelines

- **Keep it leakage-safe.** Any resampling (SMOTE, etc.) must live inside the
  modelling pipeline and be fit on training folds only — there are tests that
  enforce this.
- **Report measured numbers.** Don't hard-code metric values; let the pipelines
  compute and log them.
- **Run `make lint` and `make test`** before opening a PR. CI runs both.
- Conventional, descriptive commit messages are appreciated.

## Project layout

See the [Project structure](README.md#-project-structure) section of the README.
