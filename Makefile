# CreditLens — common commands. On Windows, run these under WSL2 (Ubuntu) or
# translate to PowerShell. `make help` lists targets.

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environments
.PHONY: venv-ml
venv-ml:  ## Create the ds-ml virtualenv and install ML deps
	$(PY) -m venv .venv-ml && . .venv-ml/bin/activate && \
	  pip install -U pip && pip install -r requirements-ml.txt && pip install -e .

.PHONY: venv-rag
venv-rag:  ## Create the ds-rag virtualenv and install RAG deps
	$(PY) -m venv .venv-rag && . .venv-rag/bin/activate && \
	  pip install -U pip && pip install -r requirements-rag.txt && pip install -e .

# ---------------------------------------------------------------- data
.PHONY: synth
synth:  ## Generate small synthetic datasets so everything runs offline
	$(PY) -m src.data.make_synthetic --all

.PHONY: download
download:  ## Print/run instructions to fetch the real public datasets
	$(PY) -m src.data.download --help

# ---------------------------------------------------------------- services
.PHONY: mlflow
mlflow:  ## Start a local MLflow server (SQLite backend + local artifacts)
	mlflow server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# ---------------------------------------------------------------- workstreams
.PHONY: credit
credit:  ## Run the full credit-risk pipeline (features -> scorecard -> GBM -> SHAP)
	$(PY) -m src.pipelines.run_credit

.PHONY: fraud
fraud:  ## Run the fraud benchmark (GBM vs PyTorch vs Keras) and register champion
	$(PY) -m src.pipelines.run_fraud

.PHONY: rag-ingest
rag-ingest:  ## Ingest policy PDFs into Chroma
	$(PY) -m src.rag.ingest

.PHONY: rag-eval
rag-eval:  ## Evaluate the RAG assistant (RAGAS + Recall@k/MRR + lookup-time benchmark)
	$(PY) -m src.rag.eval

# ---------------------------------------------------------------- quality
.PHONY: test
test:  ## Run the fast test suite
	pytest -m "not slow and not spark and not rag"

.PHONY: test-all
test-all:  ## Run every test (needs Spark/Java and a running Ollama)
	pytest

.PHONY: lint
lint:  ## Lint with ruff
	ruff check src tests

.PHONY: fmt
fmt:  ## Auto-format with ruff
	ruff format src tests && ruff check --fix src tests
