"""Central configuration: paths, seeds, MLflow URI, dataset schemas.

Import `from src.config import CFG` anywhere. Everything is resolved relative to
the repo root so the code runs the same on Windows (WSL2), macOS, and Linux.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --- repo layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Data root is overridable via CREDITLENS_DATA_DIR so the test suite can use an
# isolated directory and never clobber the data/ the user generated for the
# pipelines. Defaults to <repo>/data for normal runs.
DATA = Path(os.environ.get("CREDITLENS_DATA_DIR", str(ROOT / "data")))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

ARTIFACTS = ROOT / "artifacts"
CHROMA_DIR = ROOT / "chroma"
POLICY_CORPUS = RAW / "policy"

for _p in (RAW, INTERIM, PROCESSED, ARTIFACTS, CHROMA_DIR, POLICY_CORPUS):
    _p.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def rel_to_root(p) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


@dataclass(frozen=True)
class Config:
    # reproducibility
    seed: int = int(_env("RANDOM_SEED", "42"))

    # mlflow
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_credit: str = "creditlens-credit-risk"
    experiment_fraud: str = "creditlens-fraud"

    # spark
    spark_driver_memory: str = _env("SPARK_DRIVER_MEMORY", "6g")
    spark_app_name: str = "creditlens-features"

    # llm / rag
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    ollama_host: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_llm_model: str = _env("OLLAMA_LLM_MODEL", "llama3.1:8b")
    ollama_embed_model: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    openai_llm_model: str = _env("OPENAI_LLM_MODEL", "gpt-4o-mini")
    chroma_collection: str = "policy_docs"

    # rag retrieval params
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_rag_attempts: int = 3  # circuit breaker for the LangGraph corrective loop

    # paths (kept as fields so they're easy to override in tests)
    root: Path = field(default=ROOT)
    raw: Path = field(default=RAW)
    interim: Path = field(default=INTERIM)
    processed: Path = field(default=PROCESSED)
    artifacts: Path = field(default=ARTIFACTS)
    chroma_dir: Path = field(default=CHROMA_DIR)
    policy_corpus: Path = field(default=POLICY_CORPUS)


CFG = Config()


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and (if present) torch/tensorflow. Returns the seed used."""
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
    return seed
