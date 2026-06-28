"""LLM + embedding provider switch with an OFFLINE fallback.

Default provider is local **Ollama** (free). An `openai` switch is provided for
faster recruiter demos. When neither is reachable, a deterministic *extractive*
fallback lets the retrieval pipeline, graph, and Recall@k/MRR evaluation run with
zero external services — so the whole RAG workstream is testable in CI.

    from src.rag.llm import get_llm, get_embedder
    llm = get_llm(); print(llm.complete("...", context=["..."]))
"""
from __future__ import annotations

import re
from typing import Protocol

from src.config import CFG


class LLM(Protocol):
    def complete(self, question: str, context: list[str]) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- LLMs
class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or CFG.ollama_llm_model
        self.host = host or CFG.ollama_host

    def complete(self, question: str, context: list[str]) -> str:
        import ollama

        client = ollama.Client(host=self.host)
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        prompt = (
            "Answer the question using ONLY the policy context below. If the answer "
            "is not in the context, say you cannot find it. Cite the snippet number.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )
        resp = client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip()


class OpenAILLM:
    def __init__(self, model: str | None = None):
        self.model = model or CFG.openai_llm_model

    def complete(self, question: str, context: list[str]) -> str:
        from openai import OpenAI

        client = OpenAI()
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        msg = (
            "Answer using ONLY the context. Cite snippet numbers. "
            f"If absent, say so.\n\nContext:\n{ctx}\n\nQuestion: {question}"
        )
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": msg}]
        )
        return resp.choices[0].message.content.strip()


class ExtractiveLLM:
    """Offline fallback 'LLM': returns the context sentence most overlapping the
    question. Deterministic, no network — good enough to exercise the pipeline and
    to make faithfulness trivially true (the answer IS the retrieved text)."""

    def complete(self, question: str, context: list[str]) -> str:
        if not context:
            return "I cannot find an answer in the provided policy documents."
        q = set(_tokens(question))
        best, best_score = "", -1.0
        for chunk in context:
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                s = set(_tokens(sent))
                score = len(q & s) / (len(s) + 1)
                if score > best_score:
                    best, best_score = sent.strip(), score
        return best or context[0][:300]


# ---------------------------------------------------------------------- Embedders
class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or CFG.ollama_embed_model
        self.host = host or CFG.ollama_host

    def embed(self, texts: list[str]) -> list[list[float]]:
        import ollama

        client = ollama.Client(host=self.host)
        return [client.embeddings(model=self.model, prompt=t)["embedding"] for t in texts]


class HashingEmbedder:
    """Offline fallback embedder: deterministic hashed bag-of-words vectors.
    Not semantic, but lets the vector path run without Ollama for tests/CI."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import math

        vecs = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in _tokens(t):
                v[hash(tok) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


# ------------------------------------------------------------------------ factory
def get_llm() -> LLM:
    provider = CFG.llm_provider.lower()
    if provider == "ollama" and _ollama_ready():
        return OllamaLLM()
    if provider == "openai" and _openai_ready():
        return OpenAILLM()
    return ExtractiveLLM()


def get_embedder() -> Embedder:
    if CFG.llm_provider.lower() == "ollama" and _ollama_ready():
        return OllamaEmbedder()
    return HashingEmbedder()


def provider_status() -> str:
    if CFG.llm_provider.lower() == "ollama" and _ollama_ready():
        return f"ollama ({CFG.ollama_llm_model})"
    if CFG.llm_provider.lower() == "openai" and _openai_ready():
        return f"openai ({CFG.openai_llm_model})"
    return "offline-extractive-fallback"


def _ollama_ready() -> bool:
    try:
        import ollama

        ollama.Client(host=CFG.ollama_host).list()
        return True
    except Exception:
        return False


def _openai_ready() -> bool:
    import os

    try:
        import openai  # noqa: F401

        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
