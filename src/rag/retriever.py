"""Retrieval over the policy index, abstracting the Chroma vs offline backends.

`get_retriever()` returns a callable retriever exposing `.retrieve(query, k)` ->
list[RetrievedChunk]. It prefers a persisted Chroma collection; otherwise it loads
the offline chunk store and ranks by cosine over hashed embeddings, with a TF-IDF
lexical fallback. The eval module uses the `.source` field as retrieval ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import CFG
from src.rag.ingest import Chunk, load_offline_chunks
from src.rag.llm import get_embedder


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class ChromaRetriever:
    def __init__(self):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(CFG.chroma_dir))
        self.col = self.client.get_collection(CFG.chroma_collection)
        self.embedder = get_embedder()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or CFG.top_k
        q = self.embedder.embed([query])[0]
        res = self.col.query(query_embeddings=[q], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[0.0] * len(docs)])[0]
        return [
            RetrievedChunk(text=d, source=m.get("source", "?"), score=1.0 - dist)
            for d, m, dist in zip(docs, metas, dists)
        ]


class OfflineVectorRetriever:
    """Cosine similarity over hashed embeddings of the offline chunk store."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = get_embedder()
        self.matrix = self.embedder.embed([c.text for c in chunks])

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or CFG.top_k
        q = self.embedder.embed([query])[0]
        scored = []
        for c, v in zip(self.chunks, self.matrix):
            scored.append((c, _cosine(q, v)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [RetrievedChunk(text=c.text, source=c.source, score=s) for c, s in scored[:k]]


class LexicalRetriever:
    """TF-IDF-ish lexical fallback (no embeddings at all)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.df: dict[str, int] = {}
        self.tokenized = []
        for c in chunks:
            toks = _tokens(c.text)
            self.tokenized.append(toks)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(chunks)

    def _idf(self, t: str) -> float:
        return math.log((self.n + 1) / (self.df.get(t, 0) + 1)) + 1.0

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or CFG.top_k
        q = _tokens(query)
        scored = []
        for c, toks in zip(self.chunks, self.tokenized):
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = sum(tf.get(t, 0) * self._idf(t) for t in q)
            scored.append((c, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [RetrievedChunk(text=c.text, source=c.source, score=float(s)) for c, s in scored[:k]]


def get_retriever():
    # 1) persisted Chroma collection
    try:
        return ChromaRetriever()
    except Exception:
        pass
    # 2) offline store -> vector, else lexical
    chunks = load_offline_chunks()
    if not chunks:
        from src.rag.ingest import build_chunks

        chunks = build_chunks()
    try:
        return OfflineVectorRetriever(chunks)
    except Exception:
        return LexicalRetriever(chunks)


def _tokens(text: str):
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
