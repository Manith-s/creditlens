"""RAG smoke tests: chunking, retrieval returns >=1 chunk, corrective graph
terminates (loop guard), and retrieval metrics are sane."""
from __future__ import annotations

from src.config import CFG
from src.rag.eval import load_golden, mean_reciprocal_rank, recall_at_k
from src.rag.graph import answer_question
from src.rag.ingest import build_chunks, chunk_text
from src.rag.retriever import get_retriever


def test_chunking_no_tiny_sliver_explosion():
    """Regression guard for the overlap bug that emitted ~150 tiny chunks per doc."""
    text = "Sentence one. " * 200  # ~2800 chars
    chunks = chunk_text(text, size=800, overlap=150)
    assert 2 <= len(chunks) <= 8, f"unexpected chunk count {len(chunks)}"
    assert all(len(c) > 150 for c in chunks[:-1])


def test_corpus_chunks_reasonable():
    chunks = build_chunks()
    assert len(chunks) >= 4
    assert len(chunks) < 60, "chunk count exploded — overlap bug regressed"
    assert len({c.source for c in chunks}) >= 4


def test_retrieval_returns_chunks():
    r = get_retriever()
    hits = r.retrieve("adverse action notice timing", CFG.top_k)
    assert len(hits) >= 1
    assert all(h.source for h in hits)


def test_graph_terminates_and_is_grounded():
    res = answer_question("How many days to send an adverse action notice?")
    assert "answer" in res
    assert res["attempts"] <= CFG.max_rag_attempts  # loop guard held
    assert res["answer"]


def test_graph_circuit_breaker_on_nonsense():
    """An unanswerable question must hit the breaker, not loop forever."""
    res = answer_question("What is the airspeed velocity of an unladen swallow?")
    assert res["attempts"] <= CFG.max_rag_attempts


def test_retrieval_metrics_sane():
    r = get_retriever()
    golden = load_golden()
    rec = recall_at_k(r, golden, CFG.top_k)
    mrr = mean_reciprocal_rank(r, golden, CFG.top_k)
    assert 0.0 <= rec <= 1.0 and 0.0 <= mrr <= 1.0
    assert rec >= 0.4, f"recall@{CFG.top_k}={rec:.2f} too low even for the offline backend"
