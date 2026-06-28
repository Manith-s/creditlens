"""Evaluate the RAG assistant."""
from __future__ import annotations

import json
import time

from src.config import CFG
from src.rag.graph import answer_question
from src.rag.llm import get_llm, provider_status
from src.rag.retriever import get_retriever


def load_golden() -> list[dict]:
    path = CFG.raw / "policy_golden_qa.json"
    if not path.exists():
        raise FileNotFoundError(
            "policy_golden_qa.json missing. Run `python -m src.data.make_policy_corpus`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def recall_at_k(retriever, golden: list[dict], k: int) -> float:
    hits = 0
    for qa in golden:
        retrieved = retriever.retrieve(qa["question"], k)
        sources = {r.source for r in retrieved}
        hits += int(qa["source"] in sources)
    return hits / len(golden)


def mean_reciprocal_rank(retriever, golden: list[dict], k: int) -> float:
    total = 0.0
    for qa in golden:
        retrieved = retriever.retrieve(qa["question"], k)
        rr = 0.0
        for rank, r in enumerate(retrieved, 1):
            if r.source == qa["source"]:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(golden)


def ragas_scores(golden: list[dict]) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        return None
    try:
        retriever = get_retriever()
        rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        for qa in golden:
            res = answer_question(qa["question"], retriever=retriever)
            ctx = [r.text for r in retriever.retrieve(qa["question"], CFG.top_k)]
            rows["question"].append(qa["question"])
            rows["answer"].append(res["answer"])
            rows["contexts"].append(ctx)
            rows["ground_truth"].append(qa["ground_truth"])
        ds = Dataset.from_dict(rows)
        result = evaluate(
            ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        return {k: float(v) for k, v in result.items()}
    except Exception:
        return None


def groundedness_proxy(golden: list[dict]) -> float:
    grounded = 0
    for qa in golden:
        res = answer_question(qa["question"])
        grounded += int(res.get("grounded", False))
    return grounded / len(golden)


def lookup_time_benchmark(golden: list[dict], repeats: int = 3) -> dict:
    retriever = get_retriever()
    llm = get_llm()
    all_chunks = _all_context()
    full_ctx_chars = sum(len(c) for c in all_chunks)

    rag_times, base_times, rag_ctx_chars = [], [], []
    for qa in golden:
        q = qa["question"]
        topk = [r.text for r in retriever.retrieve(q, CFG.top_k)]
        rag_ctx_chars.append(sum(len(c) for c in topk))

        t0 = time.perf_counter()
        for _ in range(repeats):
            llm.complete(q, topk)
        rag_times.append((time.perf_counter() - t0) / repeats)

        t0 = time.perf_counter()
        for _ in range(repeats):
            llm.complete(q, all_chunks)
        base_times.append((time.perf_counter() - t0) / repeats)

    rag_ctx = _median(rag_ctx_chars)
    ctx_reduction = (full_ctx_chars - rag_ctx) / full_ctx_chars * 100 if full_ctx_chars else 0.0
    rag_med, base_med = _median(rag_times), _median(base_times)
    time_improvement = (base_med - rag_med) / base_med * 100 if base_med else 0.0

    return {
        "context_reduction_pct": round(ctx_reduction, 1),
        "rag_context_chars_median": int(rag_ctx),
        "full_context_chars": full_ctx_chars,
        "llm_provider": provider_status(),
        "wallclock_improvement_pct": round(time_improvement, 1),
        "rag_median_s": round(rag_med, 5),
        "full_context_median_s": round(base_med, 5),
        "methodology": "Headline = context-char reduction (top-k vs full corpus); wallclock secondary.",
    }


def _all_context() -> list[str]:
    from src.rag.ingest import build_chunks

    return [c.text for c in build_chunks()]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run() -> dict:
    golden = load_golden()
    retriever = get_retriever()
    print(f"RAG eval over {len(golden)} golden Q&A | provider={provider_status()}")

    retrieval = {
        f"recall@{CFG.top_k}": round(recall_at_k(retriever, golden, CFG.top_k), 3),
        "recall@1": round(recall_at_k(retriever, golden, 1), 3),
        f"mrr@{CFG.top_k}": round(mean_reciprocal_rank(retriever, golden, CFG.top_k), 3),
    }
    ragas = ragas_scores(golden)
    generation = ragas or {"faithfulness_proxy": round(groundedness_proxy(golden), 3),
                           "note": "RAGAS skipped (needs ds-rag env + a reachable LLM)"}
    timing = lookup_time_benchmark(golden)

    report = {"retrieval": retrieval, "generation": generation, "lookup_time": timing}
    out = CFG.artifacts / "rag_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
