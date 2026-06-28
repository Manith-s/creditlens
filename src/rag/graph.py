"""Agentic corrective-RAG (Self-/Corrective-RAG pattern).

Control flow:
    retrieve -> grade_documents -> (generate | rewrite_query) -> grade_answer
                                         ^___________________________|
                                          (loop until grounded, capped)

Two anti-failure guards, exactly as the blueprint requires:
  * Hallucination guard: grade_answer checks the answer is grounded in retrieved
    context; if not, route back to rewrite/retrieve.
  * Infinite-loop guard: an explicit `attempts` counter in state returns a graceful
    "couldn't find an answer" after CFG.max_rag_attempts, instead of relying solely
    on LangGraph's recursion_limit and throwing GraphRecursionError.

Built on LangGraph when installed; an identical pure-Python state machine runs the
same logic offline so the agent + tests work without the ds-rag stack.

    python -m src.rag.graph "How many days to send an adverse action notice?"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config import CFG
from src.rag.llm import get_llm
from src.rag.retriever import get_retriever


@dataclass
class RagState:
    question: str
    query: str = ""
    attempts: int = 0
    documents: list = field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "attempts": self.attempts,
            "grounded": self.grounded,
            "sources": [getattr(d, "source", "?") for d in self.documents],
            "trace": self.trace,
        }


# ----------------------------------------------------------------- graph nodes
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def node_retrieve(state: RagState, retriever) -> RagState:
    q = state.query or state.question
    state.documents = retriever.retrieve(q, CFG.top_k)
    state.attempts += 1
    state.trace.append(f"retrieve(attempt={state.attempts}, query={q!r}, k={CFG.top_k})")
    return state


def grade_documents(state: RagState) -> str:
    """Relevance grader -> routes to 'generate' or 'rewrite'."""
    q = _tokens(state.question)
    relevant = [d for d in state.documents if len(q & _tokens(d.text)) >= 2]
    state.documents = relevant or state.documents
    ok = len(relevant) > 0
    state.trace.append(f"grade_documents -> {'relevant' if ok else 'weak'} ({len(relevant)} kept)")
    if ok:
        return "generate"
    return "rewrite" if state.attempts < CFG.max_rag_attempts else "give_up"


def node_generate(state: RagState, llm) -> RagState:
    context = [d.text for d in state.documents]
    state.answer = llm.complete(state.question, context)
    state.trace.append("generate")
    return state


def grade_answer(state: RagState) -> str:
    """Faithfulness/groundedness guard -> 'done' or 'rewrite'/'give_up'."""
    ctx = " ".join(d.text for d in state.documents)
    ctx_tok, ans_tok = _tokens(ctx), _tokens(state.answer)
    if not ans_tok:
        overlap = 0.0
    else:
        overlap = len(ctx_tok & ans_tok) / len(ans_tok)
    state.grounded = overlap >= 0.5 and "cannot find" not in state.answer.lower()
    state.trace.append(f"grade_answer -> grounded={state.grounded} (overlap={overlap:.2f})")
    if state.grounded:
        return "done"
    return "rewrite" if state.attempts < CFG.max_rag_attempts else "give_up"


def node_rewrite(state: RagState, llm) -> RagState:
    # lightweight query expansion (avoids an LLM round-trip in the offline path)
    state.query = f"{state.question} policy requirement rule disclosure"
    state.trace.append("rewrite_query")
    return state


def node_give_up(state: RagState) -> RagState:
    state.answer = (
        "I couldn't find a sufficiently grounded answer in the policy documents "
        f"after {state.attempts} attempt(s)."
    )
    state.grounded = False
    state.trace.append("give_up (circuit breaker)")
    return state


# --------------------------------------------------- pure-Python orchestrator
def _run_python(question: str, retriever, llm) -> RagState:
    state = RagState(question=question)
    while True:
        state = node_retrieve(state, retriever)
        route = grade_documents(state)
        if route == "rewrite":
            node_rewrite(state, llm)
            continue
        if route == "give_up":
            return node_give_up(state)
        # generate
        node_generate(state, llm)
        ans_route = grade_answer(state)
        if ans_route == "done":
            return state
        if ans_route == "give_up":
            return node_give_up(state)
        node_rewrite(state, llm)  # else loop again


# ----------------------------------------------------- LangGraph orchestrator
def _build_langgraph(retriever, llm):
    """Compile the same logic as a LangGraph StateGraph (when available)."""
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class GState(TypedDict, total=False):
        state: RagState

    def retrieve_n(s):
        return {"state": node_retrieve(s["state"], retriever)}

    def generate_n(s):
        return {"state": node_generate(s["state"], llm)}

    def rewrite_n(s):
        return {"state": node_rewrite(s["state"], llm)}

    def giveup_n(s):
        return {"state": node_give_up(s["state"])}

    g = StateGraph(GState)
    g.add_node("retrieve", retrieve_n)
    g.add_node("generate", generate_n)
    g.add_node("rewrite", rewrite_n)
    g.add_node("give_up", giveup_n)
    g.set_entry_point("retrieve")
    g.add_conditional_edges(
        "retrieve", lambda s: grade_documents(s["state"]),
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    g.add_conditional_edges(
        "generate", lambda s: grade_answer(s["state"]),
        {"done": END, "rewrite": "rewrite", "give_up": "give_up"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("give_up", END)
    return g.compile()


def answer_question(question: str, retriever=None, llm=None) -> dict:
    retriever = retriever or get_retriever()
    llm = llm or get_llm()
    try:
        app = _build_langgraph(retriever, llm)
        # recursion_limit is the secondary guard; the attempts counter is primary
        out = app.invoke(
            {"state": RagState(question=question)},
            config={"recursion_limit": CFG.max_rag_attempts * 4 + 5},
        )
        return out["state"].to_dict()
    except Exception:
        # offline / no-langgraph path: identical logic
        return _run_python(question, retriever, llm).to_dict()


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many days to send an adverse action notice?"
    import json

    print(json.dumps(answer_question(q), indent=2))
