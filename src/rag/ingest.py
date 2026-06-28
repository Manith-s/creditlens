"""Ingest policy documents -> chunks -> embeddings -> vector store.

Primary path: LlamaIndex readers + Ollama `nomic-embed-text` + ChromaDB persistent
collection (the production pattern). Offline path: a self-contained chunk store
(JSON) queried by cosine over HashingEmbedder or TF-IDF -- so retrieval and the
Recall@k/MRR evaluation run with zero external services.

    python -m src.rag.ingest          # build the index
    python -m src.rag.ingest --stats  # show what's indexed
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

from src.config import CFG
from src.rag.llm import get_embedder, provider_status


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # filename, used as the retrieval ground-truth label


# --------------------------------------------------------------------- chunking
def _read_documents() -> list[tuple[str, str]]:
    """Return [(filename, text)] from the policy corpus (markdown + PDF)."""
    docs: list[tuple[str, str]] = []
    corpus = CFG.policy_corpus
    for p in sorted(corpus.glob("*.md")) + sorted(corpus.glob("*.txt")):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    for p in sorted(corpus.glob("*.pdf")):
        try:
            from pypdf import PdfReader

            text = "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)
            docs.append((p.name, text))
        except Exception:  # noqa: BLE001 -- skip unreadable PDFs, keep the rest
            continue
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character chunking with overlap, snapped to paragraph/sentence boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # snap to a sentence/paragraph boundary (only when not at the very end)
        if end < len(text):
            window = text[start:end]
            m = list(re.finditer(r"[.!?]\s|\n\n", window))
            if m:
                end = start + m[-1].end()
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break  # reached the end; do NOT keep emitting tiny overlap slivers
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for fname, text in _read_documents():
        for i, ck in enumerate(chunk_text(text, CFG.chunk_size, CFG.chunk_overlap)):
            chunks.append(Chunk(id=f"{fname}::{i}", text=ck, source=fname))
    return chunks


# ------------------------------------------------------------------- index build
def ingest(use_chroma: bool | None = None) -> dict:
    """Build the index. Returns a small stats dict. Writes either a Chroma
    collection (if available) or an offline chunk store JSON."""
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError(
            "No policy documents found. Run `python -m src.data.make_policy_corpus` "
            "or `python -m src.data.download --policy --run`."
        )

    chroma_ok = _try_chroma(chunks) if use_chroma in (None, True) else False
    if not chroma_ok:
        _write_offline_store(chunks)

    stats = {
        "n_chunks": len(chunks),
        "n_sources": len({c.source for c in chunks}),
        "backend": "chromadb" if chroma_ok else "offline-json",
        "embedder": provider_status(),
    }
    print(json.dumps(stats, indent=2))
    return stats


def _try_chroma(chunks: list[Chunk]) -> bool:
    try:
        import chromadb
    except ImportError:
        return False
    try:
        client = chromadb.PersistentClient(path=str(CFG.chroma_dir))
        # fresh collection each ingest (Chroma migrations are irreversible; this
        # avoids dimension clashes when the embedder changes)
        try:
            client.delete_collection(CFG.chroma_collection)
        except Exception:
            pass
        col = client.create_collection(CFG.chroma_collection, metadata={"hnsw:space": "cosine"})
        embedder = get_embedder()
        embs = embedder.embed([c.text for c in chunks])
        col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embs,
            metadatas=[{"source": c.source} for c in chunks],
        )
        return True
    except Exception:  # noqa: BLE001 -- fall back to offline store
        return False


def _offline_store_path():
    return CFG.chroma_dir / "offline_chunks.json"


def _write_offline_store(chunks: list[Chunk]) -> None:
    _offline_store_path().write_text(
        json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8"
    )


def load_offline_chunks() -> list[Chunk]:
    path = _offline_store_path()
    if not path.exists():
        return []
    return [Chunk(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-chroma", action="store_true", help="force the offline store")
    args = ap.parse_args()
    if args.stats:
        chunks = build_chunks()
        print(json.dumps(
            {"n_chunks": len(chunks), "n_sources": len({c.source for c in chunks})}, indent=2
        ))
        return
    ingest(use_chroma=not args.no_chroma)


if __name__ == "__main__":
    main()
