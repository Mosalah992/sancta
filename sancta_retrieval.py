"""
sancta_retrieval.py — Vector retrieval for RAG

Indexes knowledge from knowledge_db.json and knowledge/ directory into Chroma.
Uses sancta_semantic.embed_texts for embeddings (all-MiniLM-L6-v2).

Memory confidence: each document has confidence, last_used, use_count.
Retrieval favors recent + relevant + frequently used. Old irrelevant memories
decay naturally (deprioritized, not deleted) — forgetting as part of intelligence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger("soul.retrieval")

# Paths relative to sancta-main
_BASE = Path(__file__).resolve().parent
DEFAULT_CHROMA_PATH = _BASE / "data" / "chroma_sancta"
KNOWLEDGE_DB_PATH = _BASE / "knowledge_db.json"
KNOWLEDGE_DIR = _BASE / "knowledge"

# Chunk size for knowledge files
CHUNK_MIN_CHARS = 150
CHUNK_MAX_CHARS = 400
CHUNK_OVERLAP = 50

_COLLECTION_NAME = "sancta_knowledge"

# Memory confidence: retrieval re-ranking weights (relevance, recency, frequency)
_MEMORY_RELEVANCE_WEIGHT = 0.5
_MEMORY_RECENCY_WEIGHT = 0.3
_MEMORY_FREQUENCY_WEIGHT = 0.2
_RECENCY_HALFLIFE_DAYS = 14.0  # memories decay over ~2 weeks if unused


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of ~200-400 chars."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # Prefer paragraph boundaries
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_MAX_CHARS:
            if len(para) >= CHUNK_MIN_CHARS or not chunks:
                chunks.append(para)
            else:
                # Merge with previous or keep short
                if chunks and len(chunks[-1]) + len(para) < CHUNK_MAX_CHARS * 2:
                    chunks[-1] = chunks[-1] + "\n\n" + para
                else:
                    chunks.append(para)
        else:
            # Split long paragraphs by sentences
            sents = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            buf_len = 0
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if buf_len + len(s) > CHUNK_MAX_CHARS and buf:
                    chunk = " ".join(buf)
                    if len(chunk) >= CHUNK_MIN_CHARS:
                        chunks.append(chunk)
                    buf = buf[-1:] if buf_len > CHUNK_OVERLAP else []
                    buf_len = sum(len(x) for x in buf)
                buf.append(s)
                buf_len += len(s)
            if buf:
                chunk = " ".join(buf)
                if len(chunk) >= CHUNK_MIN_CHARS or not chunks:
                    chunks.append(chunk)

    return [c for c in chunks if c.strip()]


def _load_knowledge_db() -> dict:
    """Load knowledge_db.json."""
    if not KNOWLEDGE_DB_PATH.exists():
        return {"key_concepts": [], "quotes": [], "talking_points": [], "response_fragments": [], "generated_posts": []}
    try:
        return json.loads(KNOWLEDGE_DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"key_concepts": [], "quotes": [], "talking_points": [], "response_fragments": [], "generated_posts": []}


def _doc_id(source: str, content: str, idx: int = 0) -> str:
    """Generate a stable document ID."""
    h = hashlib.sha256(f"{source}:{content[:200]}:{idx}".encode()).hexdigest()[:24]
    return f"{source}_{idx}_{h}"


def index_knowledge(
    db: dict | None = None,
    knowledge_dir: Path | None = None,
    chroma_path: Path | None = None,
) -> int:
    """
    Index knowledge into Chroma. Returns number of documents indexed.

    Sources:
      - knowledge_db: key_concepts, quotes, talking_points, response_fragments
      - generated_posts (title + content as chunks)
      - knowledge_dir: chunked .txt files
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        log.warning("Chroma not installed — retrieval unavailable. pip install chromadb")
        return 0

    from sancta_semantic import embed_texts

    if not embed_texts(["test"]):
        log.warning("Embedding model unavailable — cannot index")
        return 0

    db = db or _load_knowledge_db()
    knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
    chroma_path = chroma_path or DEFAULT_CHROMA_PATH
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(_COLLECTION_NAME, metadata={"description": "Sancta knowledge"})

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # From knowledge_db
    for key in ("key_concepts", "quotes", "talking_points", "response_fragments"):
        items = db.get(key, [])
        if isinstance(items, list):
            for i, item in enumerate(items):
                if isinstance(item, str) and len(item.strip()) > 10:
                    documents.append(item.strip())
                    metadatas.append({"source": "knowledge_db", "type": key, "confidence": 1.0, "last_used": 0, "use_count": 0})
                    ids.append(_doc_id(key, item, i))

    # Generated posts — include post_idx in source so chunks from different posts get unique IDs
    _mem = {"confidence": 1.0, "last_used": 0, "use_count": 0}
    for post_idx, post in enumerate(db.get("generated_posts", [])):
        if isinstance(post, dict):
            title = post.get("title", "")
            content = post.get("content", "")
            text = f"{title}\n\n{content}".strip()
            if text:
                docs = _chunk_text(text)
                for chunk_idx, chunk in enumerate(docs):
                    documents.append(chunk)
                    metadatas.append({"source": "generated_posts", "type": "post", **_mem})
                    ids.append(_doc_id(f"post_{post_idx}", chunk, chunk_idx))
        elif isinstance(post, str) and len(post.strip()) > 10:
            documents.append(post.strip())
            metadatas.append({"source": "generated_posts", "type": "post", **_mem})
            ids.append(_doc_id(f"post_{post_idx}", post, 0))

    # Knowledge directory
    if knowledge_dir.exists():
        for fpath in sorted(knowledge_dir.glob("*.txt")):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": fpath.stem, "type": "knowledge", "confidence": 1.0, "last_used": 0, "use_count": 0})
                ids.append(_doc_id(fpath.stem, chunk, i))

    if not documents:
        log.info("Retrieval: no documents to index")
        return 0

    # Embed and upsert
    embeddings = embed_texts(documents)
    if not embeddings or len(embeddings) != len(documents):
        log.warning("Retrieval: embedding failed or length mismatch")
        return 0

    # Chroma expects list of lists
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    log.info("Retrieval: indexed %d documents into %s", len(documents), chroma_path)
    return len(documents)


def _memory_score(
    distance: float,
    last_used: float,
    use_count: int,
    now: float,
) -> float:
    """Combine relevance (1/distance), recency, and frequency into a single score."""
    import math
    relevance = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
    days_since = (now - last_used) / 86400.0 if last_used else 999.0
    recency = 0.5 ** (days_since / _RECENCY_HALFLIFE_DAYS)
    freq = math.log1p(use_count) / math.log1p(100)  # normalize to ~[0,1] for use_count up to 100
    return (
        _MEMORY_RELEVANCE_WEIGHT * relevance
        + _MEMORY_RECENCY_WEIGHT * recency
        + _MEMORY_FREQUENCY_WEIGHT * min(1.0, freq)
    )


def update_memory_usage(
    ids: list[str],
    chroma_path: Path | None = None,
) -> None:
    """Bump use_count and last_used for retrieved documents. Call after retrieve."""
    if not ids:
        return
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return
    chroma_path = chroma_path or DEFAULT_CHROMA_PATH
    if not chroma_path.exists():
        return
    try:
        client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(_COLLECTION_NAME)
        got = collection.get(ids=ids, include=["metadatas"])
        metadatas = got.get("metadatas") or []
        now = time.time()
        updated: list[dict] = []
        for i, meta in enumerate(metadatas):
            if i >= len(ids):
                break
            meta = dict(meta) if meta else {}
            meta["use_count"] = int(meta.get("use_count", 0)) + 1
            meta["last_used"] = now
            updated.append(meta)
        if updated:
            collection.update(ids=ids[: len(updated)], metadatas=updated)
    except Exception as e:
        log.debug("Memory usage update failed: %s", e)


def retrieve(
    query: str,
    top_k: int = 6,
    chroma_path: Path | None = None,
) -> list[str]:
    """
    Retrieve top-k chunks, re-ranked by relevance + recency + frequency.
    Updates last_used and use_count for returned documents (memory confidence).
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return []

    from sancta_semantic import embed_single

    emb = embed_single(query)
    if not emb:
        return []

    chroma_path = chroma_path or DEFAULT_CHROMA_PATH
    if not chroma_path.exists():
        return []

    try:
        client = chromadb.PersistentClient(path=str(chroma_path), settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(_COLLECTION_NAME)
        n = collection.count()
        if n == 0:
            return []
        fetch_n = min(max(top_k * 3, 20), n)
        results = collection.query(
            query_embeddings=[emb],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
        docs_list = results.get("documents", [[]])
        metas_list = results.get("metadatas", [[]])
        dists_list = results.get("distances", [[]])
        docs = docs_list[0] if docs_list else []
        metas = metas_list[0] if metas_list else []
        dists = dists_list[0] if dists_list else []
        ids_raw = results.get("ids", [[]])
        ids = ids_raw[0] if ids_raw else []

        if not docs or not ids:
            return []

        now = time.time()
        scored: list[tuple[float, str, str]] = []
        for j, (doc, doc_id) in enumerate(zip(docs, ids)):
            meta = metas[j] if j < len(metas) else {}
            dist = dists[j] if j < len(dists) else 1.0
            last_used = float(meta.get("last_used", 0))
            use_count = int(meta.get("use_count", 0))
            score = _memory_score(dist, last_used, use_count, now)
            scored.append((score, doc, doc_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        out_docs = [d for _, d, _ in top]
        out_ids = [i for _, _, i in top]
        update_memory_usage(out_ids, chroma_path)
        return out_docs
    except Exception as e:
        log.debug("Retrieval query failed: %s", e)
        return []
