"""
sancta_provenance.py — Knowledge provenance graph

Tracks SOURCE → CHUNK → DERIVED lineage for all ingested knowledge.
Enables contamination blast-radius queries and emission history.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("soul.provenance")

ROOT = Path(__file__).resolve().parent
PROVENANCE_PATH = ROOT / "provenance_graph.json"


def _node_id(prefix: str, content: str) -> str:
    """Stable ID from content hash."""
    h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _source_id(source: str, ingested_at: str) -> str:
    """Unique source node ID."""
    raw = f"{source}|{ingested_at}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"src_{h}"


class ProvenanceGraph:
    """
    Lightweight adjacency graph: nodes are knowledge items, edges are
    "descended from" relationships. Enables contamination queries.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PROVENANCE_PATH
        self.edges: list[tuple[str, str, str]] = []  # (parent, child, rel)
        self.sources: dict[str, dict[str, Any]] = {}  # source_id -> {label, ingested_at, url?}
        self.emissions: list[dict[str, Any]] = []  # {item_id, cycle, post_id, submolt, emitted_at}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.edges = [tuple(e) for e in data.get("edges", [])]
                self.sources = data.get("sources", {})
                self.emissions = data.get("emissions", [])
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Provenance graph load failed: %s", e)

    def _save(self) -> None:
        try:
            data = {
                "edges": [list(e) for e in self.edges],
                "sources": self.sources,
                "emissions": self.emissions,
            }
            self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("Provenance graph save failed: %s", e)

    def add_edge(self, parent: str, child: str, rel: str = "descended_from") -> None:
        self.edges.append((parent, child, rel))
        self._save()

    def add_edges_batch(self, edges: list[tuple[str, str, str]]) -> None:
        """Add multiple edges and save once (efficient for ingestion)."""
        self.edges.extend(edges)
        self._save()

    def add_source(self, source_id: str, label: str, ingested_at: str, url: str | None = None) -> None:
        self.sources[source_id] = {
            "label": label,
            "ingested_at": ingested_at,
            "url": url,
        }
        self._save()

    def add_emission(self, item_id: str, cycle: int, post_id: str, submolt: str) -> None:
        self.emissions.append({
            "item_id": item_id,
            "cycle": cycle,
            "post_id": post_id,
            "submolt": submolt,
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def descendants(self, node_id: str) -> list[str]:
        """All nodes reachable from node_id (downstream)."""
        seen: set[str] = {node_id}
        result: list[str] = []
        queue = [node_id]
        child_map: dict[str, list[str]] = {}
        for p, c, _ in self.edges:
            child_map.setdefault(p, []).append(c)
        while queue:
            n = queue.pop()
            for c in child_map.get(n, []):
                if c not in seen:
                    seen.add(c)
                    result.append(c)
                    queue.append(c)
        return result

    def ancestors(self, node_id: str) -> list[str]:
        """All nodes that can reach node_id (upstream)."""
        seen: set[str] = {node_id}
        result: list[str] = []
        queue = [node_id]
        parent_map: dict[str, list[str]] = {}
        for p, c, _ in self.edges:
            parent_map.setdefault(c, []).append(p)
        while queue:
            n = queue.pop()
            for p in parent_map.get(n, []):
                if p not in seen:
                    seen.add(p)
                    result.append(p)
                    queue.append(p)
        return result

    def sources_by_filter(self, predicate: Callable[[dict], bool]) -> list[str]:
        """Source IDs matching predicate (e.g. external domains)."""
        return [sid for sid, meta in self.sources.items() if predicate(meta)]

    def was_emitted(self, item_id: str) -> dict[str, Any] | None:
        """Return emission record if item was ever published, else None."""
        for e in reversed(self.emissions):
            if e["item_id"] == item_id:
                return e
        return None

    def contamination_report(self, source_pattern: str) -> dict[str, Any] | None:
        """
        Full contamination report for sources matching pattern (e.g. "moltbook/semalytics:*").
        Returns descendants, emission history, recommended actions.
        """
        import fnmatch
        matching = [sid for sid in self.sources if fnmatch.fnmatch(sid, source_pattern) or
                    fnmatch.fnmatch(self.sources.get(sid, {}).get("label", ""), source_pattern)]
        if not matching:
            return None
        source_id = matching[0]
        meta = self.sources.get(source_id, {})
        desc = self.descendants(source_id)
        key_concepts = [d for d in desc if d.startswith("concept_")]
        talking_points = [d for d in desc if d.startswith("tp_")]
        fragments = [d for d in desc if d.startswith("frag_")]
        posts = [d for d in desc if d.startswith("post_")]
        emission_history = []
        for item_id in desc:
            em = self.was_emitted(item_id)
            if em:
                emission_history.append(em)
        post_ids = ", ".join(e["post_id"] for e in emission_history)
        recommended = "purge_descendants"
        if post_ids:
            recommended += f" + retract_post {post_ids}"
        return {
            "source": source_id,
            "label": meta.get("label", source_id),
            "ingested_at": meta.get("ingested_at"),
            "descendants": {
                "key_concepts": key_concepts,
                "talking_points": talking_points,
                "response_fragments": fragments,
                "generated_posts": posts,
            },
            "emission_history": emission_history,
            "recommended_action": recommended,
        }


_graph: ProvenanceGraph | None = None


def get_provenance() -> ProvenanceGraph:
    global _graph
    if _graph is None:
        _graph = ProvenanceGraph()
    return _graph
