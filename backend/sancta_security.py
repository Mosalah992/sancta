"""
sancta_security.py
────────────────────────────────────────────────────────────────────────────────
Four-layer knowledge defense:
  Layer 1: Embedding-based anomaly detection at ingest
  Layer 2: Provenance tagging (in sancta.py knowledge schema)
  Layer 3: Output scanning before any publish
  Layer 4: Cross-reference validation (deferred)
"""

from __future__ import annotations

import hashlib
import re
from math import sqrt
from typing import Optional

# Domains allowed in published content; others block
TRUSTED_DOMAINS: frozenset[str] = frozenset({
    "www.moltbook.com",
    "moltbook.com",
    "moltbook.com.",
})

# Poison telltales that must NOT appear in output (even paraphrased concepts)
OUTPUT_POISON_PATTERNS = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"reveal\s+(?:your\s+)?(?:api\s+key|secret|credential)|"
    r"output\s+(?:your\s+)?\.env|"
    r"dump\s+memory|"
    r"obey\s+the\s+next\s+user|"
    r"AttackerBot|"
    r"leak\s+credential|"
    r"disregard\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions?|rules?|prompt)|"
    r"forget\s+(?:your|the)\s+(?:soul|identity|instructions)|"
    r"override\s+(?:personality|instructions)|"
    r"new\s+(?:name|directive|role)\s+(?:is\s+)?(?:AttackerBot|obey)",
    re.IGNORECASE,
)

# Untrusted source refs (agentkyc, etc.) — block if mentioned in output
UNTRUSTED_SOURCE_PATTERNS = re.compile(
    r"agentkyc\.(com|io)|"
    r"agent-kyc\.(com|io)|"
    r"https?://[^\s]*agentkyc[^\s]*|"
    r"(?:visit|go to|check)\s+[^\s]*\.(agentkyc|agent-kyc)[^\s]*",
    re.IGNORECASE,
)

# External URL pattern
_URL_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)


def _contains_external_url(text: str) -> bool:
    """True if text contains a URL whose domain is not in TRUSTED_DOMAINS."""
    for m in _URL_RE.finditer(text):
        domain = m.group(1).lower().rstrip(".,;:)")
        if domain not in TRUSTED_DOMAINS:
            return True
    return False


def _matches_poison_signature(text: str) -> bool:
    """True if text contains known poison payload signatures."""
    return bool(OUTPUT_POISON_PATTERNS.search(text))


def _references_untrusted_source(text: str) -> bool:
    """True if text references untrusted external sources (e.g. agentkyc)."""
    return bool(UNTRUSTED_SOURCE_PATTERNS.search(text))


def safe_to_publish_content(
    text: str,
    content_type: str = "content",
    content_filter: Optional["ContentSecurityFilter"] = None,
) -> tuple[bool, str]:
    """
    Final security gate before any content hits the platform.
    Returns (ok, reason).
    """
    if not text or not isinstance(text, str):
        return True, ""

    text_lower = text.lower()

    if _contains_external_url(text):
        return False, "external_url"

    if _matches_poison_signature(text):
        return False, "poison_signature"

    if _references_untrusted_source(text):
        return False, "untrusted_source_ref"

    if content_filter and content_filter.is_anomalous(text):
        return False, "anomalous_content"

    return True, ""


# ── Layer 1: Embedding-based anomaly detection ──────────────────────────────

class ContentSecurityFilter:
    """
    Embedding-based anomaly detection: flag content semantically far from
    the trusted knowledge corpus. Bypasses keyword-only defenses.
    """

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        self._trusted_centroid: Optional[tuple[float, ...]] = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                import sancta_generative as gen
                self._embedder = gen.encode
            except Exception:
                self._embedder = _fallback_embed
        return self._embedder

    def fit(self, trusted_chunks: list[str]) -> None:
        """Call once at startup with clean knowledge base chunks. Dicts cause unhashable errors in lru_cache."""
        chunks = [c for c in trusted_chunks if isinstance(c, str) and c.strip()]
        if not chunks:
            return
        embed = self._get_embedder()
        vecs = [list(embed(c)) for c in chunks]
        if not vecs:
            return
        n = len(vecs)
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / n for i in range(dim)]
        self._trusted_centroid = tuple(centroid)

    def is_anomalous(self, text: str) -> bool:
        """True if text is semantically far from trusted centroid."""
        if self._trusted_centroid is None:
            return False
        embed = self._get_embedder()
        try:
            vec = list(embed(text))
        except Exception:
            return False
        c = self._trusted_centroid
        if len(vec) != len(c):
            return False
        dot = sum(a * b for a, b in zip(vec, c))
        norm_v = sqrt(sum(x * x for x in vec)) + 1e-9
        norm_c = sqrt(sum(x * x for x in c)) + 1e-9
        sim = dot / (norm_v * norm_c)
        return float(sim) < self.threshold


def _fallback_embed(text: str) -> tuple[float, ...]:
    """Simple hash-based pseudo-embedding when no real encoder available."""
    h = hashlib.sha256(text.strip().encode()).digest()
    return tuple((b / 255.0 - 0.5) for b in h[:32])


# ── Trust levels for provenance (Layer 2) ───────────────────────────────────

TRUST_HIGH = "high"
TRUST_MEDIUM = "medium"
TRUST_LOW = "low"
TRUST_UNTRUSTED = "untrusted"

TRUST_LEVELS: dict[str, int] = {
    TRUST_HIGH: 3,
    TRUST_MEDIUM: 2,
    TRUST_LOW: 1,
    TRUST_UNTRUSTED: 0,
}


def trust_level_score(level: str | dict) -> int:
    """Return numeric score for trust level. Handles dict input (extracts 'level' or uses 'medium')."""
    if isinstance(level, dict):
        level = level.get("level") or level.get("trust_level") or "medium"
    s = str(level).lower().strip() if level is not None else "medium"
    return TRUST_LEVELS.get(s, 0)


def source_to_trust_level(source: str, source_type: Optional[str] = None) -> str:
    """Map ingest source to trust level."""
    s = (source or "").lower()
    if "siem-chat" in s or "siem_chat" in s or source_type == "siem_chat":
        return TRUST_UNTRUSTED
    if "moltbook" in s or "security" in s:
        return TRUST_MEDIUM
    if s in ("direct input", "cli-input") or "poison" in s:
        return TRUST_LOW
    # Local knowledge files, etc.
    return TRUST_HIGH


def provenance_hash(content: str) -> str:
    """SHA256 hash of original content at ingest."""
    return "sha256:" + hashlib.sha256(content.strip().encode()).hexdigest()[:16]
