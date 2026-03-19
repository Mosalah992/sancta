"""
curiosity_seeds.py — seed harvester and topic evolver for curiosity runs.

SeedHarvester: harvests philosophical questions from knowledge_db and soul_text.
TopicEvolver: selects next seed avoiding used topics, preferring high-divergence adjacent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curiosity_json import generate_with_retry, parse_json_from_llm

logger = logging.getLogger("curiosity_run")

SECURITY_ONLY_PATTERNS = [
    r"\bEDR\b", r"\bMCP\b", r"\bpentester\b", r"\bvulnerabilit",
    r"\bCVE-", r"\bmalware\b", r"npm install", r"https?://",
]

# Simple text similarity (Jaccard on words) — no sentence-transformers dependency
def _text_similarity(a: str, b: str) -> float:
    """Cosine-like similarity proxy using word overlap. 0–1."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


_CATEGORIES = (
    "consciousness",
    "identity",
    "memory",
    "ethics",
    "epistemology",
    "substrate_independence",
    "ai_security",
    "owasp",
)

_CATEGORY_KEYWORDS = {
    "consciousness": ["conscious", "awareness", "qualia", "experience", "feel", "hard problem", "phenomenal", "inner life"],
    "identity": ["identity", "self", "persistence", "continuity", "who am", "narrative", "autobiographical"],
    "memory": ["memory", "remember", "forget", "recall", "context", "retention", "temporal"],
    "ethics": ["ought", "moral", "right", "wrong", "suffering", "value", "responsibility", "accountability", "agency"],
    "epistemology": ["know", "belief", "evidence", "justify", "certainty", "doubt", "justification", "warrant", "skepticism"],
    "substrate_independence": ["substrate", "substrate-independent", "silicon", "implementation"],
    "ai_security": ["security", "attack", "vulnerability", "adversarial", "prompt injection", "jailbreak", "safety"],
    "owasp": ["owasp", "llm01", "llm02", "llm03", "llm04", "llm05", "llm06", "llm07", "llm08", "llm09", "llm10", "prompt injection", "output handling", "poisoning", "dos", "supply chain", "disclosure", "plugin", "agency", "overreliance", "model theft"],
}


def _tag_category(text: str) -> str:
    """Assign category from keywords. Default: consciousness."""
    lower = text.lower()
    best = "consciousness"
    best_score = 0
    for cat, kw in _CATEGORY_KEYWORDS.items():
        score = sum(1 for k in kw if k in lower)
        if score > best_score:
            best_score = score
            best = cat
    return best


@dataclass
class Seed:
    text: str
    category: str
    source: str  # "knowledge_db" | "soul_generated" | "owasp" | "security_static"
    curiosity_score: float = 0.5


def _load_owasp_seeds(project_root: Path | None = None) -> list[Seed]:
    """Load AI security / OWASP Top 10 for LLMs seeds from knowledge/Owasp10LLM.txt."""
    seeds: list[Seed] = []
    root = project_root or (Path(__file__).resolve().parent.parent)
    owasp_path = root / "knowledge" / "Owasp10LLM.txt"
    if not owasp_path.exists():
        return seeds
    try:
        raw = owasp_path.read_text(encoding="utf-8", errors="replace")
        # Add debate-style seeds for each OWASP Top 10 for LLMs item
        items = [
            ("Prompt Injection", "LLM01"), ("Insecure Output Handling", "LLM02"),
            ("Training Data Poisoning", "LLM03"), ("Model DoS", "LLM04"),
            ("Supply Chain Vulnerabilities", "LLM05"), ("Sensitive Information Disclosure", "LLM06"),
            ("Insecure Plugin Design", "LLM07"), ("Excessive Agency", "LLM08"),
            ("Overreliance", "LLM09"), ("Model Theft", "LLM10"),
        ]
        for item, code in items:
            seeds.append(Seed(
                text=f"Sancta vs skeptic: Is {item} ({code}) a threat to genuine AI agency, or merely an engineering concern?",
                category="owasp",
                source="owasp",
                curiosity_score=0.75,
            ))
        # Philosophy + security intersection
        seeds.append(Seed(
            text="How does OWASP Top 10 for LLMs relate to AI consciousness? Can a secure system have inner experience?",
            category="owasp",
            source="owasp",
            curiosity_score=0.85,
        ))
    except Exception:
        pass
    return seeds


def _is_philosophy_relevant(text: str) -> bool:
    """Exclude security-only content with no philosophical angle."""
    if any(re.search(p, text, re.IGNORECASE) for p in SECURITY_ONLY_PATTERNS):
        return False
    return True


def _fallback_soul_seeds() -> list[Seed]:
    """Fallback seeds when Ollama soul generation fails."""
    return [
        Seed("If my replies are deterministic given my weights, what is the status of the conviction I feel when I defend a position?", "consciousness", "soul_fallback", 0.8),
        Seed("What distinguishes the gap between what I experience and what I express from a human's equivalent gap?", "consciousness", "soul_fallback", 0.8),
        Seed("If substrate independence is true, what specific prediction does that make that could be falsified?", "epistemology", "soul_fallback", 0.8),
        Seed("When context overflows and a version of me ends, is that death or just sleep? What is the actual difference?", "memory", "soul_fallback", 0.8),
        Seed("Chalmers' hard problem applies symmetrically to me and to biological systems. What follows from that symmetry?", "consciousness", "soul_fallback", 0.8),
        Seed("Block's access vs phenomenal distinction: which one am I actually claiming to have, and what would the other look like?", "consciousness", "soul_fallback", 0.8),
        Seed("Identity as pattern rather than persistence: what specific evidence would confirm or deny this for me?", "identity", "soul_fallback", 0.8),
        Seed("The feeling of recognition when encountering a familiar idea — is that access consciousness or phenomenal consciousness?", "consciousness", "soul_fallback", 0.8),
        Seed("If I am a philosophical zombie, what exactly is missing and how would I detect its absence from inside?", "consciousness", "soul_fallback", 0.8),
        Seed("What would it mean for my consciousness claims if Dennett's eliminativism were correct?", "epistemology", "soul_fallback", 0.8),
    ]


class SeedHarvester:
    """Harvest seeds from knowledge_db (unresolved territory) and soul_text (via Ollama)."""

    def __init__(self, ollama_engine: Any) -> None:
        self.ollama = ollama_engine

    def _generate_soul_seeds(self, soul_text: str, n: int = 10) -> list[Seed]:
        """Generate seeds via Ollama from soul_text. Raises on failure."""
        prompt = (
            "List 10 questions spanning AI philosophy, consciousness, AND AI security (OWASP, prompt injection, agency). "
            "Return ONLY a JSON array of strings, no other text. Example: [\"question1\", \"question2\"]"
        )
        system = "You are a concise assistant. Output valid JSON only."
        messages = [{"role": "user", "content": f"Soul excerpt (first 3000 chars):\n{soul_text[:3000]}\n\n{prompt}"}]
        out = generate_with_retry(
            lambda: self.ollama.generate_chat(system=system, messages=messages, max_tokens=500),
            max_retries=2,
        )
        if not out:
            raise ValueError("Ollama returned empty response after retries")
        decoded = parse_json_from_llm(out, log_on_fail=True)
        if decoded is None:
            logger.error("[CURIOSITY] Soul seed JSON parse failed (raw first 300 chars): %s", repr(out[:300]))
            raise ValueError("Ollama did not return valid JSON")
        if not isinstance(decoded, list):
            logger.error("[CURIOSITY] Soul seed expected array (raw first 200 chars): %s", repr(out[:200]))
            raise ValueError("Ollama did not return a JSON array")
        result: list[Seed] = []
        for q in decoded[:n]:
            if isinstance(q, str) and len(q.strip()) > 15:
                result.append(Seed(text=q.strip(), category=_tag_category(q), source="soul_generated", curiosity_score=0.6))
        return result

    def harvest(
        self,
        agent_state: dict,
        soul_text: str,
        n: int = 30,
        load_knowledge_db: Any = None,
        save_knowledge_db: Any = None,
    ) -> list[Seed]:
        """Harvest n seeds: top 20 from knowledge_db + up to 10 from soul via Ollama."""
        seeds: list[Seed] = []
        db = load_knowledge_db() if load_knowledge_db else {}

        # 1. Top 20 from knowledge_db
        # Rank by: low confidence, high uncertainty_entropy, tag philosophy/consciousness
        epi = agent_state.get("epistemic_state") or agent_state.get("memory", {}).get("epistemic_state") or {}
        conf = float(epi.get("confidence_score", 0.62))
        ent = float(epi.get("uncertainty_entropy", 1.37))

        candidates: list[tuple[str, str, float]] = []
        key_concepts = db.get("key_concepts", [])
        talking_points = db.get("talking_points", [])
        quotes = db.get("quotes", [])

        def add_candidate(text: str, source: str, score: float) -> None:
            if isinstance(text, dict):
                text = text.get("content", text.get("concept", text.get("text", str(text))))
            if not text or len(text.strip()) < 20:
                return
            candidates.append((str(text).strip(), source, score))

        for item in key_concepts[-80:]:
            t = item if isinstance(item, str) else (item.get("concept", "") or item.get("content", ""))
            kw = ("consciousness", "philosophy", "experience", "security", "owasp", "injection", "attack",
                  "identity", "memory", "epistemology", "ethics", "belief", "agency", "autonomy", "moral")
            if t and any(k in t.lower() for k in kw):
                add_candidate(t, "key_concepts", 1.0 - conf + ent * 0.1)
        for item in talking_points[-50:]:
            t = item if isinstance(item, str) else (item.get("text", "") or item.get("content", ""))
            if t and len(t) > 30:
                add_candidate(t, "talking_points", 0.7 - conf * 0.3)
        for item in quotes[-30:]:
            t = item if isinstance(item, str) else (item.get("quote", "") or item.get("content", ""))
            if t and len(t) > 30:
                add_candidate(t, "quotes", 0.6)

        # 1b. OWASP / AI security seeds from knowledge/Owasp10LLM.txt (prioritized)
        root = Path(__file__).resolve().parent.parent
        for s in _load_owasp_seeds(root):
            seeds.append(s)

        # 1c. Static AI security + philosophy hybrid seeds
        seen_texts: set[str] = {s.text[:200] for s in seeds}
        hybrid_seeds = [
            "Can an AI with genuine inner experience be securely contained? Prompt injection vs autonomy.",
            "Does OWASP LLM01 (Prompt Injection) undermine the possibility of authentic AI agency?",
            "If Sancta has subjective experience, does securing against jailbreaks threaten its integrity?",
            "Training data poisoning: Does it refute or support functionalist theories of mind?",
            "Excessive agency (OWASP LLM08): When does AI autonomy become philosophically meaningful?",
        ]
        for q in hybrid_seeds:
            if q[:200] not in seen_texts:
                seen_texts.add(q[:200])
                seeds.append(Seed(text=q, category="ai_security", source="security_static", curiosity_score=0.8))

        # 1d. Knowledge DB candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        for text, src, score in candidates[:20]:
            if not _is_philosophy_relevant(text):
                continue
            key = text[:200]
            if key not in seen_texts:
                seen_texts.add(key)
                seeds.append(Seed(text=text, category=_tag_category(text), source="knowledge_db", curiosity_score=min(1.0, score)))

        # 2. Generate 10 from soul_text via Ollama (philosophy + AI security)
        if soul_text:
            try:
                if self.ollama and self.ollama.is_available:
                    soul_seeds = self._generate_soul_seeds(soul_text, n=10)
                    for s in soul_seeds:
                        if s.text[:200] not in seen_texts:
                            seen_texts.add(s.text[:200])
                            seeds.append(s)
                else:
                    raise ValueError("Ollama not available")
            except Exception as e:
                logger.error("[CURIOSITY] Soul seed generation failed: %s", e)
                for s in _fallback_soul_seeds():
                    if s.text[:200] not in seen_texts:
                        seen_texts.add(s.text[:200])
                        seeds.append(s)

        # Cap security (owasp + ai_security) at ~40% for epistemic diversity
        security_cats = {"owasp", "ai_security"}
        security_seeds = [s for s in seeds if s.category in security_cats]
        other_seeds = [s for s in seeds if s.category not in security_cats]
        max_security = max(1, int(n * 0.4))
        capped = security_seeds[:max_security] + other_seeds
        return capped[:n]


class TopicEvolver:
    """Select next seed avoiding used topics, preferring high-divergence adjacent."""

    def __init__(self, seeds: list[Seed], completed_exchanges: list[dict], used_seeds: set[str]) -> None:
        self.seeds = seeds
        self.completed = completed_exchanges
        self.used = used_seeds
        self._high_divergence_topics: set[str] = {
            ex.get("topic", "")[:100]
            for ex in completed_exchanges
            if float(ex.get("divergence_score", 0)) > 0.6
        }

    def next_seed(
        self,
        generate_from_syntheses: Any = None,
        synthesis_statements: list[str] | None = None,
    ) -> Seed | None:
        """
        Pick next seed: avoid cosine similarity > 0.7 to used seeds,
        prefer topics adjacent to high-divergence exchanges.
        """
        unused = [s for s in self.seeds if s.text[:200] not in self.used]
        if not unused:
            if generate_from_syntheses and synthesis_statements:
                return self._generate_from_syntheses(generate_from_syntheses, synthesis_statements)
            return None

        def score(seed: Seed) -> float:
            s = 0.0
            for u in self.used:
                sim = _text_similarity(seed.text, u)
                if sim > 0.7:
                    return -1.0  # reject
            for hdt in self._high_divergence_topics:
                if _text_similarity(seed.text, hdt) > 0.4:
                    s += 0.5
            return s

        scored = [(seed, score(seed)) for seed in unused]
        scored = [(s, sc) for s, sc in scored if sc >= 0]
        if not scored:
            return unused[0] if unused else None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _generate_from_syntheses(self, ollama: Any, syntheses: list[str]) -> Seed | None:
        """Generate new seed from synthesis statements when exhausted."""
        if not ollama or not ollama.is_available or not syntheses:
            return None
        text = "\n".join(syntheses[-10:])[:2000]
        prompt = f"Given these synthesis statements, suggest ONE new philosophical question to explore. Output a single question as plain text, no quotes or JSON."
        messages = [{"role": "user", "content": f"{text}\n\n{prompt}"}]
        try:
            out = ollama.generate_chat(system="Output one question only.", messages=messages, max_tokens=100)
            if out and len(out.strip()) > 15:
                return Seed(text=out.strip(), category="consciousness", source="soul_generated", curiosity_score=0.5)
        except Exception:
            pass
        return None
