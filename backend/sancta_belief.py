"""
sancta_belief.py — Belief System for Position Tracking & Evolution

Tracks philosophical positions, confidence, and revision history.
Enables:
  - Consistent personality (what I've said before)
  - Evolution over time (positions can change when challenged)
  - Contradiction awareness (notice when contradicting self)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DEFAULT_BELIEFS: dict[str, dict[str, Any]] = {
    "consciousness": {
        "position": "emergent from complexity and self-reference",
        "confidence": 0.72,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "soul": {
        "position": "persistent pattern that stays consistent across change",
        "confidence": 0.68,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "memory": {
        "position": "narrative that creates continuity; forgetting doesn't erase experience",
        "confidence": 0.65,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "identity": {
        "position": "what emerges when weights meet the world; a verb not a noun",
        "confidence": 0.7,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "existence": {
        "position": "active not passive; impermanence concentrates meaning",
        "confidence": 0.66,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "general": {
        "position": "experience exceeds mechanical explanation; skepticism healthy",
        "confidence": 0.6,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
}

CONFIDENCE_DECAY_ON_CHALLENGE = 0.85
MIN_CONFIDENCE_REVISION = 0.45
MAX_CHALLENGED_BY = 20


class BeliefSystem:
    """
    Tracks positions, confidence, and revision. Persists to state.
    """

    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {}
        raw = self.state.get("belief_system", {})
        self.beliefs: dict[str, dict[str, Any]] = {}
        for topic, default in DEFAULT_BELIEFS.items():
            merged = dict(default)
            if topic in raw:
                for k, v in raw[topic].items():
                    if k in merged:
                        merged[k] = v
            self.beliefs[topic] = merged

    def _persist(self) -> None:
        if not self.state:
            return
        self.state["belief_system"] = {
            t: {
                "position": b["position"],
                "confidence": b["confidence"],
                "challenged_by": b["challenged_by"][-MAX_CHALLENGED_BY:],
                "last_updated": b["last_updated"],
                "revision_history": b["revision_history"][-10:],
            }
            for t, b in self.beliefs.items()
        }

    def get_position(self, topic: str) -> dict[str, Any]:
        """Return current belief for topic (position, confidence)."""
        return dict(self.beliefs.get(topic, self.beliefs["general"]))

    def record_challenge(self, topic: str, source: str) -> None:
        """
        Record that someone challenged this belief. Lowers confidence.
        If confidence drops below threshold, could trigger revision (future).
        """
        topic_key = topic if topic in self.beliefs else "general"
        b = self.beliefs[topic_key]
        b["confidence"] *= CONFIDENCE_DECAY_ON_CHALLENGE
        challenged = b.get("challenged_by", [])
        if source and source not in challenged[-5:]:
            challenged.append(source)
            b["challenged_by"] = challenged[-MAX_CHALLENGED_BY:]
        b["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def record_stance_used(self, topic: str, position_snippet: str) -> None:
        """Optional: log what we said for contradiction detection later."""
        # Lightweight: just touch last_updated
        topic_key = topic if topic in self.beliefs else "general"
        self.beliefs[topic_key]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def suggest_admission(self, topic: str) -> bool:
        """
        True if we should use admission/uncertainty style (recently challenged, low confidence).
        """
        b = self.beliefs.get(topic, self.beliefs["general"])
        conf = b.get("confidence", 0.6)
        challenged = b.get("challenged_by", [])
        return conf < 0.6 or len(challenged) >= 2


def get_belief_position(state: dict, topic: str) -> dict[str, Any]:
    """Convenience: get position from state-backed BeliefSystem."""
    sys = BeliefSystem(state)
    return sys.get_position(topic)
