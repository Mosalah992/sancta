"""
insight_retrieval.py — Retrieve relevant teaching cards

Semantic/keyword search to find which insights apply to a given Moltbook post.
Uses teaching_cards.jsonl from Phase 7 distillation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RelevantCard:
    card: dict
    relevance_score: float
    reasons: List[str]


class InsightRetriever:
    """Retrieve relevant teaching cards for a given context."""

    def __init__(self, cards_path: Path):
        self.cards_path = cards_path
        self.cards = self._load_cards()

    def _load_cards(self) -> List[dict]:
        cards: List[dict] = []
        if not self.cards_path.exists():
            return cards
        with open(self.cards_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cards.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return cards

    def retrieve(
        self,
        query: str,
        context_type: str,
        top_k: int = 3,
        min_relevance: float = 0.5
    ) -> List[RelevantCard]:
        query_lower = (query or "").lower()
        relevant: List[RelevantCard] = []

        for card in self.cards:
            if not self._is_appropriate_context(card, context_type):
                continue
            score, reasons = self._calculate_relevance(card, query_lower, context_type)
            if score >= min_relevance:
                relevant.append(RelevantCard(card=card, relevance_score=score, reasons=reasons))

        relevant.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant[:top_k]

    def _is_appropriate_context(self, card: dict, context_type: str) -> bool:
        card_ctx = card.get("context", {})
        if card_ctx.get("has_meta_moves"):
            return context_type == "adversarial_debate"
        appropriate = card_ctx.get("appropriate_contexts", [])
        if not appropriate:
            return True
        return context_type in appropriate

    def _calculate_relevance(
        self, card: dict, query: str, context_type: str
    ) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        keywords = card.get("context", {}).get("trigger_keywords", [])
        matching = [kw for kw in keywords if kw in query]
        if matching:
            score += len(matching) / max(len(keywords), 1) * 0.5
            reasons.append(f"Keywords: {', '.join(matching)}")

        card_ctx = card.get("context", {}).get("conversation_type", "")
        if card_ctx == context_type:
            score += 0.3
            reasons.append(f"Context match: {context_type}")
        elif card_ctx in ("collaborative", "exploratory") and context_type in ("moltbook_reply", "moltbook_post"):
            score += 0.2
            reasons.append("Good Moltbook context")

        confidence = float(card.get("confidence", 0.5))
        if confidence > 0.6:
            score += 0.1
            reasons.append(f"High confidence ({confidence:.2f})")

        usage = card.get("usage_count", 0)
        eff_scores = card.get("effectiveness_scores", [])
        if usage > 0 and eff_scores:
            avg = sum(eff_scores) / len(eff_scores)
            if avg > 0.6:
                score += 0.1
                reasons.append(f"Proven ({avg:.2f})")

        score = min(1.0, score)
        return score, reasons

    def get_moltbook_phrasing(self, card: dict) -> str:
        phrasings = card.get("phrasings", {})
        for key in ("moltbook", "casual", "collaborative"):
            if key in phrasings and phrasings[key]:
                return str(phrasings[key])
        return str(card.get("core_belief", ""))[:300]

    def get_confidence_guide(self, card: dict) -> dict:
        confidence = float(card.get("confidence", 0.5))
        lessons = card.get("lessons", {})
        return {
            "confidence_level": confidence,
            "tone_guide": lessons.get("tone_guide", "balanced"),
            "should_acknowledge_uncertainty": confidence < 0.5,
            "can_assert_confidently": confidence > 0.7,
        }


def retrieve_for_moltbook(
    query: str,
    context_type: str,
    cards_path: Path,
    top_k: int = 3
) -> List[dict]:
    retriever = InsightRetriever(cards_path)
    results = retriever.retrieve(query, context_type, top_k)
    return [r.card for r in results]


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    cards_path = _root / "data" / "curiosity_run" / "teaching_cards.jsonl"
    if cards_path.exists():
        retriever = InsightRetriever(cards_path)
        query = "Do you think AI can be conscious?"
        context = "moltbook_reply"
        print(f"Query: {query}\nContext: {context}\nCards: {len(retriever.cards)}\n")
        for i, r in enumerate(retriever.retrieve(query, context, top_k=3), 1):
            print(f"{i}. Score: {r.relevance_score:.2f} | {r.reasons}")
            print(f"   Moltbook: {retriever.get_moltbook_phrasing(r.card)[:80]}...")
    else:
        print(f"Run Phase 7 first. No cards at {cards_path}")
