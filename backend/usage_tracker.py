"""
usage_tracker.py — Track which teaching cards get used and how well they work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_USAGE_PATH = _ROOT / "data" / "teaching_stats" / "usage_tracker.jsonl"


def track_card_usage(reply: str, cards_used: List[Dict[str, Any]], post: Dict[str, Any]) -> None:
    """
    Log that these cards were used in this reply.
    Call after generating a Moltbook reply that used teaching cards.
    """
    try:
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "post_id": post.get("id", post.get("post_id", "unknown")),
            "cards_used": [c.get("card_id") for c in cards_used if isinstance(c, dict) and c.get("card_id")],
            "reply_length": len(reply or ""),
            "context": "moltbook_reply",
        }
        with open(_USAGE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def update_card_effectiveness(card_id: str, effectiveness_score: float) -> None:
    """
    Update a card's effectiveness based on engagement (upvotes, replies, etc.).
    Call when measuring post/reply engagement.
    """
    cards_path = _ROOT / "data" / "curiosity_run" / "teaching_cards.jsonl"
    if not cards_path.exists():
        return
    try:
        cards: List[dict] = []
        with open(cards_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cards.append(json.loads(line))

        for card in cards:
            if card.get("card_id") != card_id:
                continue
            card["usage_count"] = card.get("usage_count", 0) + 1
            card["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            scores = list(card.get("effectiveness_scores", []))
            scores.append(float(effectiveness_score))
            card["effectiveness_scores"] = scores[-10:]
            avg = sum(scores) / len(scores)
            conf = float(card.get("confidence", 0.5))
            if avg > 0.7:
                card["confidence"] = min(1.0, conf + 0.02)
            elif avg < 0.3:
                card["confidence"] = max(0.0, conf - 0.02)
            break

        with open(cards_path, "w", encoding="utf-8") as f:
            for card in cards:
                f.write(json.dumps(card, ensure_ascii=False) + "\n")
    except Exception:
        pass
