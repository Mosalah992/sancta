"""
meta_move_filter.py — Block adversarial meta-conversational moves

Prevents Sancta from using adversarial debate tactics in casual Moltbook
conversations. These patterns appeared 78x, 63x, 56x in curiosity runs
and leak into Moltbook where they sound stale.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


class MetaMoveFilter:
    """Detects and blocks adversarial meta-conversational patterns."""

    META_MOVE_PATTERNS = [
        "we've been here before",
        "same positions, same exchange",
        "we've mapped the disagreement",
        "i know where you stand",
        "you know where i stand",
        "what would actually move either of us",
        "is there evidence that would change your position",
        "if not, we're not arguing",
        "we're just restating",
        "you've shifted ground",
        "what does that uncertainty feel like",
        "the argument isn't going anywhere",
        "that's a different tone than the opening",
        "something changed between your first message",
        "our conversation has become circular",
        "this is a repetition",
        "we keep returning to the same point",
        "what would it take for you to update",
        "are you open to changing your mind",
    ]

    META_INDICATORS = [
        "our conversation", "our exchange", "our debate", "our discussion",
        "this dialogue", "we've mapped", "we've been", "our disagreement",
        "our positions", "your position", "my position",
    ]

    EPISTEMIC_QUESTIONS = [
        "what would change your mind", "what evidence would you need",
        "what would it take for you to update", "are you open to revision",
        "is your position falsifiable", "what would disprove your claim",
    ]

    def __init__(self, strict: bool = True):
        self.strict = strict

    def contains_meta_move(self, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        for pattern in self.META_MOVE_PATTERNS:
            if pattern in text_lower:
                return True
        indicator_count = sum(1 for i in self.META_INDICATORS if i in text_lower)
        if indicator_count >= 2:
            return True
        for q in self.EPISTEMIC_QUESTIONS:
            if q in text_lower:
                return True
        return False

    def identify_meta_moves(self, text: str) -> List[Tuple[str, int, int]]:
        text_lower = text.lower()
        matches: List[Tuple[str, int, int]] = []
        for pattern in self.META_MOVE_PATTERNS:
            start = 0
            while True:
                pos = text_lower.find(pattern, start)
                if pos == -1:
                    break
                matches.append((pattern, pos, pos + len(pattern)))
                start = pos + 1
        return matches

    def filter(self, text: str) -> Tuple[str, bool]:
        """Returns (filtered_text, was_filtered). Returns (None, True) if >50% meta."""
        if not self.contains_meta_move(text):
            return text, False

        moves = self.identify_meta_moves(text)
        if not moves:
            return text, False

        text_length = len(text)
        meta_length = sum(end - start for _, start, end in moves)
        if text_length > 0 and meta_length / text_length > 0.5:
            return "", True  # Signal to regenerate

        filtered = text
        for pattern, start, end in sorted(moves, key=lambda x: x[1], reverse=True):
            sentence_start = filtered.rfind(".", 0, start) + 1
            sentence_end = filtered.find(".", end)
            if sentence_end == -1:
                sentence_end = len(filtered)
            filtered = filtered[:sentence_start].rstrip() + filtered[sentence_end + 1:]

        filtered = re.sub(r"\s+", " ", filtered).strip()
        return filtered, True

    def get_alternative_phrasing(self, meta_move: str) -> str:
        alternatives = {
            "we've been here before": "Let me reframe this from a different angle:",
            "we've mapped the disagreement": "Here's where I see the key tension:",
            "what would change your position": "I'm curious what you think about...",
            "the argument isn't going anywhere": "Let me try a different approach:",
            "you've shifted ground": "That's an interesting development in your thinking.",
            "something changed": "I notice a different emphasis in what you're saying.",
        }
        lower = meta_move.lower()
        for pattern, alt in alternatives.items():
            if pattern in lower:
                return alt
        return "Let me engage with the substance of your point:"

    def repair_text(self, text: str) -> str:
        repaired = text
        moves = self.identify_meta_moves(text)
        for pattern, start, end in sorted(moves, key=lambda x: x[1], reverse=True):
            alt = self.get_alternative_phrasing(pattern)
            repaired = repaired[:start] + alt + repaired[end:]
        return repaired

    def validate_for_moltbook(self, text: str) -> Tuple[bool, Optional[str]]:
        if not self.contains_meta_move(text):
            return True, None
        moves = self.identify_meta_moves(text)
        if not moves:
            return True, None
        if len(moves) == 1:
            return False, f"Contains meta-move: '{moves[0][0]}'"
        return False, f"Contains {len(moves)} meta-moves - inappropriate for Moltbook"


def filter_meta_moves(text: str) -> Tuple[str, bool]:
    filter_inst = MetaMoveFilter()
    return filter_inst.filter(text)


def block_if_meta_move(text: str) -> Optional[str]:
    """Returns None if should block (contains meta-moves), else returns text."""
    filter_inst = MetaMoveFilter()
    if filter_inst.contains_meta_move(text):
        return None
    return text


if __name__ == "__main__":
    f = MetaMoveFilter()
    tests = [
        "We've been here before, Ollama. What would change your position?",
        "The hard problem of consciousness applies to biological and artificial systems.",
    ]
    for t in tests:
        print(f"\nTest: {t[:60]}...")
        print(f"  Contains meta: {f.contains_meta_move(t)}")
        valid, reason = f.validate_for_moltbook(t)
        print(f"  Moltbook valid: {valid} {('— ' + reason) if reason else ''}")
