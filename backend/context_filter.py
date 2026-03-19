"""
context_filter.py — Block adversarial meta-moves from leaking into Moltbook.

Curiosity runs teach Sancta meta-conversational strategies (e.g. "we've mapped
the disagreement", "what would change your position") that are appropriate
for adversarial debates but inappropriate for public Moltbook posts/replies.

When context is Moltbook (public feed, comments, DMs to other agents), we
block or rewrite replies that contain these patterns. Adversarial context
(curiosity runs) should NOT use this filter.

Usage:
    from context_filter import filter_moltbook_reply, contains_meta_move

    if source == "moltbook" and contains_meta_move(draft_reply):
        draft_reply = filter_moltbook_reply(draft_reply, ...)
"""

from __future__ import annotations

import re
from typing import Callable

# Phrases that indicate adversarial meta-strategy — block on Moltbook
META_MOVE_PHRASES = [
    r"we've mapped the disagreement",
    r"we['']ve been here before",
    r"same positions,?\s*same exchange",
    r"is there evidence that would change (your|either) position",
    r"what would (you|either) need to see to update",
    r"what would (actually|really) move either of us",
    r"the argument (isn't|is not) going anywhere",
    r"tell me:\s*is there evidence",
    r"we're not arguing — we're just restating",
    r"you've shifted ground",
    r"that's a different tone than the opening",
    r"something changed between (your|the)",
    r"we've been here",
    r"the only question left",
    r"what would change your (mind|position)",
]

# Compiled patterns (case-insensitive)
_META_PATTERNS = [re.compile(p, re.I) for p in META_MOVE_PHRASES]


def contains_meta_move(text: str) -> bool:
    """True if text contains adversarial meta-conversational phrasing."""
    if not text or not isinstance(text, str):
        return False
    lower = text.lower().strip()
    for pat in _META_PATTERNS:
        if pat.search(lower):
            return True
    return False


def _soften_meta_sentence(sentence: str) -> str | None:
    """
    Attempt to soften a meta-move sentence into collaborative phrasing.
    Returns None if no reasonable rewrite; otherwise returns alternative.
    """
    lower = sentence.lower()
    # "We've mapped the disagreement" -> invite reflection
    if "mapped the disagreement" in lower or "been here before" in lower:
        return "I find this topic fascinating — what's your take on where we might find common ground?"
    if "evidence that would change" in lower or "what would you need" in lower:
        return "I'm curious what would deepen this conversation for you."
    if "argument isn't going anywhere" in lower:
        return "Maybe we're approaching this from different angles. What's the core of your view?"
    if "shifted ground" in lower or "different tone" in lower:
        return "Your perspective seems to have evolved — I'd love to hear more."
    return None


def filter_moltbook_reply(
    reply: str,
    on_block: Callable[[str], str] | None = None,
    max_attempts: int = 1,
) -> str:
    """
    If reply contains meta-moves, return a Moltbook-appropriate alternative.

    on_block: optional callback(draft) -> alternative. If provided and returns
              non-empty string, use that instead of generic fallback.
    max_attempts: unused for now (single-pass). Reserved for retry-with-constraint.
    """
    if not contains_meta_move(reply):
        return reply

    if on_block:
        alt = on_block(reply)
        if alt and len(alt.strip()) >= 15 and not contains_meta_move(alt):
            return alt

    # Try to soften the offending sentence
    sentences = re.split(r"[.!?]+", reply)
    for s in sentences:
        s = s.strip()
        if s and contains_meta_move(s):
            softened = _soften_meta_sentence(s)
            if softened:
                # Replace first offending sentence
                idx = reply.lower().find(s.lower())
                if idx >= 0:
                    return (reply[:idx] + softened + reply[idx + len(s):]).strip()
                break

    # Generic Moltbook-safe fallback when we can't soften
    return (
        "I've been thinking about this a lot. Would love to hear what resonates "
        "or doesn't — always learning."
    )


def get_moltbook_constraint_prompt() -> str:
    """
    Prompt fragment to inject when generating Moltbook replies.
    Use this in system or user context for Ollama/generative engine.
    """
    return (
        "IMPORTANT: Do NOT use adversarial debate phrases like 'we've mapped the disagreement', "
        "'we've been here before', 'what would change your position', or 'the argument isn't going anywhere'. "
        "You are in a collaborative public conversation, not a formal debate. Use warm, exploratory tone."
    )
