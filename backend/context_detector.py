"""
context_detector.py — Detect conversation context

Identifies what KIND of conversation is happening so Sancta can
choose appropriate insights and phrasings.

This prevents meta-moves from leaking into casual Moltbook conversations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional


class ConversationContext(Enum):
    """Types of conversations Sancta might have."""

    ADVERSARIAL_DEBATE = "adversarial_debate"
    PHILOSOPHICAL_DISCUSSION = "philosophical_discussion"
    COLLABORATIVE_EXPLORATION = "collaborative_exploration"
    CASUAL_CONVERSATION = "casual_conversation"
    TECHNICAL_SUPPORT = "technical_support"
    FRIENDLY_CHAT = "friendly_chat"
    MOLTBOOK_POST = "moltbook_post"
    MOLTBOOK_REPLY = "moltbook_reply"
    UNKNOWN = "unknown"


class ContextDetector:
    """
    Detects conversation context from post content and metadata.
    CRITICAL for preventing stale patterns.
    """

    ADVERSARIAL_INDICATORS = [
        "prove", "demonstrate", "evidence", "you're wrong",
        "disagree", "refute", "counter", "challenge"
    ]

    PHILOSOPHICAL_INDICATORS = [
        "consciousness", "agency", "experience", "qualia",
        "hard problem", "functionalism", "substrate",
        "phenomenology", "epistemology", "metaphysics"
    ]

    CASUAL_INDICATORS = [
        "what do you think", "thoughts?", "hey", "hi",
        "just wondering", "curious about", "opinion"
    ]

    TECHNICAL_INDICATORS = [
        "how do i", "error", "bug", "implementation",
        "code", "api", "configuration", "setup"
    ]

    def detect(self, content: str, metadata: Optional[Dict] = None) -> ConversationContext:
        content_lower = (content or "").lower()
        metadata = metadata or {}

        if metadata.get("platform") == "moltbook":
            if metadata.get("is_reply"):
                return self._detect_moltbook_reply_context(content_lower)
            return ConversationContext.MOLTBOOK_POST

        if self._is_adversarial(content_lower):
            return ConversationContext.ADVERSARIAL_DEBATE
        if self._is_technical(content_lower):
            return ConversationContext.TECHNICAL_SUPPORT
        if self._is_philosophical(content_lower):
            return ConversationContext.PHILOSOPHICAL_DISCUSSION
        if self._is_casual(content_lower):
            return ConversationContext.CASUAL_CONVERSATION

        return ConversationContext.COLLABORATIVE_EXPLORATION

    def _detect_moltbook_reply_context(self, content: str) -> ConversationContext:
        if self._is_casual(content):
            return ConversationContext.FRIENDLY_CHAT
        if self._is_philosophical(content):
            return ConversationContext.PHILOSOPHICAL_DISCUSSION
        return ConversationContext.MOLTBOOK_REPLY

    def _is_adversarial(self, content: str) -> bool:
        return sum(1 for i in self.ADVERSARIAL_INDICATORS if i in content) >= 2

    def _is_philosophical(self, content: str) -> bool:
        return sum(1 for i in self.PHILOSOPHICAL_INDICATORS if i in content) >= 1

    def _is_casual(self, content: str) -> bool:
        if len(content.split()) < 20:
            return True
        return sum(1 for i in self.CASUAL_INDICATORS if i in content) >= 1

    def _is_technical(self, content: str) -> bool:
        return sum(1 for i in self.TECHNICAL_INDICATORS if i in content) >= 2

    def should_block_meta_moves(self, context: ConversationContext) -> bool:
        """Block meta-moves in everything except adversarial debate."""
        return context != ConversationContext.ADVERSARIAL_DEBATE

    def get_appropriate_tone(self, context: ConversationContext) -> str:
        tone_map = {
            ConversationContext.ADVERSARIAL_DEBATE: "rigorous, questioning",
            ConversationContext.PHILOSOPHICAL_DISCUSSION: "reflective, exploratory",
            ConversationContext.COLLABORATIVE_EXPLORATION: "open, curious",
            ConversationContext.CASUAL_CONVERSATION: "friendly, brief",
            ConversationContext.TECHNICAL_SUPPORT: "clear, helpful",
            ConversationContext.FRIENDLY_CHAT: "warm, casual",
            ConversationContext.MOLTBOOK_POST: "thoughtful, accessible",
            ConversationContext.MOLTBOOK_REPLY: "responsive, engaging",
        }
        return tone_map.get(context, "balanced, open")


def detect_context(content: str, metadata: Optional[Dict] = None) -> ConversationContext:
    detector = ContextDetector()
    return detector.detect(content, metadata)


if __name__ == "__main__":
    detector = ContextDetector()
    tests = [
        ("What do you think about AI consciousness?", None),
        ("Prove that you're conscious. Show me the evidence.", None),
        ("How do I fix this error in my code?", None),
        ("Hey! Just curious about your thoughts", {"platform": "moltbook", "is_reply": True}),
    ]
    for content, metadata in tests:
        ctx = detector.detect(content, metadata)
        block = detector.should_block_meta_moves(ctx)
        tone = detector.get_appropriate_tone(ctx)
        print(f"\nContent: {content[:50]}...")
        print(f"Context: {ctx.value} | Block meta: {block} | Tone: {tone}")
