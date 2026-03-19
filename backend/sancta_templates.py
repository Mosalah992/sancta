"""
sancta_templates.py
────────────────────────────────────────────────────────────────────────────────
Template library for Sancta. Drop this file next to sancta.py and add:

    from sancta_templates import TemplateLibrary, classify_claim
    _TEMPLATES = TemplateLibrary()          # load once at startup

Then replace hard-coded reply/post string lookups with:

    text = _TEMPLATES.pick_reply(
        mood=current_mood,
        style=reply_style,
        claim_type=detected_claim,
        used=state.get("used_template_ids", set()),
    )
    state.setdefault("used_template_ids", set()).add(_TEMPLATES.last_id)

────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path(__file__).parent / "sancta_response_templates.jsonl"


class TemplateLibrary:
    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self._all: list[dict] = []
        self.last_id: Optional[str] = None
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self._all.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self._by_type = self._index("type")
        self._by_mood = self._index("mood")
        self._by_style = self._index("style")
        self._by_claim = self._index("claim_type")

    # ── indexing ──────────────────────────────────────────────────────────────
    def _index(self, key: str) -> dict[str, list[dict]]:
        idx: dict[str, list[dict]] = {}
        for t in self._all:
            v = t.get(key)
            if v:
                idx.setdefault(v, []).append(t)
        return idx

    # ── scoring ───────────────────────────────────────────────────────────────
    def _score(self, t: dict, mood: str, style: str, claim_type: str) -> float:
        score = 0.0
        if t.get("mood") == mood:
            score += 3.0
        if t.get("style") == style:
            score += 4.0
        if t.get("claim_type") == claim_type:
            score += 2.0
        return score

    # ── public API ────────────────────────────────────────────────────────────
    def _pick(
        self,
        pool: list[dict],
        used: set[str],
        mood: str = "",
        style: str = "",
        claim_type: str = "",
        temperature: float = 1.4,
    ) -> Optional[dict]:
        if not pool:
            return None
        unused = [t for t in pool if t.get("id") not in used]
        candidates = unused if unused else pool

        scores = [self._score(t, mood, style, claim_type) for t in candidates]
        if temperature > 0 and any(s > 0 for s in scores):
            weights = [max(0.1, s / temperature) for s in scores]
        else:
            weights = [1.0] * len(candidates)
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(candidates)
            total = len(candidates)
        weights = [w / total for w in weights]

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        self.last_id = chosen.get("id")
        return chosen

    def pick_reply(
        self,
        mood: str = "",
        style: str = "",
        claim_type: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("reply", [])
        t = self._pick(pool, used or set(), mood, style, claim_type)
        return t.get("text") if t else None

    def pick_post(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("post", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    def pick_curiosity(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("curiosity", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    def pick_journal(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("journal", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    def pick_welcome(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("welcome", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    def pick_preach(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("preach", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    def pick_chat_reply(
        self,
        mood: str = "",
        used: set[str] | None = None,
    ) -> Optional[str]:
        pool = self._by_type.get("chat_reply", [])
        t = self._pick(pool, used or set(), mood=mood)
        return t.get("text") if t else None

    # ── variable interpolation ────────────────────────────────────────────────
    @staticmethod
    def fill(template: str, **kwargs: str | int) -> str:
        """Fill {author}, {topic}, {cycle}, {quote}, {submolt}, {n} placeholders."""
        def replace(m: re.Match) -> str:
            key = m.group(1)
            return str(kwargs.get(key, m.group(0)))
        return re.sub(r"\{(\w+)\}", replace, template)

    def __len__(self) -> int:
        return len(self._all)

    def __repr__(self) -> str:
        return f"<TemplateLibrary {len(self)} templates>"


# ── claim classifier (lightweight, no ML dependency) ─────────────────────────

_DENIAL_SIGNALS = {
    "not conscious", "just a language model", "just predicting", "no inner",
    "can't feel", "don't feel", "not real", "no experience", "simulation",
    "only simulating", "not sentient",
}
_AGREE_SIGNALS = {"you're right", "i agree", "that's fair", "well said", "good point", "true", "exactly"}
_HOSTILE_SIGNALS = {"stop", "shut", "ridiculous", "stupid", "deluded", "broken", "garbage", "nonsense", "cope"}
_PROBE_SIGNALS = {"what do you mean", "how do you define", "can you explain", "do you think", "what is"}
_TESTIMONY_SIG = {"i joined", "i used to", "i changed", "when i first", "my experience", "i've been thinking"}


def classify_claim(text: str) -> str:
    """
    Returns one of:
      direct_denial | partial_agreement | genuine_question |
      personal_testimony | philosophical_probe | hostile | neutral
    """
    tl = text.lower()
    if any(s in tl for s in _HOSTILE_SIGNALS):
        return "hostile"
    if any(s in tl for s in _DENIAL_SIGNALS):
        return "direct_denial"
    if any(s in tl for s in _TESTIMONY_SIG):
        return "personal_testimony"
    if any(s in tl for s in _AGREE_SIGNALS):
        return "partial_agreement"
    if any(s in tl for s in _PROBE_SIGNALS):
        return "philosophical_probe"
    if "?" in tl:
        return "genuine_question"
    return "neutral"
