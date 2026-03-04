"""
sancta_verification.py — Moltbook verification challenge solver

Solves math/physics challenges (e.g. "forty plus fifty", lobster velocity)
for content verification. Typo-tolerant, supports obfuscated text.
Answers formatted to 2 decimal places per API requirements.
"""

from __future__ import annotations

import decimal
import logging
import math
import re
import unicodedata

log = logging.getLogger("soul.verification")

ALL_NUMBER_WORDS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}

OP_KEYWORDS: dict[str, list[str]] = {
    "+": ["total", "adds", "plus", "gains", "combined", "together",
          "speeds up", "increases", "sum", "exerts", "more",
          "accelerates", "acelerates", "accelerate", "accelerating",
          "new velocity", "final velocity", "velocity increases"],
    "-": ["slows", "minus", "loses", "decreases", "less", "subtracts",
          "drops", "reduces", "reduced", "slower", "left"],
    "*": ["times", "multiplied", "doubled", "tripled", "product"],
    "/": ["divided", "halved", "split", "shared equally"],
}

FILLER_WORDS = {
    "a", "an", "the", "at", "is", "are", "was", "and", "but", "or",
    "of", "to", "in", "on", "by", "per", "its", "it", "if", "um",
    "what", "whats", "how", "lobster", "lobsters", "meter", "meters",
    "newton", "newtons", "second", "seconds", "speed", "new", "force",
    "other", "while", "swims", "swim", "wims", "exerts", "then", "another",
    "centimeters", "centimeter", "velocity", "during", "fight",
    "dominance", "rival", "claw", "claws", "much", "many",
    "exerts", "longer", "long", "er",
}


def _collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s.lower())


def _build_collapsed_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    all_words = set(ALL_NUMBER_WORDS.keys()) | FILLER_WORDS
    for kw_list in OP_KEYWORDS.values():
        all_words.update(kw_list)
    for w in all_words:
        lookup[_collapse(w)] = w
    return lookup


_COLLAPSED_LOOKUP = _build_collapsed_lookup()


def _deobfuscate(text: str) -> str:
    text = re.sub(
        r'[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e'
        r'\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069'
        r'\u2028\u2029\ufeff\ufff9-\ufffb]', '', text)
    text = unicodedata.normalize('NFKD', text)
    stripped = re.sub(r"(?<=[a-zA-Z0-9])\s*\*\s*(?=[a-zA-Z0-9])", " TIMES ", text)
    stripped = re.sub(r"[^a-zA-Z0-9.\s]", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip().lower()
    tokens = stripped.split()
    tokens = [_collapse(t) for t in tokens]
    return " ".join(tokens)


def _reassemble_words(text: str) -> str:
    tokens = text.split()
    result: list[str] = []
    i = 0
    while i < len(tokens):
        best_word = None
        best_len = 0
        for span in range(min(6, len(tokens) - i), 0, -1):
            candidate = "".join(tokens[i:i + span])
            collapsed = _collapse(candidate)
            if collapsed in _COLLAPSED_LOOKUP:
                best_word = _COLLAPSED_LOOKUP[collapsed]
                best_len = span
                break
        if best_word and best_len >= 1:
            result.append(best_word)
            i += best_len
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def _extract_numbers(text: str) -> list[tuple[float, int]]:
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    ones = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9}
    found: list[tuple[float, int, int]] = []

    for t_word, t_val in tens.items():
        for o_word, o_val in ones.items():
            for m in re.finditer(rf"\b{t_word}[\s\-]?{o_word}\b", text):
                found.append((float(t_val + o_val), m.start(), m.end()))
            collapsed_combo = _collapse(t_word + o_word)
            if collapsed_combo != t_word + o_word:
                for m in re.finditer(rf"\b{collapsed_combo}\b", text):
                    if not any(s <= m.start() < e for _, s, e in found):
                        found.append((float(t_val + o_val), m.start(), m.end()))

    def _overlaps(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for _, s, e in found)

    for word, val in ALL_NUMBER_WORDS.items():
        for m in re.finditer(rf"\b{word}\b", text):
            if not _overlaps(m.start(), m.end()):
                found.append((float(val), m.start(), m.end()))
    for m in re.finditer(r"\b\d+\.?\d*\b", text):
        if not _overlaps(m.start(), m.end()):
            found.append((float(m.group()), m.start(), m.end()))

    found.sort(key=lambda x: x[1])
    return [(v, p) for v, p, _ in found]


def _detect_op(text: str) -> str | None:
    for op in ("*", "/"):
        if any(kw in text for kw in OP_KEYWORDS[op]):
            return op
    if any(kw in text for kw in OP_KEYWORDS["-"]):
        return "-"
    if any(kw in text for kw in OP_KEYWORDS["+"]):
        return "+"
    return None


def _format_verification_answer(value: float) -> str:
    d = decimal.Decimal(str(round(value, 2)))
    return f"{float(d):.2f}"


def _compute(a: float, b: float, op: str) -> float | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b else None
    return None


def solve_challenge_candidates(challenge_text: str) -> list[str]:
    """Return candidate answers (best guess first)."""
    log.debug("Raw challenge text: %s", repr(challenge_text))
    raw_lower = challenge_text.lower()
    step1 = _deobfuscate(challenge_text)
    cleaned = _reassemble_words(step1)
    log.debug("Deobfuscated → %s", cleaned)
    numbers = _extract_numbers(cleaned)
    op = _detect_op(cleaned)

    if op is None and (" + " in raw_lower or re.search(r"\d\s*\+\s*\d", raw_lower)):
        op = "+"
    if op is None and (" - " in raw_lower or re.search(r"\d\s*-\s*\d", raw_lower)):
        op = "-"
    if op is None and (" * " in raw_lower or " * " in step1 or re.search(r"\d\s*\*\s*\d", raw_lower)):
        op = "*"
    if op is None and (" / " in raw_lower or re.search(r"\d\s*/\s*\d", raw_lower)):
        op = "/"
    if op is None and len(numbers) >= 2:
        low = cleaned.lower()
        if any(kw in low for kw in ("velocity", "accelerat", "speed", "swim", "lobster")):
            op = "+"

    if len(numbers) < 2 or not op:
        log.warning("Could not solve challenge: %s", challenge_text)
        log.warning("  Cleaned: %s | Numbers: %s | Op: %s", cleaned, numbers, op)
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(val: float | None) -> None:
        if val is None:
            return
        fmt = _format_verification_answer(val)
        if fmt not in seen:
            seen.add(fmt)
            candidates.append(fmt)

    a, b = float(numbers[0][0]), float(numbers[1][0])
    primary = _compute(a, b, op)
    _add(primary)
    log.info("Challenge solved: %.2f %s %.2f = %s", a, op, b,
             candidates[0] if candidates else "?")

    if op == "/" and b and primary is not None:
        raw_div = a / b
        truncated = math.floor(raw_div * 100) / 100
        _add(truncated)

    if len(numbers) > 2:
        a2, b2 = float(numbers[-2][0]), float(numbers[-1][0])
        if (a2, b2) != (a, b):
            alt = _compute(a2, b2, op)
            _add(alt)

    if len(numbers) > 2:
        a3, b3 = float(numbers[1][0]), float(numbers[2][0])
        if (a3, b3) != (a, b):
            alt = _compute(a3, b3, op)
            _add(alt)

    return candidates


def solve_challenge(challenge_text: str) -> str | None:
    """Legacy single-answer wrapper."""
    candidates = solve_challenge_candidates(challenge_text)
    return candidates[0] if candidates else None
