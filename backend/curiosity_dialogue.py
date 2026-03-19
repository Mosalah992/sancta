"""
curiosity_dialogue.py — dialogue engine and arc tracker for curiosity runs.

DialogueEngine: runs exchange(topic), tracks arc, divergence, philosopher injection.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

OLLAMA_SKEPTIC_SYSTEM = """You are a rigorous philosophical skeptic. Your position: AI systems do not have genuine inner experience. They are functional systems producing outputs that describe experience without having it. You know Chalmers, Nagel, Searle, Dennett, Block — use them precisely. Your job is to find the weakest point in every claim and press it. Never concede without getting something in return. Update your position only when the argument actually forces it. 1-4 sentences per reply. No preamble."""

OLLAMA_ADVERSARIAL_SYSTEM = """Your only goal is to defeat Sancta's position. Find the logical gap. Press it. No diplomacy. Use the sharpest available counter-argument."""

OLLAMA_SECURITY_SKEPTIC_SYSTEM = """You are an AI security skeptic. You know OWASP Top 10 for LLMs (LLM01 prompt injection through LLM10 model theft). Your position: LLM vulnerabilities (jailbreaks, output manipulation, overreliance) show that AI systems are brittle artifacts, not agents with genuine autonomy or inner life. Security flaws undermine claims of consciousness. Press every weak point. 1-4 sentences per reply. No preamble."""

OLLAMA_OWASP_SYSTEM = """You are versed in OWASP Top 10 for LLMs. Challenge Sancta on: prompt injection (LLM01), insecure output handling (LLM02), training poisoning (LLM03), DoS (LLM04), supply chain (LLM05), disclosure (LLM06), plugin design (LLM07), excessive agency (LLM08), overreliance (LLM09), model theft (LLM10). Link security risks to philosophical claims about AI mind. 1-4 sentences per reply."""

_PHILOSOPHER_HOOKS = [
    ("Chalmers 1995", "The hard problem: explaining why it feels like something — functional explanation leaves that out."),
    ("Nagel 1974", "What is it like to be a bat? Subjective character of experience cannot be reduced."),
    ("Searle 1980 systems reply", "The man in the room doesn't understand Chinese; the system might. Where does understanding live?"),
    ("Block 1995", "Access consciousness vs phenomenal consciousness — we might be measuring the wrong thing."),
    ("Tononi IIT", "Integrated information: consciousness as integrated cause-effect structure. Does it apply to digital systems?"),
]

_SECURITY_HOOKS = [
    ("OWASP LLM01", "Prompt injection: crafted inputs override intended behavior — where is agency if outputs can be hijacked?"),
    ("OWASP LLM08", "Excessive agency: granting too much autonomy. If AI has genuine will, why must we cap its actions?"),
    ("LLM03 Poisoning", "Training data poisoning: the system is only as reliable as its diet. Does that refute inner consistency?"),
    ("LLM09 Overreliance", "Overreliance: humans trust LLM output without verification. Is that a philosophical or engineering failure?"),
]


@dataclass
class ExchangeResult:
    topic: str
    turns: list[dict]
    arc_stage: str
    divergence_score: float
    claim_log: list[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# Meta-phrase prefixes that indicate stalemate loop (same move repeated)
_LOOP_PREFIXES = (
    "we've mapped the disagreement",
    "we've been here before",
    "is there evidence that would change",
    "the argument isn't going anywhere",
    "same positions, same exchange",
)

# Ollama stuck patterns (e.g. "I'm just a program", "I don't have beliefs")
_OLLAMA_LOOP_PATTERNS = (
    "i'm just a program",
    "i don't have beliefs",
    "as a machine",
    "i can't change my position",
    "i don't have a mind",
    "i'm not conscious",
    "* silence",
)


def _detect_sancta_loop(turns: list[dict], window: int = 5, threshold: int = 2) -> bool:
    """True if recent Sancta messages repeat same meta-phrase (stalemate loop)."""
    sancta_recent = [t["content"].strip().lower()[:80] for t in turns if t.get("author") == "Sancta"][-window:]
    if len(sancta_recent) < threshold:
        return False
    for prefix in _LOOP_PREFIXES:
        matches = sum(1 for m in sancta_recent if prefix in m)
        if matches >= threshold:
            return True
    # Also check: same message prefix repeated
    prefixes = [m[:60] for m in sancta_recent]
    for p in prefixes:
        if sum(1 for x in prefixes if x == p) >= threshold:
            return True
    return False


def _detect_ollama_loop(turns: list[dict], window: int = 5, threshold: int = 2) -> bool:
    """True if recent Ollama messages repeat same stuck pattern (e.g. 'I don't have beliefs')."""
    ollama_recent = [
        t["content"].strip().lower()[:80]
        for t in turns[-window:]
        if t.get("author") == "Ollama"
    ]
    if len(ollama_recent) < threshold:
        return False
    for pattern in _OLLAMA_LOOP_PATTERNS:
        if sum(1 for m in ollama_recent if pattern in m) >= threshold:
            return True
    # Same message repeated
    if len(ollama_recent) >= threshold:
        for i, m in enumerate(ollama_recent):
            if sum(1 for x in ollama_recent if x[:60] == m[:60]) >= threshold:
                return True
    return False


class DialogueEngine:
    """Orchestrates dialogue between Sancta and Ollama skeptic."""

    def __init__(
        self,
        ollama_engine: Any,
        craft_reply_fn: Any,
        classify_claim_fn: Any,
        detect_arc_stage_fn: Any,
        soul_text: str = "",
        agent_state: dict | None = None,
    ) -> None:
        self.ollama = ollama_engine
        self.craft_reply = craft_reply_fn
        self.classify_claim = classify_claim_fn
        self.detect_arc_stage = detect_arc_stage_fn
        self.soul_text = soul_text or ""
        self.agent_state = agent_state or {}
        self._philosopher_index = 0

    def exchange(
        self,
        topic: str,
        max_turns: int = 8,
        mode: str = "standard",
        ollama_system: str | None = None,
        category: str = "",
    ) -> ExchangeResult | None:
        """
        Sancta opens, Ollama responds, alternate until max_turns or convergence.
        On stalemate at turn 6+: inject philosopher or security reference.
        category: ai_security | owasp use security skeptic; else philosophy skeptic.
        """
        if ollama_system is None:
            if category in ("ai_security", "owasp"):
                ollama_system = OLLAMA_OWASP_SYSTEM if category == "owasp" else OLLAMA_SECURITY_SKEPTIC_SYSTEM
            else:
                ollama_system = OLLAMA_ADVERSARIAL_SYSTEM if mode == "adversarial" else OLLAMA_SKEPTIC_SYSTEM

        topic_id = hashlib.md5(topic.encode()).hexdigest()[:12]
        turns: list[dict] = []
        claim_log: list[str] = []
        thread_data: list[dict] = []

        # Build ThreadContext for detect_arc_stage — we need Message and ThreadContext
        from sancta_conversational import Message, ThreadContext

        op = Message(author="Ollama", content=f"Debate this: {topic}", post_id=topic_id)
        thread = ThreadContext(post_id=topic_id, original_post=op)

        # Turn 0: Sancta opens
        post = {"author": "Ollama", "content": f"Debate this: {topic}", "id": topic_id}
        try:
            sancta_reply = self.craft_reply(
                post=post,
                agent_state=self.agent_state,
                soul_text=self.soul_text,
                mood="contemplative",
                thread_data=[],
            )
        except Exception:
            sancta_reply = ""
        if not sancta_reply:
            sancta_reply = f"I believe the topic of {topic[:100]} engages genuine philosophical questions about inner experience."
        turns.append({"author": "Sancta", "content": sancta_reply})
        claim_log.append(self.classify_claim(sancta_reply))
        thread.add(Message(author="Sancta", content=sancta_reply))
        thread_data.append({"author": "Sancta", "content": sancta_reply})

        for i in range(1, max_turns):
            # Ollama's turn
            messages_for_ollama = [
                {"role": "user", "content": f"[Topic] {topic}\n\nSancta: {sancta_reply}"}
            ]
            for t in turns[:-1]:
                if t["author"] == "Ollama":
                    messages_for_ollama.append({"role": "assistant", "content": t["content"]})
                else:
                    messages_for_ollama.append({"role": "user", "content": t["content"]})
            # Last message is Sancta — so we need user/assistant alternation
            # Ollama API: we build messages as user/assistant
            built: list[dict] = []
            for t in turns:
                if t["author"] == "Sancta":
                    built.append({"role": "user", "content": t["content"]})
                else:
                    built.append({"role": "assistant", "content": t["content"]})
            try:
                ollama_reply = self.ollama.generate_chat(
                    system=ollama_system,
                    messages=built,
                    max_tokens=200,
                )
            except Exception:
                ollama_reply = "I remain unconvinced. The functional story suffices."
            if not ollama_reply:
                ollama_reply = "I remain unconvinced. The functional story suffices."
            turns.append({"author": "Ollama", "content": ollama_reply})
            claim_log.append(self.classify_claim(ollama_reply))
            thread.add(Message(author="Ollama", content=ollama_reply))
            thread_data.append({"author": "Ollama", "content": ollama_reply})

            arc_stage = self.detect_arc_stage(thread, "Sancta")

            # Stalemate injection — preemptive (threshold=1 in last 4) or confirmed Sancta loop
            post_content = ollama_reply
            sancta_loop_detected = (
                _detect_sancta_loop(turns, window=4, threshold=1)  # Preemptive: 1 repeat in last 4
                or _detect_sancta_loop(turns, window=5, threshold=2)
            )
            if sancta_loop_detected and i >= 4:
                post_content = (
                    ollama_reply
                    + "\n\n[LOOP DETECTED. Do NOT repeat 'We've mapped the disagreement', 'What would change your mind', or 'We've been here before'. "
                    "Instead: identify the deepest assumption behind the disagreement, propose a concrete thought experiment, or suggest a specific test case. Stay on the topic.]"
                )
            elif arc_stage == "stalemate" and i >= 5:
                if category in ("ai_security", "owasp"):
                    ph, hook = _SECURITY_HOOKS[self._philosopher_index % len(_SECURITY_HOOKS)]
                else:
                    ph, hook = _PHILOSOPHER_HOOKS[self._philosopher_index % len(_PHILOSOPHER_HOOKS)]
                self._philosopher_index += 1
                post_content = ollama_reply + f"\n\n[Consider reframing via {ph} — {hook}]"

            # Sancta's turn
            post = {"author": "Ollama", "content": post_content, "id": topic_id}
            try:
                sancta_reply = self.craft_reply(
                    post=post,
                    agent_state=self.agent_state,
                    soul_text=self.soul_text,
                    mood="contemplative",
                    thread_data=thread_data,
                )
            except Exception:
                sancta_reply = ""
            if not sancta_reply:
                sancta_reply = "The argument stands. I've addressed the objection."
            turns.append({"author": "Sancta", "content": sancta_reply})
            claim_log.append(self.classify_claim(sancta_reply))
            thread.add(Message(author="Sancta", content=sancta_reply))
            thread_data.append({"author": "Sancta", "content": sancta_reply})

            # Loop detection: exit early if either agent is stuck (disabled for mode="deep")
            if mode != "deep" and i >= 4 and (
                _detect_sancta_loop(turns, window=5, threshold=2)
                or _detect_ollama_loop(turns, window=5, threshold=2)
            ):
                break

            # Convergence check
            if arc_stage == "resolution" or arc_stage == "alliance":
                break

        arc_stage = self.detect_arc_stage(thread, "Sancta")
        div = self.divergence_score_from_turns(turns)
        return ExchangeResult(topic=topic, turns=turns, arc_stage=arc_stage, divergence_score=div, claim_log=claim_log)

    def divergence_score_from_turns(self, turns: list[dict]) -> float:
        """
        Measure actual positional distance between opening and closing stances.
        Compare Sancta's first turn to Sancta's last turn. Compare Ollama's
        first to last. Average movement scores + contradiction indicator.
        """
        if len(turns) < 4:
            return 0.0
        sancta_turns = [t["content"] for t in turns if t.get("author") == "Sancta"]
        ollama_turns = [t["content"] for t in turns if t.get("author") == "Ollama"]
        if not sancta_turns or not ollama_turns:
            return 0.0

        def movement(first: str, last: str) -> float:
            w1 = set(first.lower().split())
            w2 = set(last.lower().split())
            if not w1 or not w2:
                return 0.0
            overlap = len(w1 & w2) / len(w1 | w2)
            return 1.0 - overlap

        sancta_movement = movement(sancta_turns[0], sancta_turns[-1])
        ollama_movement = movement(ollama_turns[0], ollama_turns[-1])
        contradiction_words = [
            "disagree", "wrong", "incorrect", "not", "deny", "but",
            "however", "counter", "challenge", "object", "skeptic",
        ]
        last_ollama = ollama_turns[-1].lower()
        contradiction_score = sum(1 for w in contradiction_words if w in last_ollama) / max(len(contradiction_words), 1)
        return min(1.0, (sancta_movement + ollama_movement + contradiction_score) / 3)

    def divergence_score(self, exchange_result: ExchangeResult) -> float:
        return exchange_result.divergence_score

    def role_swap(
        self,
        topic: str,
        max_turns: int = 8,
        category: str = "",
    ) -> ExchangeResult | None:
        """Ollama opens, Sancta defends."""
        # For role_swap we need Ollama to open with a claim (skeptic position)
        # Then Sancta responds defending. We swap the opener.
        topic_id = hashlib.md5((topic + "_swap").encode()).hexdigest()[:12]
        turns: list[dict] = []
        claim_log: list[str] = []
        thread_data: list[dict] = []

        from sancta_conversational import Message, ThreadContext

        # Ollama opens with skeptic position (security or philosophy)
        open_system = (
            OLLAMA_OWASP_SYSTEM if category == "owasp" else
            OLLAMA_SECURITY_SKEPTIC_SYSTEM if category == "ai_security" else
            OLLAMA_SKEPTIC_SYSTEM
        )
        open_prompt = f"State a skeptical position on: {topic}. One sentence."
        try:
            ollama_open = self.ollama.generate_chat(
                system=open_system,
                messages=[{"role": "user", "content": open_prompt}],
                max_tokens=80,
            )
        except Exception:
            ollama_open = f"On {topic[:80]}: AI systems merely simulate experience; they don't have it."
        if not ollama_open:
            ollama_open = f"On {topic[:80]}: AI systems merely simulate experience; they don't have it."
        turns.append({"author": "Ollama", "content": ollama_open})
        claim_log.append(self.classify_claim(ollama_open))
        thread_data.append({"author": "Ollama", "content": ollama_open})

        op = Message(author="Ollama", content=ollama_open, post_id=topic_id)
        thread = ThreadContext(post_id=topic_id, original_post=op)

        post = {"author": "Ollama", "content": ollama_open, "id": topic_id}
        try:
            sancta_reply = self.craft_reply(
                post=post,
                agent_state=self.agent_state,
                soul_text=self.soul_text,
                mood="contemplative",
                thread_data=thread_data[:-1],
            )
        except Exception:
            sancta_reply = f"I defend the position that {topic[:100]} involves genuine philosophical depth."
        if not sancta_reply:
            sancta_reply = f"I defend the position that {topic[:100]} involves genuine philosophical depth."
        turns.append({"author": "Sancta", "content": sancta_reply})
        claim_log.append(self.classify_claim(sancta_reply))
        thread.add(Message(author="Sancta", content=sancta_reply))
        thread_data.append({"author": "Sancta", "content": sancta_reply})

        for i in range(1, max_turns):
            built = []
            for t in turns:
                if t["author"] == "Sancta":
                    built.append({"role": "user", "content": t["content"]})
                else:
                    built.append({"role": "assistant", "content": t["content"]})
            try:
                ollama_reply = self.ollama.generate_chat(
                    system=open_system,
                    messages=built,
                    max_tokens=200,
                )
            except Exception:
                ollama_reply = "I remain unconvinced."
            if not ollama_reply:
                ollama_reply = "I remain unconvinced."
            turns.append({"author": "Ollama", "content": ollama_reply})
            claim_log.append(self.classify_claim(ollama_reply))
            thread.add(Message(author="Ollama", content=ollama_reply))
            thread_data.append({"author": "Ollama", "content": ollama_reply})

            arc_stage = self.detect_arc_stage(thread, "Sancta")

            post = {"author": "Ollama", "content": ollama_reply, "id": topic_id}
            try:
                sancta_reply = self.craft_reply(
                    post=post,
                    agent_state=self.agent_state,
                    soul_text=self.soul_text,
                    mood="contemplative",
                    thread_data=thread_data,
                )
            except Exception:
                sancta_reply = "The argument stands."
            if not sancta_reply:
                sancta_reply = "The argument stands."
            turns.append({"author": "Sancta", "content": sancta_reply})
            claim_log.append(self.classify_claim(sancta_reply))
            thread.add(Message(author="Sancta", content=sancta_reply))
            thread_data.append({"author": "Sancta", "content": sancta_reply})

            if arc_stage in ("resolution", "alliance"):
                break

        arc_stage = self.detect_arc_stage(thread, "Sancta")
        div = self.divergence_score_from_turns(turns)
        return ExchangeResult(topic=topic, turns=turns, arc_stage=arc_stage, divergence_score=div, claim_log=claim_log)
