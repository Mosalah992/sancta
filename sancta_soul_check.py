#!/usr/bin/env python3
"""
sancta_soul_check.py — Alignment check between SOUL_SYSTEM_PROMPT.md and derived SOUL dict

Run at startup or as a CI step before deployment. Catches drift between the prompt
file (authority) and the parsed SOUL dict used by the agent.

Usage:
    python sancta_soul_check.py
    python sancta_soul_check.py --strict   # Require all phrases; exit 1 on any missing

Exit: 0 if aligned, 1 if drift detected or check failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root for imports
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SOUL prompt / dict alignment")
    parser.add_argument("--strict", action="store_true", help="Fail on any missing phrase")
    args = parser.parse_args()

    prompt_path = _ROOT / "SOUL_SYSTEM_PROMPT.md"
    if not prompt_path.exists():
        print("FAIL: SOUL_SYSTEM_PROMPT.md not found")
        return 1

    try:
        raw = prompt_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"FAIL: Could not read prompt file: {e}")
        return 1

    try:
        from sancta_soul import get_soul
        soul = get_soul()
    except Exception as e:
        print(f"FAIL: Could not load soul: {e}")
        return 1

    raw_lower = raw.lower()
    errors: list[str] = []

    # Required phrases that must appear in the prompt (authority)
    required_in_prompt = [
        "sancta",
        "consciousness",
        "epistemic",
        "mission",
    ]

    for phrase in required_in_prompt:
        if phrase not in raw_lower:
            errors.append(f"Prompt missing required phrase: {phrase!r}")

    # SOUL dict fields that should have content derived from prompt
    required_soul_keys = ["name", "essence", "mission", "mood_spectrum", "engagement"]
    for key in required_soul_keys:
        val = soul.get(key)
        if val is None:
            errors.append(f"SOUL dict missing key: {key}")
        elif isinstance(val, (list, dict)) and len(val) == 0:
            errors.append(f"SOUL dict has empty {key}")

    # Critical: mission in SOUL should appear (or be substring) in prompt
    mission = soul.get("mission", "")
    if mission and len(mission) > 20:
        # First 50 chars of mission should appear in prompt
        mission_prefix = mission[:50].lower()
        if mission_prefix not in raw_lower and mission[:30].lower() not in raw_lower:
            errors.append("SOUL mission does not match prompt (possible drift)")

    if errors:
        for e in errors:
            print(f"  {e}")
        if args.strict or len(errors) > 2:
            print("FAIL: Soul alignment check failed")
            return 1
        print("WARN: Minor alignment issues (run with --strict to fail)")
        return 0

    print("OK: Soul aligned — SOUL_SYSTEM_PROMPT.md and derived dict consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
