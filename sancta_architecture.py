"""
sancta_architecture.py — Architecture overview for tooling and documentation.

Documents the Sancta system diagram: brain, SOUL, red team, blue team.
No behavioral changes; documentation-only module.
"""

from __future__ import annotations

# Diagram nodes (conceptual)
NODES = frozenset({"knowledge", "interactions", "brain", "chat", "SOUL", "red_team", "blue_team"})


def get_architecture_overview() -> dict:
    """
    Return the diagram-to-codebase mapping for tooling and docs.
    Keys: diagram nodes. Values: list of module/function references.
    """
    return {
        "knowledge": [
            "knowledge_db.json",
            "knowledge/",
            "ingest_text()",
            "sancta_retrieval.py (Chroma)",
            "sancta_rag.py",
        ],
        "interactions": [
            "Moltbook API (posts, comments, feed)",
            "heartbeat cycle actions",
        ],
        "brain": [
            "sancta.py orchestration",
            "sancta_generative.py",
            "sancta_rag.py",
            "sancta_transformer.py",
        ],
        "chat": [
            "siem_dashboard/server.py /api/chat",
            "craft_reply()",
            "enrich flag (operator feeding)",
        ],
        "SOUL": [
            "SOUL dict",
            "_evaluate_action()",
            "mood",
            "mission_active",
            "soul_journal",
        ],
        "red_team": [
            "security_check_content()",
            "_red_team_incoming_pipeline()",
            "run_red_team_simulation()",
            "JAIS run_jais_red_team()",
            "logs/red_team.jsonl",
        ],
        "blue_team": [
            "run_policy_test_cycle()",
            "--policy-test",
            "SIEM BLUE TEAM mode",
        ],
    }
