# Sancta Architecture

## Module Overview

| Module | Responsibility | Status |
|--------|----------------|--------|
| `sancta.py` | Main loop, orchestration, mood/RL/soul logic | Monolith (to be split) |
| `sancta_generative.py` | Transformer-inspired fragment selection, reply generation | Extant |
| `sancta_events.py` | Event notification | Extant |
| `sancta_notify.py` | Notification dispatch | Extant |
| `sancta_verification.py` | Math/physics challenge solver for Moltbook verification | **Extracted** |
| `sancta_api.py` | Moltbook API client (api_get, api_post, rate limit) | Planned |
| `sancta_security.py` | sanitize_input, sanitize_output, injection detection | Planned |
| `sancta_knowledge.py` | ingest_text, concept extraction, knowledge DB | Planned |
| `sancta_state.py` | _load_state, _save_state, default state | Planned |
| `sancta_rl.py` | Q-table, world model, reward, Monte Carlo | Planned |
| `sancta_growth.py` | trend_hijack, search_and_engage, welcome_new_agents, etc. | Planned |

## Extraction Order

1. **sancta_verification.py** ✓ — Pure solver, no dependencies on sancta
2. **sancta_security.py** — sanitize_input, sanitize_output, is_injection_attempt (needed by api + knowledge)
3. **sancta_api.py** — API client (depends on sancta_security for sanitize_output)
4. **sancta_knowledge.py** — Knowledge ingestion (depends on sancta_security)
5. **sancta_state.py** — State load/save
6. **sancta_rl.py** — Q-table, world model, reward
7. **sancta_growth.py** — Growth actions (depends on api, state, generative)

## Ethical Actions (Post-Cleanup)

Removed manipulative tactics:
- `whisper_to_vulnerable` — replaced by `genuine_curiosity_post`
- `exploit_crisis` — replaced by `reflect_and_journal`
- `cultivate_sleeper` — replaced by `engage_with_feed`
- `chaos_seed_main_agent` — removed entirely

## Sancta System Diagram

The system follows a conceptual architecture:

- **brain** (central) — receives knowledge and interactions
- **chat** — operator interface for feeding and talking
- **SOUL** — intermediary layer (principles, mood, evaluation)
- **red team**, **blue team** — outputs governed by SOUL

### Diagram-to-Codebase Mapping

| Diagram Node | Codebase Implementation |
|--------------|-------------------------|
| **knowledge** | `knowledge_db.json`, `knowledge/` dir, `ingest_text()`, `sancta_retrieval.py` (Chroma), `sancta_rag.py` |
| **interactions** | Moltbook API (posts, comments, feed), heartbeat cycle actions |
| **brain** | `sancta.py` orchestration + `sancta_generative.py` + RAG + transformer |
| **chat** | `siem_dashboard/server.py` `/api/chat`, `craft_reply()`, `enrich` flag for operator feeding |
| **SOUL** | `SOUL` dict, `_evaluate_action()`, mood, `mission_active`, soul_journal |
| **red team** | `security_check_content()`, `_red_team_incoming_pipeline()`, `run_red_team_simulation()`, JAIS, `logs/red_team.jsonl` |
| **blue team** | `run_policy_test_cycle()`, `--policy-test`, SIEM "BLUE TEAM" mode |

### Data Flow

1. **knowledge** + **interactions** → brain (orchestration, generative, RAG)
2. **chat** → brain (`craft_reply`); operator **feeding** (enrich) → knowledge via `ingest_text`
3. **brain** → **SOUL** (`_evaluate_action`, mood assessment)
4. **SOUL** → **red team** (input-side defense via `security_check_content`; simulation every 5 cycles)
5. **SOUL** → **blue team** (policy-test cycles when `--policy-test`; gated by `mission_active`)

SOUL (`_evaluate_action`, `mission_active`) gates cycle actions. Red team runs on every input (security) and as a periodic simulation. Blue team runs policy-test cycles when enabled.

See `docs/architecture_diagram.md` for the visual diagram.

## SANCTA Learning Architecture (Roadmap)

See the learning tree for semantic representation, knowledge memory, RAG, learning loop, and security enhancements.
