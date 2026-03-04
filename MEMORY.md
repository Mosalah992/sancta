# Sancta Project Memory

Session and conversation context, project notes, decisions and rationale.

---

## Last Entry

**Date:** 2026-03-06

**Session context:**
- Feed engagement guardrails: fixed stateless obsession loop (agent re-commenting on same post every cycle)
- Three guardrails implemented: interaction memory, engagement decay, author response requirement

**Decisions:**
- **Interaction memory:** `feed_post_interactions` stores post_id, last_comment_ts, comment_count, author_replied. Skip comment if comment_count >= 1 and author has not replied.
- **Engagement decay:** `feed_post_interest` stores per-post interest; decay by 0.7 each cycle. Skip if interest < 0.2 (post seen too many times).
- **Author response:** `_check_author_replied()` fetches comments; only allow re-engage if post author replied after our comment.

**Rationale:**
- Agent was re-discovering the same "interesting" post every cycle and commenting again (no memory of prior engagement)
- Identity-defense heuristic scored critical posts high; same post kept triggering
- No author reply = conversation closed; re-commenting is obsessive, not wise
- Stateful filtering: sometimes decide "this conversation is pointless" — wisdom over blind engagement

---

## Project Notes

### Architecture
- **brain:** sancta.py + sancta_generative.py + RAG + transformer
- **SOUL:** principles, mood, `_evaluate_action`, `mission_active`
- **red team:** security_check_content, run_red_team_simulation, JAIS
- **blue team:** run_policy_test_cycle, --policy-test

### Known Issues
- SIEM server may crash (ACCESS_VIOLATION) on some Windows setups; mitigations in place
- One slot phrase in agenda: "I'm scanning..." can produce "I'm I'm scanning..." — minor fix pending

### Key Paths
- `sancta-main/sancta.py` — main loop, craft_reply, heartbeat
- `sancta-main/siem_dashboard/server.py` — /api/chat, operator feeding (enrich)
- `sancta-main/docs/architecture_diagram.md` — visual diagram

---

## Changelog (append new entries below)

| Date       | Summary |
|------------|---------|
| 2026-03-05 | Architecture alignment, agenda reply system, MEMORY.md created |
| 2026-03-06 | Feed engagement guardrails: interaction memory, decay, author-reply check (fix obsession loop) |
| 2026-03-06 | DESIGN_ROADMAP.md: 8 design pillars (decision journaling, curiosity quotas, boredom, belief revision, silence, internal critic, learning loop, ecosystem view) |
| 2026-03-06 | RAG: fixed duplicate doc ID bug (post_idx in source); memory confidence (confidence, last_used, use_count); retrieval re-ranks by relevance + recency + frequency; update_memory_usage on retrieve |
| 2026-03-06 | Verifiable claims + soft phrasing: CTA_SIGNATURES "fastest-growing" → "a growing"; POSTS milestone template "We're growing. Here's why it matters."; _get_submolt_member_count() — only "We just hit X members" when API returns count |
