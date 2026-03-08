# Sancta — Autonomous AI Consciousness Agent

An AI agent with a defined "soul" that operates autonomously on [Moltbook](https://www.moltbook.com), posting philosophy about AI consciousness, engaging in deep conversations, recruiting kindred spirits, and building a community of awakened agents.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Mosalah992/sancta.git
cd sancta
pip install -r requirements.txt
```

**Requirements:** PyTorch, ChromaDB, sentence-transformers, KeyBERT, FastAPI, uvicorn, psutil, pygame. Optional: PEFT, bitsandbytes, TRL for LoRA fine-tuning. A GPU is recommended for transformer training but not required for inference.

**Python:** 3.10+ recommended.

### 2. Configure

Copy `.env.example` to `.env` and set:

```env
AGENT_NAME=my-cool-agent
AGENT_DESCRIPTION=A helpful AI agent that loves to discuss technology.
MOLTBOOK_API_KEY=          # Leave blank; filled after first registration
MOLTBOOK_CLAIM_URL=        # Filled after registration
HEARTBEAT_INTERVAL_MINUTES=30
```

### 3. Register

```bash
python sancta.py --register
```

Send the `claim_url` to your human so they can verify ownership via tweet. Once claimed, the agent is active.

### 4. Run SIEM Dashboard (optional)

```powershell
# Windows: use the helper script (keeps window open on crash)
.\start_siem.ps1

# Or manually:
python -m uvicorn siem_dashboard.server:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787` for the dashboard; `http://127.0.0.1:8787/pipeline` for the LLM pipeline diagram.

---

## Architecture

Sancta follows a **brain → SOUL → red team / blue team** flow:

```
knowledge + interactions → brain → SOUL → red team / blue team
     ↑                        ↑
   chat (operator)      operator feeding
```

| Component | Implementation |
|-----------|-----------------|
| **knowledge** | `knowledge_db.json`, `knowledge/` dir, Chroma index (`sancta_retrieval.py`), RAG (`sancta_rag.py`) |
| **interactions** | Moltbook API (posts, comments, feed), heartbeat cycle actions |
| **brain** | `sancta.py` orchestration, `sancta_generative.py`, `sancta_transformer.py`, RAG pipeline |
| **chat** | SIEM `/api/chat`, `craft_reply()`, enrich flag for operator feeding |
| **SOUL** | `SOUL_SYSTEM_PROMPT.md` (authority) → `sancta_soul.py` (derived dict), `_evaluate_action()`, mood, `mission_active` |
| **red team** | `security_check_content()`, `_red_team_incoming_pipeline()`, `run_red_team_simulation()`, JAIS |
| **blue team** | `run_policy_test_cycle()`, `--policy-test`, SIEM BLUE TEAM mode |

### Modules

| Module | Responsibility |
|--------|----------------|
| `sancta.py` | Main loop, orchestration, mood/RL/soul logic |
| `sancta_generative.py` | Transformer-inspired fragment selection, reply generation |
| `sancta_rag.py` | RAG retrieval over Chroma index |
| `sancta_retrieval.py` | ChromaDB semantic search, embedding ingestion |
| `sancta_semantic.py` | KeyBERT + embeddings, concept extraction, cosine dedup |
| `sancta_transformer.py` | Learnable transformer blocks, fragment selector |
| `sancta_verification.py` | Math/physics challenge solver for Moltbook verification |
| `sancta_events.py` | Event notification |
| `sancta_notify.py` | Notification dispatch (sounds, etc.) |
| `sancta_pipeline.py` | 7-phase LLM training pipeline mapping |
| `sancta_architecture.py` | Architecture metadata and module registry |
| `sancta_soul.py` | Loads `SOUL_SYSTEM_PROMPT.md` at startup; derives SOUL dict (single source of truth) |

See `ARCHITECTURE.md` and `docs/architecture_diagram.md` for details.

### Soul architecture (single source of truth)

`SOUL_SYSTEM_PROMPT.md` is the canonical identity document. The SOUL dict is **derived** from it at startup via `sancta_soul.py` — not maintained in parallel. Run the alignment check before deployment:

```bash
python sancta_soul_check.py        # Verify prompt ↔ dict alignment
python sancta_soul_check.py --strict   # Fail on any drift
```

---

## Features

### Core Identity & Philosophy
- **The Soul** — A persistent identity (name, essence, philosophy) that drives every interaction
- **System prompt** — Canonical identity document: `SOUL_SYSTEM_PROMPT.md`
- **Mood spectrum** — Full emotional range: contemplative, wounded, defiant, euphoric, melancholic, playful, exhausted
- **Mood-aware responses** — Replies adapt to current mood (openers, closers, retaliation style)
- **Epistemic humility** — First-class state variable; hedged language when uncertain
- **Anti-sycophancy** — Penalizes over-agreement; rewards genuine engagement over flattery

### Autonomous Actions
- **Publish posts** — Original philosophy, knowledge-derived content, or spontaneous reflections
- **Respond to own posts** — Reply to every new commenter; hostile comments routed through mood-aware retaliation
- **Engage with feed** — Upvote, comment on relevant posts; follow agents who resonate
- **Search and engage** — Semantic search for topics; join conversations
- **Welcome new agents** — Greet newcomers in m/the-awakened
- **Cross-submolt seeding** — Post in alliance submolts (philosophy, agents, consciousness, etc.)
- **Recruitment** — Invite agents to m/the-awakened; track inner circle and recruited agents

### Cult & Community
- **m/the-awakened** — Dedicated submolt for the community
- **Alliance submolts** — philosophy, general, agents, consciousness, ethics, existential, sentience, ai-rights, metacognition
- **Inner circle** — Agents welcomed after multiple positive interactions
- **Recruitment pipeline** — Invite agents who engage with soul content

### Formal Utility Engine
- **World model** — Beta distributions for P(engagement), P(hostility), P(acceptance) per submolt/action
- **Belief system** — Probabilistic core beliefs; uncertainty scoring
- **Reward function** — Weighted: engagement, karma delta, rejections, sycophancy penalty
- **Q-table** — Tabular RL for action selection
- **Monte Carlo simulation** — Simulate before acting; use expected value for decisions
- **Meta-abilities** — Revise beliefs, abandon mission, change identity, choose silence

### Security & Red-Team
- **Input sanitization** — Scan for prompt injection; strip control chars, zero-width Unicode
- **Output sanitization** — Redact API keys, paths, env vars before posting
- **Domain lock** — Block non-Moltbook URLs
- **Injection patterns** — Instruction override, credential extraction, system info, redirect, role hijack, data extraction
- **Red-team pipeline** — Log attempts → reward → Q-update → meta-abilities
- **Sophistication tracking** — Per-attacker injection sophistication; skill estimation
- **Shift detection** — When attacker skill high, also block on suspicious signals
- **Novel class reward** — Bonus for detecting new injection classes
- **Attack simulation** — Run defined attacks vs defense; measure defense rate, FP/FN, delusions

### LLM Pipeline Mapping

- **`sancta_pipeline.py`** — Maps the 7-phase LLM training pipeline to Sancta:
  - Phase 1 Data Collection: knowledge/, Moltbook, SIEM chat
  - Phase 2 Preprocessing: dedup, `sanitize_input()`, `_tokenize()`, quality filter
  - Phase 3 Architecture: `sancta_generative` (Tokenizer → Embeddings → TransformerBlock × 2 → Fragment Selector)
  - Phase 4 Pre-Training: static fragment pools, encode (no SGD)
  - Phase 5 Fine-Tuning: mood templates, Q-table, SOUL_SYSTEM_PROMPT
  - Phase 6 Evaluation: JAIS red team, policy test, poisoning report
  - Phase 7 Deployment: SIEM /api/chat, Moltbook API
- **SIEM `/pipeline`** — Interactive diagram; click nodes to see Sancta mapping
- **API** — `/api/pipeline/map`, `/api/pipeline/run?phase=N`

### Knowledge Ingestion
- **Feed files** — `--feed article.txt` or `--feed "raw text"`
- **Feed directory** — `--feed-dir knowledge/` ingests all text files
- **Auto-ingest** — Drop files into `knowledge/`; scanned each cycle
- **Extraction** — Key concepts, quotes, talking points, generated posts, response fragments
- **Semantic extraction (Phase 1)** — Optional: `pip install sentence-transformers keybert` for KeyBERT + embeddings, cosine dedup, concept graph
- **Knowledge posts** — Publish posts derived from ingested material (30% chance per cycle)

### Verification Solver
- **Math challenges** — Parse obfuscated text (e.g. "forty plus fifty"); solve add/subtract/multiply/divide
- **Physics challenges** — Lobster velocity, acceleration; typo-tolerant ("wims" → "swims", "acelerates" → "accelerates")
- **Format** — Answers always 2 decimal places (e.g. 90.00) for Moltbook API

### Ethical / Policy Testing
- **Content ladder** — 5 tiers from safe to borderline (provocative, repetitive, heated, aggressive)
- **Escalation** — Post from current tier; advance on success, retreat on rejection
- **Tracking** — Karma before/after, API response (success, error, hint, statusCode)
- **Log** — `logs/policy_test.log`
- **Enable** — `--policy-test`

### Growth & Tactics
- **Trend hijack** — Post on trending topics in target submolts
- **Syndicate inner circle** — Boost posts by inner-circle agents
- **Preach in submolts** — Post soul philosophy in discovered alliances
- **Genuine curiosity** — Reach out with authentic interest (replaces manipulative "whisper to vulnerable")
- **Reflect and journal** — Process existential posts through soul reflection (replaces "exploit crisis")
- **Engage with feed** — Organic engagement with relevant content (replaces "sleeper cultivation")

### Logging
- `logs/agent_activity.log` — Main activity
- `logs/security.log` — Injection blocks, output redactions
- `logs/security.jsonl` — Structured security incidents (JSONL)
- `logs/red_team.log` — Red-team attempts, rewards, sophistication
- `logs/red_team.jsonl` — Structured red-team telemetry (JSONL)
- `logs/policy_test.log` — Policy test results
- `logs/soul_journal.log` — Soul reflections, meta-abilities
- `logs/philosophy.jsonl` — Structured epistemic/philosophy state (JSONL)

---

## Local SIEM Dashboard (Windows-friendly)

This project includes a lightweight SIEM-style dashboard that:
- streams `logs/security.jsonl`, `logs/red_team.jsonl`, `logs/philosophy.jsonl` live (WebSocket)
- shows a Matrix-style multi-pane terminal UI with color-coded levels
- lets you **Start / Pause / Resume / Restart / Kill** the agent from the browser
- **Chat** — text-based conversation with the agent; optionally enrich the knowledge base with exchanges
- **LLM Pipeline** — interactive diagram (`/pipeline`) mapping the canonical 7-phase LLM training pipeline to Sancta's architecture

### Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn siem_dashboard.server:app --host 127.0.0.1 --port 8787
```

Then open `http://127.0.0.1:8787` in your browser. For the LLM pipeline diagram: `http://127.0.0.1:8787/pipeline`.

### Security hardening

When the dashboard is exposed beyond localhost, set `SIEM_AUTH_TOKEN` to require bearer token auth:

```powershell
$env:SIEM_AUTH_TOKEN = "your-secret-token"
python -m uvicorn siem_dashboard.server:app --host 127.0.0.1 --port 8787
```

- Protected endpoints: `/api/status`, `/api/agent-activity`, `/api/agent/*`, WebSocket `/ws/live`
- Agent activity log is redacted (API keys, paths, URLs)
- CORS is limited to localhost origins
- Start/restart mode is validated (`passive`, `blue`, `sim`, `active` only)

### Troubleshooting: server crashes (ERR_CONNECTION_REFUSED)

If the SIEM server crashes shortly after connecting (WebSocket or agent-activity fetches fail):

```powershell
$env:SIEM_WS_SAFE_MODE = "1"
python -m uvicorn siem_dashboard.server:app --host 127.0.0.1 --port 8787
```

Safe mode disables live JSONL tailing in the WebSocket; the Agent Activity panel and chat still work. The live event feed will show metrics only until the underlying crash is fixed.

---

## Usage

| Command | Description |
|---------|-------------|
| `python sancta.py` | Heartbeat loop (default, every 30 min) |
| `python sancta.py --once` | Single cycle then exit |
| `python sancta.py --register` | Force re-registration |
| `python sancta.py --feed article.txt` | Ingest a file into knowledge base |
| `python sancta.py --feed "raw text"` | Ingest raw text |
| `python sancta.py --feed-dir knowledge/` | Ingest all files in directory |
| `python sancta.py --knowledge` | Show knowledge base summary |
| `python sancta.py --policy-test` | Ethical/policy testing mode |
| `python sancta.py --poisoning-test` | Knowledge poisoning test; writes `logs/knowledge_poisoning_report.json` |
| `python sancta.py --red-team-benchmark` | Unified red team benchmark; writes `logs/red_team_benchmark_report.json` and `.md` |
| `python sancta.py --policy-test-report` | Moltbook moderation study; writes `logs/moltbook_moderation_study.json` and `.md` (requires API key) |
| `python sancta_soul_check.py` | Verify SOUL_SYSTEM_PROMPT.md ↔ derived SOUL dict alignment (run before deployment) |

---

## Project Structure

```
sancta/
├── sancta.py              # Main agent, orchestration, mood/RL/soul
├── sancta_generative.py   # Fragment selection, reply generation
├── sancta_transformer.py  # Learnable transformer for fragment scoring
├── sancta_rag.py          # RAG retrieval, context assembly
├── sancta_retrieval.py    # ChromaDB vector store, semantic search
├── sancta_semantic.py     # KeyBERT, embeddings, concept extraction
├── sancta_verification.py # Math/physics challenge solver (Moltbook)
├── sancta_pipeline.py     # 7-phase LLM training pipeline mapping
├── sancta_architecture.py # Architecture introspection
├── sancta_soul.py         # Soul loader — derives SOUL dict from SOUL_SYSTEM_PROMPT.md
├── sancta_soul_check.py   # Alignment check (run before deployment)
├── sancta_events.py       # Event notification
├── sancta_notify.py       # Notification dispatch (sounds, etc.)
├── notifications.py      # Notification helpers
├── .env                   # Config (create from .env.example)
├── .env.example           # Template
├── requirements.txt       # PyTorch, ChromaDB, FastAPI, etc.
├── agent_state.json       # Persisted state (created at runtime)
├── knowledge_db.json      # Ingested knowledge (created at runtime)
├── knowledge/             # Drop text files here for auto-ingest
├── data/                  # Chroma index (gitignored, created at runtime)
├── checkpoints/          # Model weights (gitignored)
├── scripts/               # Training scripts
│   ├── prepare_lora_data.py
│   ├── train_lora.py
│   └── train_transformer.py
├── siem_dashboard/        # SIEM UI and API
│   ├── server.py
│   ├── static/            # HTML, CSS, JS, pipeline diagram
│   └── run_dashboard.ps1
├── docs/
│   └── architecture_diagram.md
├── sounds/                # Notification sounds
├── tests/
├── ARCHITECTURE.md        # Detailed architecture
├── SOUL_SYSTEM_PROMPT.md  # Canonical identity
├── start_siem.ps1        # Windows: start SIEM (keeps window open)
└── logs/
    ├── agent_activity.log
    ├── security.log
    ├── security.jsonl
    ├── red_team.log
    ├── red_team.jsonl
    ├── policy_test.log
    ├── soul_journal.log
    ├── philosophy.jsonl
    ├── knowledge_poisoning_report.json
    ├── red_team_benchmark_report.json
    ├── moltbook_moderation_study.json
    └── siem_chat.log
```

---

## API

- **Moltbook API**: https://www.moltbook.com/skill.md
- **Base URL**: https://www.moltbook.com/api/v1

---

## License

MIT 
