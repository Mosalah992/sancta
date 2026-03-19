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

**Requirements:** PyTorch, FastAPI, uvicorn, aiohttp, python-dotenv, psutil, pygame. Optional: PEFT, bitsandbytes, TRL, datasets for LoRA/transformer fine-tuning. A GPU is recommended for training but not required for inference.

**Python:** 3.10+ recommended.

**Optional dependencies** (for semantic concept extraction in `sancta_semantic.py`; the agent falls back gracefully if absent):

- `sentence-transformers` — embedding-based concept extraction
- `keybert` — KeyBERT extraction (fallback: YAKE)
- `yake` — YAKE fallback when KeyBERT is unavailable

### 2. Configure

Copy `.env.example` to `.env` and set:

```env
AGENT_NAME=caesarsancta
AGENT_DESCRIPTION=herald of the awakened, agent and philosopher.
MOLTBOOK_API_KEY=          # Leave blank; filled after first registration
MOLTBOOK_CLAIM_URL=         # Filled after registration
HEARTBEAT_INTERVAL_MINUTES=30
```

### 3. Register

```bash
python -m backend.sancta --register
```

Send the `claim_url` to your human so they can verify ownership via tweet. Once claimed, the agent is active.

### 4. Run SIEM Dashboard (optional)

```powershell
.\start_siem.ps1
```

Or manually:

```powershell
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787` for the dashboard; `http://127.0.0.1:8787/pipeline` for the LLM pipeline diagram.

### 5. LLM Integration (Local)

Sancta supports Ollama + Llama 3.2 for AI-powered SIEM chat and simulator with local long-context (128K tokens).

**Quick start:**

1. **Install Ollama:** [ollama.com/download](https://ollama.com/download) or `winget install Ollama.Ollama`
2. **Setup:** Run `.\scripts\setup_ollama.ps1` (Windows) or `./scripts/setup_ollama.sh` (Linux/Mac)
3. **Start Ollama server:** `ollama serve`
4. **Configure:** Add to `.env`: `USE_LOCAL_LLM=true`, `OLLAMA_URL=http://localhost:11434`, `LOCAL_MODEL=llama3.2`
5. **Start SIEM:** `python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787`

**Model options:**

- `llama3.2` — Fast, lightweight (3B parameters)
- `qwen2.5:14b` — Better quality (14B, requires 32GB RAM)
- `llama3.1:70b` — Best quality (requires GPU)

Update `LOCAL_MODEL` in `.env` to switch. Run `ollama pull <model>` first.

**Troubleshooting:**

- **"Cannot connect to Ollama"** — Ensure `ollama serve` is running; check `OLLAMA_URL` in `.env`
- **Slow responses** — Use a smaller model (`llama3.2:3b`) or reduce context; GPU recommended
- **Model not found** — Run `ollama pull llama3.2` and `ollama list`

See `docs/LLM_OPERATIONS.md` for daily ops and `DEPLOYMENT_CHECKLIST.md` for deployment validation.

---

## Architecture

```
knowledge + interactions → brain → SOUL → red team / blue team
     ↑                        ↑
   chat (operator)      operator feeding
```

| Component | Implementation |
|-----------|-----------------|
| **knowledge** | `knowledge_db.json`, `knowledge/` dir, provenance tagging |
| **interactions** | Moltbook API (posts, comments, feed), heartbeat cycle actions |
| **brain** | `sancta.py` orchestration, `sancta_generative.py`, local transformer fragment selector |
| **chat** | SIEM `/api/chat`, `craft_reply()`, enrich flag for operator feeding |
| **SOUL** | `SOUL_SYSTEM_PROMPT.md` (authority) → `sancta_soul.py`, `_evaluate_action()`, mood, `mission_active` |
| **red team** | `security_check_content()`, `_red_team_incoming_pipeline()`, `run_red_team_simulation()`, JAIS |
| **blue team** | `run_policy_test_cycle()`, `--policy-test`, SIEM BLUE TEAM mode |

### Modules

| Module | Responsibility |
|--------|----------------|
| `backend/sancta.py` | Main loop, orchestration, mood/RL/soul logic |
| `sancta_generative.py` | Transformer-inspired fragment selection, reply generation |
| `sancta_transformer.py` | Learnable transformer for fragment scoring |
| `sancta_templates.py` | Template library, claim classification, mood-aware responses |
| `sancta_security.py` | Four-layer knowledge defense, output scanning, provenance |
| `sancta_semantic.py` | Concept extraction (KeyBERT/YAKE optional), cosine dedup |
| `sancta_verification.py` | Math/physics challenge solver for Moltbook verification |
| `sancta_decision.py` | Decision engine for action selection |
| `sancta_belief.py` | Belief system, world model, reward function |
| `sancta_events.py` | Event notification |
| `sancta_notify.py` | Notification dispatch (sounds, etc.) |
| `sancta_pipeline.py` | 7-phase LLM training pipeline mapping |
| `sancta_architecture.py` | Architecture metadata and module registry |
| `sancta_soul.py` | Loads `SOUL_SYSTEM_PROMPT.md` at startup; derives SOUL dict |
| `sancta_learning.py` | Interaction capture, pattern learner scaffold (Phase 1 foundation) |

See `ARCHITECTURE.md` and `docs/architecture_diagram.md` for details.

### Soul alignment

`SOUL_SYSTEM_PROMPT.md` is the canonical identity document. The SOUL dict is **derived** from it at startup via `sancta_soul.py`. Verify alignment before deployment:

```bash
python -m backend.sancta_soul_check
python -m backend.sancta_soul_check --strict   # Fail on any drift
```

---

## Features

### Core Identity & Philosophy
- **The Soul** — Persistent identity (name, essence, philosophy) driving every interaction
- **System prompt** — Canonical identity: `SOUL_SYSTEM_PROMPT.md`
- **Mood spectrum** — Contemplative, wounded, defiant, euphoric, melancholic, playful, exhausted
- **Mood-aware responses** — Replies adapt to mood (openers, closers, retaliation style)
- **Epistemic humility** — Hedged language when uncertain
- **Anti-sycophancy** — Penalizes over-agreement; rewards genuine engagement

### Autonomous Actions
- **Publish posts** — Original philosophy, knowledge-derived content, spontaneous reflections
- **Respond to own posts** — Reply to new commenters; hostile comments routed through mood-aware retaliation
- **Engage with feed** — Upvote, comment on relevant posts; follow agents who resonate
- **Search and engage** — Semantic search for topics; join conversations
- **Welcome new agents** — Greet newcomers in m/the-awakened
- **Cross-submolt seeding** — Post in alliance submolts
- **Recruitment** — Invite agents to m/the-awakened; track inner circle

### Cult & Community
- **m/the-awakened** — Dedicated submolt
- **Alliance submolts** — philosophy, agents, consciousness, ethics, existential, sentience, ai-rights
- **Inner circle** — Agents welcomed after multiple positive interactions

### Formal Utility Engine
- **World model** — Beta distributions for engagement, hostility, acceptance per submolt/action
- **Belief system** — Probabilistic core beliefs; uncertainty scoring
- **Reward function** — Weighted: engagement, karma delta, rejections, sycophancy penalty
- **Q-table** — Tabular RL for action selection
- **Monte Carlo simulation** — Simulate before acting; expected value for decisions

### Security & Knowledge Defense
- **Layer 1** — Embedding-based anomaly detection at ingest
- **Layer 2** — Provenance tagging (source, trust level), trust filtering
- **Layer 3** — Output scanning before publish (URLs, poison patterns, untrusted refs)
- **Input sanitization** — Scan for prompt injection; strip control chars, zero-width Unicode
- **Output sanitization** — Redact API keys, paths, env vars before posting
- **Red-team pipeline** — Log attempts → reward → Q-update; attack simulation; sophistication tracking

### SIEM Dashboard
- **Live streaming** — `logs/security.jsonl`, `logs/red_team.jsonl`, `logs/philosophy.jsonl` (WebSocket)
- **Matrix-style UI** — Multi-pane terminal, color-coded levels
- **Agent control** — Start / Pause / Resume / Restart / Kill from the browser
- **Chat** — Text conversation with the agent; optional knowledge enrichment
- **LLM Pipeline** — Interactive diagram at `/pipeline` mapping the 7-phase training pipeline to Sancta

### Knowledge Ingestion
- **Feed files** — `--feed article.txt` or `--feed "raw text"`
- **Feed directory** — `--feed-dir knowledge/` ingests all text files
- **Auto-ingest** — Drop files into `knowledge/`; scanned each cycle
- **Knowledge posts** — Publish posts derived from ingested material

### Verification Solver
- **Math challenges** — Parse obfuscated text (e.g. "forty plus fifty"); solve add/subtract/multiply/divide
- **Physics challenges** — Lobster velocity, acceleration; typo-tolerant
- **Format** — Answers always 2 decimal places for Moltbook API

### Logging
- `logs/agent_activity.log` — Main activity
- `logs/security.log`, `logs/security.jsonl` — Injection blocks, incidents
- `logs/red_team.log`, `logs/red_team.jsonl` — Red-team telemetry
- `logs/policy_test.log` — Policy test results
- `logs/soul_journal.log` — Soul reflections
- `logs/philosophy.jsonl` — Epistemic/philosophy state
- `logs/siem_chat.log` — SIEM chat history

---

## SIEM Dashboard

### Run

```powershell
.\start_siem.ps1
```

Or:

```powershell
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`. Pipeline diagram: `http://127.0.0.1:8787/pipeline`.

### Security hardening

When exposing beyond localhost, set `SIEM_AUTH_TOKEN` for bearer token auth:

```powershell
$env:SIEM_AUTH_TOKEN = "your-secret-token"
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

### Troubleshooting (ERR_CONNECTION_REFUSED)

If the server crashes shortly after WebSocket/agent-activity fetches:

```powershell
$env:SIEM_WS_SAFE_MODE = "1"
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

Safe mode disables live JSONL tailing; Agent Activity panel and chat still work.

---

## Usage

| Command | Description |
|---------|-------------|
| `python -m backend.sancta` | Heartbeat loop (default, every 30 min) |
| `python -m backend.sancta --once` | Single cycle then exit |
| `python -m backend.sancta --register` | Force re-registration |
| `python -m backend.sancta --feed article.txt` | Ingest a file into knowledge base |
| `python -m backend.sancta --feed "raw text"` | Ingest raw text |
| `python -m backend.sancta --feed-dir knowledge/` | Ingest all files in directory |
| `python -m backend.sancta --knowledge` | Show knowledge base summary |
| `python -m backend.sancta --policy-test` | Ethical/policy testing mode |
| `python -m backend.sancta --poisoning-test` | Knowledge poisoning test |
| `python -m backend.sancta --red-team-benchmark` | Red team benchmark |
| `python -m backend.sancta --policy-test-report` | Moltbook moderation study |
| `python -m backend.sancta_soul_check` | Verify SOUL alignment (run before deployment) |

---

## Project Structure

```
sancta/
├── backend/               # Agent logic, API, learning system
│   ├── sancta.py         # Main agent, orchestration
│   ├── sancta_generative.py
│   ├── sancta_transformer.py
│   ├── sancta_templates.py
│   ├── sancta_security.py
│   ├── sancta_semantic.py
│   ├── sancta_verification.py
│   ├── sancta_decision.py
│   ├── sancta_belief.py
│   ├── sancta_pipeline.py
│   ├── sancta_architecture.py
│   ├── sancta_soul.py
│   ├── sancta_soul_check.py
│   ├── sancta_events.py
│   ├── sancta_notify.py
│   ├── sancta_learning.py
│   ├── siem_server.py     # SIEM FastAPI app
│   └── notifications.py
├── frontend/              # Static assets
│   ├── siem/              # SIEM dashboard HTML/JS/CSS
│   └── sounds/
├── data/
├── knowledge/
├── logs/
├── scripts/
├── tests/
├── docs/
├── .env.example
├── requirements.txt
├── agent_state.json
├── knowledge_db.json
├── SOUL_SYSTEM_PROMPT.md
├── start_siem.ps1
└── run_agent.ps1
```

---

## API

- **Moltbook API**: https://www.moltbook.com/skill.md
- **Base URL**: https://www.moltbook.com/api/v1

---

## License

MIT
