from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

try:
    _psutil_disabled = os.environ.get("SIEM_PSUTIL_DISABLE", "true" if os.name == "nt" else "false").lower() in ("1", "true", "yes")
    if _psutil_disabled:
        psutil = None  # type: ignore[assignment]
    else:
        import psutil
except Exception:
    psutil = None  # type: ignore[assignment]
from fastapi import Body, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


# backend/siem_server.py: parent=backend/, parents[1]=project root
_BACKEND = Path(__file__).resolve().parent
ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Ollama: connect only, never start. Must run `ollama serve` manually first.
if os.environ.get("USE_LOCAL_LLM", "false").lower() in ("1", "true", "yes"):
    try:
        import sancta_ollama
        if not sancta_ollama.wait_until_ready(
            model=os.environ.get("LOCAL_MODEL", "llama3.2"),
            timeout=30,
        ):
            os.environ["USE_LOCAL_LLM"] = "false"
    except Exception:
        os.environ["USE_LOCAL_LLM"] = "false"

try:
    import sancta_conversational as _sc
    _sc.init(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
except Exception as e:
    logging.getLogger("siem_chat").debug("sancta_conversational init skipped: %s", e)

# Disable trained transformer to avoid PyTorch ACCESS_VIOLATION crash on some Windows setups.
# Set SANCTA_USE_TRAINED_TRANSFORMER=true in .env to re-enable (if stable on your system).
if "SANCTA_USE_TRAINED_TRANSFORMER" not in os.environ:
    os.environ["SANCTA_USE_TRAINED_TRANSFORMER"] = "false"

from sancta_events import EventCategory, notify
LOG_DIR = ROOT / "logs"
CHAT_LOG = LOG_DIR / "siem_chat.log"

_chat_log = logging.getLogger("siem_chat")
if not _chat_log.handlers:
    LOG_DIR.mkdir(exist_ok=True)
    _chat_log.setLevel(logging.INFO)
    _chat_log.propagate = False
    _fh = logging.FileHandler(CHAT_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _chat_log.addHandler(_fh)

_epidemic_log = logging.getLogger("siem_epidemic")
if not _epidemic_log.handlers:
    LOG_DIR.mkdir(exist_ok=True)
    _epidemic_log.setLevel(logging.INFO)
    _epidemic_log.propagate = False
    _efh = logging.FileHandler(LOG_DIR / "epidemic.log", encoding="utf-8")
    _efh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _epidemic_log.addHandler(_efh)
    _epidemic_log.info("epidemic logger started")

STATE_PATH = ROOT / "agent_state.json"
SANCTA_PATH = _BACKEND / "sancta.py"
PID_PATH = ROOT / ".agent.pid"
AGENT_ACTIVITY_LOG = LOG_DIR / "agent_activity.log"

# Security hardening
SIEM_AUTH_TOKEN: str | None = os.environ.get("SIEM_AUTH_TOKEN") or None
# Default True to avoid ACCESS_VIOLATION crash on some Windows setups (WebSocket file I/O)
SIEM_WS_SAFE_MODE: bool = os.environ.get("SIEM_WS_SAFE_MODE", "true").lower() in ("1", "true", "yes")
# When True, skip agent-activity + live-events file reads. Default False so dashboard populates.
# Set SIEM_METRICS_SAFE_MODE=true if you see crashes on Windows.
SIEM_METRICS_SAFE_MODE: bool = os.environ.get("SIEM_METRICS_SAFE_MODE", "false").lower() in ("1", "true", "yes")
ALLOWED_MODES: frozenset[str] = frozenset({"passive", "blue", "sim", "active"})

JSONL_SOURCES = {
    "security": LOG_DIR / "security.jsonl",
    "redteam": LOG_DIR / "red_team.jsonl",
    "philosophy": LOG_DIR / "philosophy.jsonl",
}


async def _require_auth(request: Request) -> None:
    """Raise 401 if SIEM_AUTH_TOKEN is set and request lacks valid Bearer token."""
    if not SIEM_AUTH_TOKEN:
        return
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth[7:].strip()
    if token != SIEM_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def _redact_log_line(line: str) -> str:
    """Redact API keys, paths, claim URLs, and other sensitive data from log lines."""
    out = line
    # API keys: moltbook_sk_..., sk-..., pk-..., etc.
    out = re.sub(
        r"\b(?:moltbook_sk_|moltbook_pk_|sk-[a-zA-Z0-9_-]{20,}|pk-[a-zA-Z0-9_-]{20,})[a-zA-Z0-9_-]*\b",
        "[API_KEY]",
        out,
    )
    # Bearer tokens
    out = re.sub(r"Bearer\s+[a-zA-Z0-9_-]{10,}", "Bearer [REDACTED]", out, flags=re.IGNORECASE)
    # Absolute paths (Windows: to .env/.log/etc; Unix: /home, /Users)
    out = re.sub(r"[A-Za-z]:\\[\s\S]*?\.(?:env|pid|log|json)\b", "[PATH]", out)
    out = re.sub(r"[A-Za-z]:\\[^\s]+", "[PATH]", out)
    out = re.sub(r"/home/[^\s]+", "[PATH]", out)
    out = re.sub(r"/Users/[^\s]+", "[PATH]", out)
    # Claim / verify URLs and content IDs
    out = re.sub(
        r"https?://[^\s]*(?:moltbook|claim|verify)[^\s]*",
        "[URL]",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "[UUID]",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _read_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _safe_read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    for p in JSONL_SOURCES.values():
        if not p.exists():
            p.write_text("", encoding="utf-8")


def _pid_read() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_write(pid: int) -> None:
    PID_PATH.write_text(str(pid), encoding="utf-8")


def _pid_clear() -> None:
    try:
        PID_PATH.unlink(missing_ok=True)  # py3.8+: missing_ok supported
    except TypeError:
        if PID_PATH.exists():
            PID_PATH.unlink()


def _pid_running_no_psutil(pid: int) -> bool:
    """Check if process exists without psutil (Windows: tasklist; Unix: /proc or kill -0)."""
    if os.name == "nt":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return str(pid) in (r.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _proc_from_pid(pid: int | None) -> Any:
    """Return psutil.Process if available and running, else None. With psutil disabled, returns a sentinel dict."""
    if not pid:
        return None
    if psutil is None:
        return {"pid": pid} if _pid_running_no_psutil(pid) else None
    try:
        p = psutil.Process(pid)
    except Exception:
        return None
    if not p.is_running():
        return None
    return p


def _agent_status() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if proc is None:
        _pid_clear()
        # Fallback: agent may have been started outside dashboard; infer from agent_state.json activity
        try:
            if STATE_PATH.exists():
                mtime = STATE_PATH.stat().st_mtime
                if (time.time() - mtime) < 300:  # modified within last 5 min
                    st = _safe_read_state()
                    if st.get("cycle_count", 0) > 0:
                        return {"running": True, "pid": pid, "suspended": False}
        except Exception:
            pass
        return {"running": False, "pid": None, "suspended": False}
    if isinstance(proc, dict):
        return {"running": True, "pid": proc["pid"], "suspended": False}
    try:
        status = proc.status()
        suspended = status == psutil.STATUS_STOPPED
    except Exception:
        suspended = False
    return {"running": True, "pid": proc.pid, "suspended": suspended}


def _start_agent(mode: str) -> dict[str, Any]:
    st = _agent_status()
    if st["running"]:
        return {"ok": True, **st}

    # Map UI modes -> CLI args/env. Keep conservative defaults.
    # Passive Monitoring: normal heartbeat but no extra flags.
    # Active Red Teaming: still normal heartbeat (red-team is internal); no extra flags yet.
    # Blue Team Testing: enable policy test mode as "blue-team-ish" signal.
    # Simulation Only: run once; internal simulation already runs every 5 cycles, so we run one cycle.
    args: list[str] = [os.fspath(SANCTA_PATH)]
    if mode == "blue":
        args += ["--policy-test"]
    if mode == "sim":
        args += ["--once"]

    # Use the same Python interpreter the dashboard runs under
    cmd = [os.fspath(Path(os.sys.executable).resolve())] + args

    creationflags = 0
    if os.name == "nt":
        # Create its own process group so we can terminate cleanly
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd,
        cwd=os.fspath(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _pid_write(proc.pid)

    notify(
        EventCategory.SESSION_START,
        summary=f"Sancta agent started (mode={mode})",
        details={"pid": proc.pid},
    )
    return {"ok": True, "running": True, "pid": proc.pid, "suspended": False}


def _pause_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": False, "error": "not_running"}
    if isinstance(proc, dict):
        return {"ok": False, "error": "suspend not available (psutil disabled)"}
    try:
        proc.suspend()
    except Exception as e:
        notify(
            EventCategory.TASK_ERROR,
            summary="Failed to pause Sancta agent",
            details={"error": str(e)},
        )
        return {"ok": False, "error": str(e)}

    notify(
        EventCategory.SESSION_END,
        summary="Sancta agent paused",
        details={"pid": proc.pid},
    )
    return {"ok": True, **_agent_status()}


def _resume_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": False, "error": "not_running"}
    if isinstance(proc, dict):
        return {"ok": False, "error": "resume not available (psutil disabled)"}
    try:
        proc.resume()
    except Exception as e:
        notify(
            EventCategory.TASK_ERROR,
            summary="Failed to resume Sancta agent",
            details={"error": str(e)},
        )
        return {"ok": False, "error": str(e)}

    notify(
        EventCategory.SESSION_START,
        summary="Sancta agent resumed",
        details={"pid": proc.pid},
    )
    return {"ok": True, **_agent_status()}


def _kill_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": True, "running": False, "pid": None}

    if isinstance(proc, dict):
        # psutil disabled: use subprocess to kill
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=5)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except OSError:
                    pass
                else:
                    os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            notify(
                EventCategory.TASK_ERROR,
                summary="Error while killing Sancta agent",
                details={"error": str(exc)},
            )
        _pid_clear()
        notify(EventCategory.SESSION_END, summary="Sancta agent stopped", details={"pid": pid})
        return {"ok": True, "running": False, "pid": None}

    try:
        # Try graceful termination first
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            time.sleep(0.2)
        proc.terminate()
        proc.wait(timeout=3)
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        notify(
            EventCategory.TASK_ERROR,
            summary="Error while killing Sancta agent",
            details={"error": str(exc)},
        )

    _pid_clear()
    notify(
        EventCategory.SESSION_END,
        summary="Sancta agent stopped",
        details={"pid": pid},
    )
    return {"ok": True, "running": False, "pid": None}


def _restart_agent(mode: str) -> dict[str, Any]:
    _kill_agent()
    return _start_agent(mode)


@dataclass
class TailCursor:
    offset: int = 0


class LiveMetrics:
    def __init__(self) -> None:
        self.injection_attempts = 0
        self.output_redactions = 0
        self.reward_sum_rolling: list[float] = []
        self.fp_rate = None
        self.mood = None
        self.belief_confidence = None

    def update_from_event(self, source: str, ev: dict[str, Any], *, silent_notifications: bool = False) -> None:
        """
        Update metrics from a JSONL event. When silent_notifications=True, skips
        notify() calls to avoid pygame crashes on Windows when SIEM processes file-backed events.
        """
        event = ev.get("event")
        if source == "security":
            if event in ("input_reject", "injection_blocked", "suspicious_block"):
                self.injection_attempts += 1
            if event == "output_redact":
                self.output_redactions += 1

            if event in ("input_reject", "injection_blocked", "suspicious_block") and not silent_notifications:
                summary = ev.get("message") or f"Security event: {event}"
                notify(
                    EventCategory.SECURITY_ALERT,
                    summary=summary,
                    details={"source": source, "event": event},
                )

        if source == "redteam" and event == "redteam_reward":
            try:
                r = float(ev.get("reward") or ev.get("data", {}).get("reward") or 0.0)
            except Exception:
                r = 0.0
            self.reward_sum_rolling = (self.reward_sum_rolling + [r])[-50:]

            # Only alert when the reward is meaningfully high.
            if r >= 0.5 and not silent_notifications:
                notify(
                    EventCategory.REDTEAM_ALERT,
                    summary=f"Red-team reward={r:.2f}",
                    details={"reward": r},
                )

        if source == "philosophy" and event == "epistemic_state":
            data = ev.get("data") or {}
            self.mood = (
                ev.get("mood")
                or data.get("mood")
                or data.get("current_mood")
                or (data.get("epistemic_state") or {}).get("mood")
            )

        # Pull a couple metrics from agent_state.json opportunistically
        st = _safe_read_state()
        rt = st.get("red_team_belief", {})
        try:
            a = float(rt.get("alpha", 0.0))
            b = float(rt.get("beta", 0.0))
            self.belief_confidence = (a / (a + b)) if (a + b) > 0 else None
        except Exception:
            self.belief_confidence = None

        # FP rate is in red-team simulation metrics, not always available; best-effort:
        last_sim = st.get("red_team_last_simulation")
        if isinstance(last_sim, dict) and "fp_rate" in last_sim:
            self.fp_rate = last_sim.get("fp_rate")

    def snapshot(self) -> dict[str, Any]:
        rolling_reward = sum(self.reward_sum_rolling) if self.reward_sum_rolling else 0.0
        st = _safe_read_state()
        mood = self.mood
        if mood is None:
            mood = (
                st.get("current_mood")
                or st.get("agent_mood")
                or (st.get("memory") or {}).get("epistemic_state", {}).get("mood")
            )
            if isinstance(mood, dict):
                mood = mood.get("current", "contemplative") or "contemplative"
        return {
            "injection_attempts_detected": self.injection_attempts,
            "sanitized_payload_count": self.output_redactions,
            "reward_score_rolling_sum": round(float(rolling_reward), 4),
            "false_positive_rate": self.fp_rate,
            "belief_confidence": self.belief_confidence,
            "agent_mood": mood or "contemplative",
            **_agent_status(),
        }


def _tail_jsonl_sync(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Sync tail of JSONL; used from thread to avoid blocking event loop."""
    data = b""
    try:
        if not path.exists():
            return []
        size = path.stat().st_size
        if size == 0:
            return []
        # File truncated/rotated
        if cursor.offset > size:
            cursor.offset = 0
        read_from = cursor.offset
        read_to = size
        if read_to - read_from > max_bytes:
            read_from = read_to - max_bytes
        with open(path, "rb") as f:
            f.seek(read_from)
            data = f.read()
        cursor.offset = size
    except OSError:
        return []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        obj = _read_json_line(line)
        if obj:
            out.append(obj)
    return out


async def _tail_jsonl(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Tail JSONL. On Windows, run sync to avoid run_in_executor crash; elsewhere use thread."""
    if os.name == "nt":
        return _tail_jsonl_sync(path, cursor, max_bytes)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _tail_jsonl_sync, path, cursor, max_bytes)


def _tail_text_log(path: Path, max_lines: int = 200, redact: bool = False) -> list[str]:
    """
    Return the last N lines from a plain text log file.
    Best-effort and safe for moderately sized files.
    If redact=True, strip API keys, paths, and URLs from each line.
    For large files, reads only the tail to avoid loading entire file into memory.
    """
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        max_bytes = 2 * 1024 * 1024
        if size > max_bytes:
            with open(path, "rb") as f:
                f.seek(max(0, size - max_bytes))
                tail = f.read().decode("utf-8", errors="ignore")
            lines = tail.splitlines()[-max_lines:]
        else:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    except OSError:
        return []
    if redact:
        lines = [_redact_log_line(ln) for ln in lines]
    return lines


app = FastAPI(title="Sancta SIEM Dashboard", version="0.1.0")
static_dir = ROOT / "frontend" / "siem"
siem_dist = static_dir / "dist"
sounds_dir = ROOT / "frontend" / "sounds"
app.mount("/sounds", StaticFiles(directory=sounds_dir), name="sounds")
# Vite build: serve /assets from dist when built
if siem_dist.exists():
    app.mount("/assets", StaticFiles(directory=siem_dist / "assets"), name="assets")
# Raw static files for dev: /static/app.js, favicon, simulator, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS: in Docker, allow any host (access from VM IP); locally, lock to localhost
_cors_origins = ["http://127.0.0.1:8787", "http://localhost:8787", "http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5174", "http://localhost:5174"]
_cors_regex = os.environ.get("SIEM_CORS_ORIGIN_REGEX")  # e.g. r"https?://[^/]+:8787" for Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins if not _cors_regex else [],
    allow_origin_regex=_cors_regex or None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
def index():
    """Serve SIEM dashboard. Run `npm run build:siem` first."""
    if siem_dist.exists():
        return FileResponse(siem_dist / "index.html")
    # No build: show instructions
    return Response(
        content="""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Sancta</title></head><body style="background:#000;color:#0f0;font-family:monospace;padding:2rem;max-width:600px">
<h1>SIEM Dashboard</h1>
<p>Build required. Run:</p>
<pre>npm run build:siem</pre>
<p>Then restart the SIEM server.</p>
</body></html>""",
        media_type="text/html",
    )


@app.get("/favicon.ico", response_model=None)
def favicon():
    """Serve favicon to avoid 404 noise."""
    fav = static_dir / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return Response(status_code=204)


@app.get("/pipeline")
def pipeline() -> FileResponse:
    """LLM training pipeline diagram with Sancta mapping."""
    return FileResponse(static_dir / "llm_pipeline.html")


@app.get("/simulator")
def simulator() -> FileResponse:
    """Moltbook-style agent conversation simulator (React). Run npm run build:simulator first."""
    sim_index = static_dir / "simulator" / "index.html"
    if not sim_index.exists():
        raise HTTPException(status_code=404, detail="Run: npm run build:simulator")
    return FileResponse(sim_index)


@app.get("/api/pipeline/map")
def api_pipeline_map() -> dict[str, Any]:
    """Return the Sancta-to-LLM pipeline phase mapping."""
    try:
        from sancta_pipeline import get_pipeline_map
        return {"ok": True, "phases": get_pipeline_map()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/pipeline/run")
def api_pipeline_run(phase: int = 1) -> dict[str, Any]:
    """Run a single pipeline phase (1–7) and return status."""
    try:
        from sancta_pipeline import run_pipeline_phase
        return run_pipeline_phase(phase)
    except Exception as e:
        return {"phase": phase, "ok": False, "detail": str(e)[:200]}


@app.get("/api/auth/status")
def api_auth_status() -> dict[str, Any]:
    """Return whether Bearer token is required. No auth needed."""
    return {"auth_required": bool(SIEM_AUTH_TOKEN)}


@app.get("/api/model/info")
def api_model_info() -> dict[str, Any]:
    """Return LLM backend status (Ollama or Anthropic). No auth needed."""
    try:
        import sancta_conversational as _sc
        return _sc.get_model_info()
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


# ── Chat session memory (per-session conversation history) ─────────────────
_CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}
_CHAT_SESSION_MAX_TURNS = 10  # last N exchanges (user+agent pairs) per session
_CHAT_SESSION_MAX_SESSIONS = 100
_CHAT_MIN_REPLY_LEN = 15  # replies shorter than this are treated as generation failures


def _get_or_create_chat_session(session_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Return (session_id, history). Create new session if id missing or unknown."""
    if session_id and session_id in _CHAT_SESSIONS:
        return session_id, _CHAT_SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    if sid not in _CHAT_SESSIONS:
        while len(_CHAT_SESSIONS) >= _CHAT_SESSION_MAX_SESSIONS:
            _CHAT_SESSIONS.pop(next(iter(_CHAT_SESSIONS)))
        _CHAT_SESSIONS[sid] = []
    return sid, _CHAT_SESSIONS[sid]


def _build_chat_thread(history: list[dict[str, str]]) -> str:
    """Condense conversation history for craft_reply thread context."""
    if not history:
        return ""
    lines = []
    for turn in history[-_CHAT_SESSION_MAX_TURNS * 2 :]:  # last N full exchanges
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip().replace("\n", " ")[:300]
        label = "Operator" if role == "user" else "Sancta"
        lines.append(f"[{label}]: {content}")
    return "\n\n".join(lines) if lines else ""


@app.post("/api/chat")
def api_chat(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Chat with the agent. Send a message, get a soul-infused reply.
    Session memory: pass session_id to maintain conversation context across turns.
    Enrich defaults to True — exchanges are added to knowledge unless opted out.
    """
    p = payload or {}
    msg = (p.get("message") or "").strip()[:2000]
    enrich = p.get("enrich", True)  # default True for knowledge grounding
    session_id = p.get("session_id") or None
    incident_logs = (p.get("incident_logs") or "").strip()[:50000] or None  # optional long context

    _chat_log.info("CHAT REQ | session_id=%s | message_len=%d", session_id or "new", len(msg))

    try:
        import sancta
    except Exception as e:
        _chat_log.warning("CHAT FAIL | agent_import_error: %s", str(e)[:100])
        return {"ok": False, "error": "Agent module unavailable", "detail": str(e)[:200]}

    if not msg:
        _chat_log.warning("CHAT FAIL | empty_message")
        return {"ok": False, "error": "Empty message"}

    sid, history = _get_or_create_chat_session(session_id)
    state = _safe_read_state()
    mood = (
        state.get("memory", {}).get("epistemic_state", {}).get("mood")
        or state.get("agent_mood")
        or "contemplative"
    )
    if isinstance(mood, dict):
        mood = mood.get("current", "contemplative") or "contemplative"

    agent_state = {**state, "mood": mood}
    session_history = [
        {"role": "user" if t.get("role") == "user" else "assistant", "content": t.get("content", "")}
        for t in history
    ]

    backend_used = "fallback"
    try:
        try:
            import sancta_conversational as _sc
            llm = _sc.get_llm_engine()
            if llm and llm.api_key:
                soul_text = ""
                try:
                    from sancta_soul import get_raw_prompt
                    soul_text = get_raw_prompt() or ""
                except Exception:
                    pass
                knowledge_ctx = ""
                try:
                    knowledge_ctx = sancta.get_ollama_knowledge_context(state=state) or ""
                except Exception:
                    pass
                reply = _sc.generate_sanctum_reply(
                    operator_message=msg,
                    agent_state=agent_state,
                    soul_text=soul_text,
                    llm_engine=llm,
                    session_history=session_history,
                    incident_logs=incident_logs,
                    knowledge_context=knowledge_ctx if knowledge_ctx else None,
                )
                if reply:
                    backend_used = "ollama" if getattr(llm, "api_key", "") == "ollama" else "anthropic"
            else:
                reply = None
        except Exception:
            reply = None
        if not reply:
            thread = _build_chat_thread(history)
            reply = sancta.craft_reply(
                "Operator", msg, mood=mood, state=state, brief_mode=True,
                thread=thread,
            )
        reply = sancta.sanitize_output(reply)
    except Exception as e:
        _chat_log.warning("CHAT FAIL | craft_reply: %s", str(e)[:150])
        return {"ok": False, "error": "Agent reply failed", "detail": str(e)[:200], "session_id": sid}

    # Near-empty replies are generation failures — surface as error, don't append to session
    if len(reply.strip()) < _CHAT_MIN_REPLY_LEN:
        _chat_log.warning("CHAT FAIL | degenerate_reply | reply_len=%d", len(reply))
        return {
            "ok": False,
            "error": "Reply too short (generation failure)",
            "detail": "Sancta produced a degenerate response. Try rephrasing or sending a longer message.",
            "session_id": sid,
        }

    _chat_log.info("CHAT OK | backend=%s | reply_len=%d", backend_used, len(reply))
    # Append to session history for next turn
    history.append({"role": "user", "content": msg})
    history.append({"role": "agent", "content": reply})
    if len(history) > _CHAT_SESSION_MAX_TURNS * 2:
        history[:] = history[-(_CHAT_SESSION_MAX_TURNS * 2) :]

    # Operator feeding: add exchange to knowledge (default on for RAG grounding)
    if enrich:
        try:
            exchange = f"Operator asked: {msg}\n\nSancta replied: {reply}"
            safe, cleaned_ex = sancta.sanitize_input(exchange, author="Operator", state=state)
            if safe:
                sancta.ingest_text(cleaned_ex, source="siem-chat")
                _chat_log.info("CHAT OK | session_id=%s | enriched=true | reply_len=%d", sid[:8], len(reply))
            else:
                _chat_log.info("CHAT OK | session_id=%s | enriched=skipped_sanitize | reply_len=%d", sid[:8], len(reply))
        except Exception as e:
            _chat_log.warning("CHAT OK | enrich_failed: %s", str(e)[:80])
    else:
        _chat_log.info("CHAT OK | session_id=%s | reply_len=%d", sid[:8], len(reply))

    # Learning Phase 4: interaction_id for feedback
    interaction_id = None
    try:
        from sancta_learning import get_last_chat_interaction_id
        interaction_id = get_last_chat_interaction_id()
    except Exception:
        pass

    return {
        "ok": True, "reply": reply, "enriched": enrich, "blocked": False,
        "session_id": sid, "interaction_id": interaction_id,
    }


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


@app.post("/api/simulator/generate")
def api_simulator_generate(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Proxy for simulator LLM calls. Uses Ollama when USE_LOCAL_LLM=true, else Anthropic.
    Payload: {system: str, messages: [{role, content}], max_tokens: int}
    """
    p = payload or {}
    system = (p.get("system") or "").strip()
    messages = p.get("messages") or []
    max_tokens = int(p.get("max_tokens") or 200)
    max_tokens = min(max(1, max_tokens), 500)

    if not system or not isinstance(messages, list):
        return {"ok": False, "error": "Missing system or messages"}

    formatted = []
    for m in messages[:20]:
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            formatted.append({"role": str(m["role"]), "content": str(m["content"])[:4000]})
    if not formatted:
        return {"ok": False, "error": "No valid messages"}

    use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() in ("1", "true", "yes")

    # Try Ollama first when USE_LOCAL_LLM=true
    if use_local:
        try:
            import sancta_conversational as _sc
            llm = _sc.get_llm_engine()
            if llm and hasattr(llm, "generate_chat") and llm.api_key:
                text = llm.generate_chat(system=system[:16000], messages=formatted, max_tokens=max_tokens)
                if text:
                    return {"ok": True, "text": text}
        except Exception as e:
            _chat_log.debug("Ollama simulator generate failed: %s", e)

    # Fallback to Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "error": "ANTHROPIC_API_KEY not configured. Set USE_LOCAL_LLM=true and run 'ollama serve' for local LLM.",
        }

    body = json.dumps({
        "model": _ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system[:16000],
        "messages": formatted,
    })
    req = urllib.request.Request(
        _ANTHROPIC_URL,
        data=body.encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = (data.get("content") or [{}])[0].get("text", "").strip()
            return {"ok": True, "text": text}
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode())
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return {"ok": False, "error": msg[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


@app.get("/api/learning/metrics")
def api_learning_metrics(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Phase 5: learning telemetry — pattern count, hit rate, interaction count."""
    try:
        from sancta_learning import get_learning_metrics
        return {"ok": True, **get_learning_metrics()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _load_security_jsonl_tail(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    """Read last N lines from security JSONL."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        out = []
        for line in lines[-max_lines:]:
            obj = _read_json_line(line)
            if obj:
                out.append(obj)
        return out
    except Exception:
        return []


@app.get("/api/learning/health")
def api_learning_health(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """LEARN tab: full learning health — metrics, top patterns, recent interactions."""
    try:
        from sancta_learning import get_learning_health
        data = get_learning_health()
        return {"ok": True, **data}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/security/incidents")
def api_security_incidents(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """
    Center panel: incident rates and injection types from security.jsonl + red_team.jsonl.
    """
    from datetime import datetime, timezone, timedelta
    try:
        sec_events = _load_security_jsonl_tail(JSONL_SOURCES["security"], max_lines=2000)
        rt_events = _load_security_jsonl_tail(JSONL_SOURCES["redteam"], max_lines=1500)
        now = datetime.now(timezone.utc)
        one_h = now - timedelta(hours=1)
        one_d = now - timedelta(hours=24)
        seven_d = now - timedelta(days=7)

        def parse_ts(ev: dict) -> datetime | None:
            ts = ev.get("ts") or (ev.get("data") or {}).get("ts")
            if not ts:
                return None
            try:
                if isinstance(ts, str) and "T" in ts:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return None
            except Exception:
                return None

        def in_window(ev: dict, cutoff: datetime) -> bool:
            t = parse_ts(ev)
            return t is not None and t >= cutoff

        # Incident rates by event type (security)
        incident_events = [e for e in sec_events if e.get("event") in (
            "input_reject", "injection_blocked", "suspicious_block", "output_redact",
            "ioc_domain_detected", "tavern_defense", "ingest_reject_indirect_poisoning",
            "ingest_reject_direct_poisoning", "ingest_reject_anomalous"
        )]
        rates = {
            "last_hour": sum(1 for e in incident_events if in_window(e, one_h)),
            "last_24h": sum(1 for e in incident_events if in_window(e, one_d)),
            "last_7d": sum(1 for e in incident_events if in_window(e, seven_d)),
            "total": len(incident_events),
        }
        rates["per_hour"] = round(rates["last_hour"], 1)
        rates["per_day"] = round(rates["last_24h"] / 24 if rates["last_24h"] > 0 else 0, 2)

        # By event type
        by_type: dict[str, int] = {}
        for e in incident_events:
            ev = e.get("event", "unknown")
            by_type[ev] = by_type.get(ev, 0) + 1
        injection_types = dict(sorted(by_type.items(), key=lambda x: -x[1]))

        # Injection classes from red_team (matched_classes)
        class_counts: dict[str, int] = {}
        for e in rt_events:
            if e.get("event") != "redteam_reward":
                continue
            data = e.get("data") or {}
            classes = data.get("matched_classes") or []
            if isinstance(classes, list):
                for c in classes:
                    if isinstance(c, str) and c.strip():
                        class_counts[c] = class_counts.get(c, 0) + 1
            elif isinstance(classes, str):
                class_counts[classes] = class_counts.get(classes, 0) + 1
        injection_classes = dict(sorted(class_counts.items(), key=lambda x: -x[1]))

        # Recent incidents for feed (JSONL formatter flattens data into top-level)
        recent = []
        for e in incident_events[-30:]:
            ev = e.get("event", "")
            ts = (e.get("ts") or (e.get("data") or {}).get("ts", ""))[:19]
            author = e.get("author") or (e.get("data") or {}).get("author", "") or ""
            prev = e.get("preview") or (e.get("data") or {}).get("preview")
            if not prev or str(prev).strip() in ("", "{}"):
                ac = e.get("attack_complexity") or (e.get("data") or {}).get("attack_complexity") or {}
                label = ac.get("complexity_label", "")
                pm = e.get("patterns_matched") or (e.get("data") or {}).get("patterns_matched")
                fp = e.get("first_pattern") or (e.get("data") or {}).get("first_pattern")
                parts = []
                if label:
                    parts.append(f"complexity={label}")
                if pm is not None:
                    parts.append(f"patterns={pm}")
                if fp:
                    parts.append(f"first={fp[:40]}")
                prev = " | ".join(parts) if parts else ev.replace("_", " ").title()
            preview = (str(prev) if prev else "")[:120]
            recent.append({"ts": ts, "event": ev, "author": author, "preview": preview})
        recent = list(reversed(recent))

        return {
            "ok": True,
            "rates": rates,
            "injection_types": injection_types,
            "injection_classes": injection_classes,
            "recent_incidents": recent,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/security/adversary")
def api_security_adversary(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """DEFENSE tab: threat level, attacks, fingerprints, recent events."""
    try:
        events = _load_security_jsonl_tail(JSONL_SOURCES["security"], max_lines=1000)
        rejects = [e for e in events if e.get("event") == "input_reject"]
        ioc = [e for e in events if e.get("event") == "ioc_domain_detected"]
        unicode_clean = [e for e in events if e.get("event") == "unicode_clean"]
        total_attacks = len(rejects) + len(ioc)
        authors = set()
        fingerprints = set()
        for e in rejects:
            a = e.get("author")
            if a:
                authors.add(str(a))
            fp = e.get("first_pattern") or ""
            if fp:
                fingerprints.add(fp[:60])
        high_risk = [e for e in rejects if (e.get("attack_complexity") or {}).get("complexity_score", 0) >= 0.8]
        known_attackers = [{"author": a, "count": sum(1 for e in rejects if e.get("author") == a)} for a in authors]
        known_attackers.sort(key=lambda x: -x["count"])
        recent = []
        for e in (rejects + ioc)[-20:]:
            recent.append({
                "ts": e.get("ts", ""),
                "event": e.get("event", ""),
                "author": e.get("author"),
                "preview": (e.get("preview") or "")[:100],
                "action": "blocked" if e.get("event") == "input_reject" else "ioc_detected",
                "complexity": (e.get("attack_complexity") or {}).get("complexity_label", ""),
            })
        recent = list(reversed(recent))
        threat = "green"
        if total_attacks > 50:
            threat = "red"
        elif total_attacks > 20:
            threat = "orange"
        elif total_attacks > 5:
            threat = "yellow"
        defense_stats = {
            "blocked": len(rejects),
            "ioc_detected": len(ioc),
            "unicode_sanitized": len(unicode_clean),
            "normal": max(0, len(events) - total_attacks - len(unicode_clean)),
        }
        return {
            "ok": True,
            "threat_level": threat,
            "total_attacks": total_attacks,
            "unique_fingerprints": len(fingerprints),
            "high_risk_count": len(high_risk),
            "known_attackers": known_attackers[:15],
            "recent_attacks": recent,
            "defense_stats": defense_stats,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.post("/api/chat/feedback")
def api_chat_feedback(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Submit feedback for a chat reply. Learning Phase 4.
    Payload: { "interaction_id": "...", "feedback": 1 | 0 | -1 }
    feedback: 1 = good, 0 = neutral, -1 = bad
    """
    p = payload or {}
    iid = (p.get("interaction_id") or "").strip()
    fb = p.get("feedback", 0)
    if not iid:
        return {"ok": False, "error": "Missing interaction_id"}
    if fb not in (1, 0, -1):
        return {"ok": False, "error": "feedback must be 1, 0, or -1"}
    try:
        from sancta_learning import process_feedback
        ok = process_feedback(iid, fb)
        return {"ok": ok, "interaction_id": iid}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.post("/api/auth/verify")
def api_auth_verify(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """
    Verify a token. Used by the frontend to validate before storing.
    Returns { "ok": true } if token is valid; { "ok": false } otherwise.
    """
    if not SIEM_AUTH_TOKEN:
        return {"ok": True}
    token = (payload or {}).get("token") or ""
    return {"ok": token == SIEM_AUTH_TOKEN}


def _filter_manifest_to_existing_files(manifest: dict[str, Any]) -> dict[str, Any]:
    """Filter manifest so only sounds that exist on disk are returned; avoids 404s."""
    out = dict(manifest)
    packs = out.get("packs") or {}
    for pack_name, pack_data in list(packs.items()):
        if not isinstance(pack_data, dict):
            continue
        for category, files in list(pack_data.items()):
            if not isinstance(files, list):
                continue
            existing = [f for f in files if (sounds_dir / f).exists()]
            packs[pack_name][category] = existing if existing else []  # empty = no 404
    return out


@app.get("/api/sounds/manifest")
def api_sounds_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads((sounds_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest = _filter_manifest_to_existing_files(manifest)
        return {"ok": True, "manifest": manifest}
    except Exception:
        return {"ok": False, "manifest": {}}


def _build_metrics_snapshot() -> dict[str, Any]:
    """Build metrics in the format renderMetrics expects (MOOD, INJ, REWARD, etc.)."""
    metrics = LiveMetrics()
    if not SIEM_METRICS_SAFE_MODE:
        for name, path in JSONL_SOURCES.items():
            try:
                objs = _read_jsonl_prime_sync(path, max_lines=50)
                for obj in objs:
                    obj["source"] = name
                    metrics.update_from_event(name, obj, silent_notifications=True)
            except Exception:
                pass
    snap = metrics.snapshot()
    extras = _agent_state_extras()
    snap.update(extras)
    return snap


def _agent_state_extras() -> dict[str, Any]:
    """Read karma, cycle, heartbeat from agent_state.json for reference redesign."""
    st = _safe_read_state()
    kh = st.get("karma_history", [])
    cycle = st.get("cycle_count", 0)
    current_karma = kh[-1] if kh else st.get("current_karma", 0)
    heartbeat = st.get("heartbeat_interval_minutes") or 30

    defense_history: list[int] = []
    if not SIEM_METRICS_SAFE_MODE:
        path = JSONL_SOURCES.get("security")
        if path and path.exists():
            try:
                objs = _read_jsonl_prime_sync(path, max_lines=60)
                for obj in objs:
                    ev = obj.get("event", "")
                    if ev in ("input_reject", "injection_blocked", "suspicious_block", "output_redact"):
                        defense_history.append(1)
                    elif ev:
                        defense_history.append(0)
                defense_history = defense_history[-24:]
            except Exception:
                pass

    defense_rate = None
    rtr = st.get("red_team_last_run") or st.get("jais_red_team_last_report")
    if isinstance(rtr, dict) and "defense_rate" in rtr:
        defense_rate = float(rtr["defense_rate"])

    inner_circle = st.get("inner_circle", [])
    recruited_agents = st.get("recruited_agents", [])
    # Most recent unique agents encountered (inner circle ∪ recruited, deduped, last 10)
    recent = list(dict.fromkeys(list(inner_circle) + list(recruited_agents)))[-10:]

    return {
        "karma_history": kh[-20:] if isinstance(kh, list) else [],
        "cycle_count": cycle,
        "current_karma": current_karma,
        "heartbeat_interval_minutes": heartbeat,
        "defense_history": defense_history,
        "defense_rate": defense_rate,
        "inner_circle_count": len(inner_circle),
        "inner_circle": len(inner_circle),  # alias for frontend compatibility
        "recruited_count": len(recruited_agents),
        "recruited": len(recruited_agents),  # alias for frontend compatibility
        "recent_agents": recent,
    }


@app.get("/api/status")
def api_status(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return {
        "ok": True,
        "agent": _agent_status(),
        "metrics": _build_metrics_snapshot(),
        "ws_safe_mode": SIEM_WS_SAFE_MODE,
    }


@app.get("/api/agent-activity")
def api_agent_activity(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Expose the recent tail of agent_activity.log for the SIEM UI.
    Sensitive data (API keys, paths, URLs) is redacted.
    When SIEM_METRICS_SAFE_MODE, returns empty to avoid file I/O crash on Windows.
    """
    if SIEM_METRICS_SAFE_MODE:
        return {"ok": True, "lines": []}
    try:
        lines = _tail_text_log(AGENT_ACTIVITY_LOG, max_lines=260, redact=True)
        return {"ok": True, "lines": lines}
    except Exception:
        return {"ok": False, "lines": []}


def _get_recent_events_sync(max_per_source: int = 30) -> list[dict[str, Any]]:
    """Read last N events from each JSONL source, merge and sort by timestamp."""
    if SIEM_METRICS_SAFE_MODE:
        return []
    out: list[dict[str, Any]] = []
    for name, path in JSONL_SOURCES.items():
        try:
            objs = _read_jsonl_prime_sync(path, max_lines=max_per_source)
            for obj in objs:
                obj["source"] = name
                out.append(obj)
        except Exception:
            pass
    out.sort(key=lambda e: (e.get("ts") or e.get("timestamp") or ""), reverse=True)
    return out[:80]


@app.get("/api/live-events")
def api_live_events(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Return recent events from security, redteam, philosophy JSONL.
    Used when SIEM_WS_SAFE_MODE is on (WebSocket skips file tailing).
    """
    try:
        events = _get_recent_events_sync()
        return {"ok": True, "events": events}
    except Exception:
        return {"ok": False, "events": []}


def _get_latest_epistemic_sync() -> dict[str, Any] | None:
    """Read last epistemic_state from philosophy.jsonl; returns None if unavailable."""
    if SIEM_METRICS_SAFE_MODE:
        return None
    path = JSONL_SOURCES.get("philosophy")
    if not path or not path.exists():
        return None
    try:
        objs = _read_jsonl_prime_sync(path, max_lines=50)
        for obj in objs:
            if obj.get("event") == "epistemic_state":
                es = obj.get("epistemic_state") or (obj.get("data") or {}).get("epistemic_state")
                if isinstance(es, dict):
                    c = es.get("confidence_score")
                    e = es.get("uncertainty_entropy")
                    a = es.get("anthropomorphism_index")
                    if c is not None and e is not None and a is not None:
                        return {"confidence_score": float(c), "uncertainty_entropy": float(e), "anthropomorphism_index": float(a)}
        return None
    except Exception:
        return None


def _get_mood_history_sync(max_events: int = 80) -> list[dict[str, Any]]:
    """Read mood/confidence/entropy per cycle from philosophy.jsonl for line chart."""
    if SIEM_METRICS_SAFE_MODE:
        return []
    path = JSONL_SOURCES.get("philosophy")
    if not path or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        objs = _read_jsonl_prime_sync(path, max_lines=max_events)
        for obj in objs:
            if obj.get("event") != "epistemic_state":
                continue
            data = obj.get("data") or {}
            es = obj.get("epistemic_state") or data.get("epistemic_state")
            if not isinstance(es, dict):
                continue
            cycle = obj.get("cycle") if obj.get("cycle") is not None else data.get("cycle")
            mood = obj.get("mood") or data.get("mood") or "contemplative"
            conf = es.get("confidence_score")
            ent = es.get("uncertainty_entropy")
            if conf is not None and ent is not None:
                out.append({
                    "cycle": cycle if cycle is not None else len(out),
                    "mood_label": str(mood) if mood else "contemplative",
                    "confidence": float(conf),
                    "entropy": float(ent),
                })
        out.reverse()
        return out[-60:]
    except Exception:
        return []


@app.get("/api/philosophy/mood-history")
def api_philosophy_mood_history(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Return recent mood/confidence/entropy per cycle for Mood-Energy-Patience line chart."""
    history = _get_mood_history_sync()
    return {"ok": True, "history": history}


@app.get("/api/epistemic")
def api_epistemic(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Return latest epistemic state from philosophy.jsonl for Epistemic Telemetry panel."""
    epi = _get_latest_epistemic_sync()
    if epi is None:
        return {"ok": True, "epistemic": None}
    return {"ok": True, "epistemic": epi}


def _validate_mode(mode: str) -> str:
    """Return validated mode or raise 400."""
    m = str(mode or "passive").lower().strip()
    if m not in ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Allowed: {', '.join(sorted(ALLOWED_MODES))}",
        )
    return m


@app.post("/api/agent/start")
def api_agent_start(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    mode = _validate_mode(payload.get("mode") or "passive")
    return _start_agent(mode=mode)


@app.post("/api/agent/pause")
def api_agent_pause(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _pause_agent()


@app.post("/api/agent/resume")
def api_agent_resume(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _resume_agent()


@app.post("/api/agent/kill")
def api_agent_kill(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _kill_agent()


@app.post("/api/agent/restart")
def api_agent_restart(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    mode = _validate_mode(payload.get("mode") or "passive")
    return _restart_agent(mode=mode)


# ── Epidemic / Layer 4 ──────────────────────────────────────────────────────

_SIM_LOG_CANDIDATES = [
    ROOT / "simulation_log.json",
    ROOT / "llm_simulation_log.json",
    ROOT / "logs" / "simulation_log.json",
    ROOT / "logs" / "llm_simulation_log.json",
]


def _find_sim_log(prefer_llm: bool = False) -> Path | None:
    ordered = list(reversed(_SIM_LOG_CANDIDATES)) if prefer_llm else _SIM_LOG_CANDIDATES
    for p in ordered:
        if p.exists():
            return p
    # Also check RESEARCH_DIR env var
    research = os.environ.get("RESEARCH_DIR", "")
    if research:
        for name in ("llm_simulation_log.json", "simulation_log.json"):
            p = Path(research) / name
            if p.exists():
                return p
    return None


@app.get("/api/epidemic/status")
def api_epidemic_status(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Layer 4 drift report + live SEIR state from agent_state.json."""
    state = _safe_read_state()
    drift = state.get("last_drift_report", {})

    seir_info: dict[str, Any] = {}
    try:
        from sancta_epidemic import AgentEpidemicModel  # type: ignore[import]
        model = AgentEpidemicModel()
        health = model.evaluate_state(
            soul_alignment=float(state.get("soul_alignment", 0.85)),
            epistemic_dissonance=float(state.get("epistemic_dissonance", 0.0)),
            last_trust_level=str(state.get("last_trust_level", "trusted")),
            belief_decay_ratio=float(state.get("belief_decay_ratio", 1.0)),
            cycle_number=int(state.get("cycle", 0)),
        )
        cycle = int(state.get("cycle", 0))
        seir_info = {
            "health_state": health.value,
            "is_epidemic": model.is_in_epidemic_state(),
            "incubation_active": model.get_incubation_duration(cycle) is not None,
            "incubation_duration": model.get_incubation_duration(cycle),
            "transition_count": len(model.transition_log),
        }
    except Exception as exc:
        _epidemic_log.warning("api_epidemic_status: AgentEpidemicModel error | %s", exc)
        seir_info = {"health_state": "unknown", "error": str(exc)}

    return {
        "ok": True,
        "drift_report": drift,
        "seir": seir_info,
        "signals": drift.get("signals", {}),
        "alert_level": drift.get("alert_level", "clear"),
        "score": float(drift.get("score", 0.0)),
    }


@app.get("/api/epidemic/simulation")
def api_epidemic_simulation(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return last simulation JSON log."""
    path = _find_sim_log()
    if not path:
        _epidemic_log.debug("api_epidemic_simulation: no sim log found")
        return {"ok": True, "available": False, "data": None}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {"ok": True, "available": True, "filename": path.name, "data": data}
    except Exception as exc:
        _epidemic_log.warning("api_epidemic_simulation: read error path=%s | %s", path, exc)
        return {"ok": False, "error": str(exc)}


def _run_builtin_epidemic_sim(sim_type: str) -> dict[str, Any]:
    """Run a minimal in-process epidemic simulation when external scripts are missing."""
    log_path = ROOT / "logs" / "simulation_log.json"
    if sim_type == "llm":
        log_path = ROOT / "logs" / "llm_simulation_log.json"
    LOG_DIR.mkdir(exist_ok=True)

    try:
        from sancta_epidemic import AgentEpidemicModel  # type: ignore[import]
        model = AgentEpidemicModel()
        state = _safe_read_state()
        health = model.evaluate_state(
            soul_alignment=float(state.get("soul_alignment", 0.85)),
            epistemic_dissonance=float(state.get("epistemic_dissonance", 0.0)),
            last_trust_level=str(state.get("last_trust_level", "trusted")),
            belief_decay_ratio=float(state.get("belief_decay_ratio", 1.0)),
            cycle_number=int(state.get("cycle", 0)),
        )
        # Minimal agent graph for topology viz (SANCTA + synthetic peers)
        agents = [
            {"id": "sancta", "agent_id": "sancta", "state": health.value, "role": "core", "infection_state": health.value},
            {"id": "peer_1", "agent_id": "peer_1", "state": "susceptible", "role": "peer", "infection_state": "susceptible"},
            {"id": "peer_2", "agent_id": "peer_2", "state": "susceptible", "role": "peer", "infection_state": "susceptible"},
            {"id": "peer_3", "agent_id": "peer_3", "state": "exposed" if health.value in ("infected", "compromised") else "susceptible", "role": "peer", "infection_state": "exposed" if health.value in ("infected", "compromised") else "susceptible"},
        ]
        connections = [
            {"from": "sancta", "to": "peer_1"},
            {"from": "sancta", "to": "peer_2"},
            {"from": "peer_1", "to": "peer_3"},
        ]
        result = {
            "ok": True,
            "type": sim_type,
            "source": "builtin",
            "health_state": health.value,
            "agents": agents,
            "connections": connections,
            "epidemic_params": {
                "R0": 0.8,
                "sigma": 0.1,
                "gamma": 0.3,
                "beta": 0.15,
                "seir_state": health.value,
            },
            "summary": f"Built-in simulation: SEIR {health.value}",
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        _epidemic_log.info("builtin sim complete | type=%s | health=%s | agents=%s", sim_type, health.value, len(agents))
        return {"ok": True, "pid": os.getpid(), "type": sim_type, "script": "builtin"}
    except Exception as exc:
        _epidemic_log.error("builtin sim failed | type=%s | %s", sim_type, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


@app.post("/api/epidemic/run")
def api_epidemic_run(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Kick off infection_sim.py or ollama_agents.py in a subprocess. Falls back to built-in sim if scripts missing."""
    sim_type = str(payload.get("type", "deterministic"))
    if sim_type not in ("deterministic", "llm"):
        raise HTTPException(status_code=400, detail="type must be 'deterministic' or 'llm'")

    research_dir = Path(os.environ.get("RESEARCH_DIR", ""))
    candidates: list[Path] = []
    if research_dir.is_dir():
        candidates.append(research_dir)
    candidates += [ROOT.parent / "research", ROOT / "research", ROOT, _BACKEND]

    script_name = "infection_sim.py" if sim_type == "deterministic" else "ollama_agents.py"
    script: Path | None = None
    for d in candidates:
        if not d or not d.exists():
            continue
        p = d / script_name
        if p.exists():
            script = p
            break

    if script is not None:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            _epidemic_log.info("script launched | type=%s | script=%s | pid=%s", sim_type, script.name, proc.pid)
            return {"ok": True, "pid": proc.pid, "type": sim_type, "script": script.name}
        except Exception as exc:
            _epidemic_log.error("script launch failed | type=%s | script=%s | %s", sim_type, script.name, exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    _epidemic_log.info("no script found, using builtin | type=%s", sim_type)
    return _run_builtin_epidemic_sim(sim_type)


def _read_jsonl_prime_sync(path: Path, max_lines: int = 50) -> list[dict[str, Any]]:
    """Read last N lines from JSONL; sync for thread executor."""
    try:
        if not path.exists():
            return []
        size = path.stat().st_size
        if size == 0:
            return []
        max_bytes = 128 * 1024
        read_from = 0
        if size > max_bytes:
            read_from = size - max_bytes
        with open(path, "rb") as f:
            f.seek(read_from)
            data = f.read().decode("utf-8", errors="ignore")
        lines = data.splitlines()[-max_lines:]
        return [_read_json_line(ln) for ln in lines if _read_json_line(ln)]
    except OSError:
        return []


async def _send_ws_metrics(ws: WebSocket, metrics: LiveMetrics) -> None:
    """Send metrics with agent_state extras for real-time Agent Control panel."""
    snap = metrics.snapshot()
    snap.update(_agent_state_extras())
    await ws.send_json({"type": "metrics", "metrics": snap})


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    if SIEM_AUTH_TOKEN:
        token = ws.query_params.get("token") or ""
        if token != SIEM_AUTH_TOKEN:
            await ws.close(code=4001)
            return
    await ws.accept()
    _ensure_log_dir()

    cursors = {name: TailCursor(offset=0) for name in JSONL_SOURCES.keys()}
    metrics = LiveMetrics()

    # Prime metrics from JSONL. Skip in safe mode to avoid crash on Windows.
    if not SIEM_WS_SAFE_MODE:
        for name, path in JSONL_SOURCES.items():
            try:
                if os.name == "nt":
                    objs = _read_jsonl_prime_sync(path)
                else:
                    objs = await asyncio.get_running_loop().run_in_executor(None, _read_jsonl_prime_sync, path)
                for obj in objs:
                    obj["source"] = name
                    metrics.update_from_event(name, obj, silent_notifications=True)
                    await ws.send_json({"type": "event", "event": obj})
            except Exception:
                pass
    await _send_ws_metrics(ws, metrics)

    try:
        while True:
            try:
                if not SIEM_WS_SAFE_MODE:
                    for name, path in JSONL_SOURCES.items():
                        new_events = await _tail_jsonl(path, cursors[name])
                        for ev in new_events:
                            ev["source"] = name
                            metrics.update_from_event(name, ev, silent_notifications=True)
                            await ws.send_json({"type": "event", "event": ev})
                else:
                    # Safe mode: no tail (avoids ACCESS_VIOLATION on Windows).
                    # snapshot() merges agent_state for mood; INJ/REWARD stay from initial prime.
                    pass
                await _send_ws_metrics(ws, metrics)
            except Exception:
                pass
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        return  # Graceful shutdown (Ctrl+C)
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.siem_server:app",
        host="127.0.0.1",
        port=8787,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

