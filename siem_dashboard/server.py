from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import psutil
from fastapi import Body, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable trained transformer to avoid PyTorch ACCESS_VIOLATION crash on some Windows setups.
# Set SANCTA_USE_TRAINED_TRANSFORMER=true in .env to re-enable (if stable on your system).
if "SANCTA_USE_TRAINED_TRANSFORMER" not in os.environ:
    os.environ["SANCTA_USE_TRAINED_TRANSFORMER"] = "false"

import logging

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
STATE_PATH = ROOT / "agent_state.json"
SANCTA_PATH = ROOT / "sancta.py"
PID_PATH = ROOT / ".agent.pid"
AGENT_ACTIVITY_LOG = LOG_DIR / "agent_activity.log"

# Security hardening
SIEM_AUTH_TOKEN: str | None = os.environ.get("SIEM_AUTH_TOKEN") or None
# Default True to avoid ACCESS_VIOLATION crash on some Windows setups (WebSocket file I/O)
SIEM_WS_SAFE_MODE: bool = os.environ.get("SIEM_WS_SAFE_MODE", "true").lower() in ("1", "true", "yes")
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


def _proc_from_pid(pid: int | None) -> psutil.Process | None:
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
    except psutil.Error:
        return None
    if not p.is_running():
        return None
    return p


def _agent_status() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"running": False, "pid": None, "suspended": False}
    try:
        status = proc.status()
        suspended = status == psutil.STATUS_STOPPED
    except psutil.Error:
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
    try:
        proc.suspend()
    except psutil.Error as e:
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
    try:
        proc.resume()
    except psutil.Error as e:
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

    def update_from_event(self, source: str, ev: dict[str, Any]) -> None:
        event = ev.get("event")
        if source == "security":
            if event in ("input_reject", "injection_blocked", "suspicious_block"):
                self.injection_attempts += 1
            if event == "output_redact":
                self.output_redactions += 1

            if event in ("input_reject", "injection_blocked", "suspicious_block"):
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
            if r >= 0.5:
                notify(
                    EventCategory.REDTEAM_ALERT,
                    summary=f"Red-team reward={r:.2f}",
                    details={"reward": r},
                )

        if source == "philosophy" and event == "epistemic_state":
            self.mood = (ev.get("mood") or ev.get("data", {}).get("mood"))

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
        return {
            "injection_attempts_detected": self.injection_attempts,
            "sanitized_payload_count": self.output_redactions,
            "reward_score_rolling_sum": round(float(rolling_reward), 4),
            "false_positive_rate": self.fp_rate,
            "belief_confidence": self.belief_confidence,
            "agent_mood": self.mood,
            **_agent_status(),
        }


def _tail_jsonl_sync(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Sync tail of JSONL; used from thread to avoid blocking event loop."""
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

    out: list[dict[str, Any]] = []
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        obj = _read_json_line(line)
        if obj:
            out.append(obj)
    return out


async def _tail_jsonl(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Tail JSONL in thread to avoid blocking event loop."""
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
static_dir = Path(__file__).resolve().parent / "static"
sounds_dir = ROOT / "sounds"
app.mount("/sounds", StaticFiles(directory=sounds_dir), name="sounds")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS lockdown: allow only localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8787", "http://localhost:8787", "http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


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


@app.post("/api/chat")
def api_chat(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Chat with the agent. Send a message, get a soul-infused reply.
    Optional: enrich knowledge base with the exchange (user + agent).
    """
    _chat_log.info("CHAT REQ | message_len=%d", len((payload or {}).get("message") or ""))

    try:
        import sancta
    except Exception as e:
        _chat_log.warning("CHAT FAIL | agent_import_error: %s", str(e)[:100])
        return {"ok": False, "error": "Agent module unavailable", "detail": str(e)[:200]}

    msg = (payload or {}).get("message") or ""
    enrich = bool((payload or {}).get("enrich", False))
    msg = msg.strip()[:2000]
    if not msg:
        _chat_log.warning("CHAT FAIL | empty_message")
        return {"ok": False, "error": "Empty message"}

    state = _safe_read_state()
    mood = (
        state.get("memory", {}).get("epistemic_state", {}).get("mood")
        or state.get("agent_mood")
        or "contemplative"
    )
    if isinstance(mood, dict):
        mood = mood.get("current", "contemplative") or "contemplative"

    try:
        reply = sancta.craft_reply("Operator", msg, mood=mood, state=state, brief_mode=True)
        reply = sancta.sanitize_output(reply)
    except Exception as e:
        _chat_log.warning("CHAT FAIL | craft_reply: %s", str(e)[:150])
        return {"ok": False, "error": "Agent reply failed", "detail": str(e)[:200]}

    # Operator feeding: adds exchange to knowledge (brain input).
    if enrich:
        try:
            exchange = f"Operator asked: {msg}\n\nSancta replied: {reply}"
            safe, cleaned_ex = sancta.sanitize_input(exchange, author="Operator", state=state)
            if safe:
                sancta.ingest_text(cleaned_ex, source="siem-chat")
                _chat_log.info("CHAT OK | enriched=true | reply_len=%d", len(reply))
            else:
                _chat_log.info("CHAT OK | enriched=skipped_sanitize | reply_len=%d", len(reply))
        except Exception as e:
            _chat_log.warning("CHAT OK | enrich_failed: %s", str(e)[:80])
    else:
        _chat_log.info("CHAT OK | reply_len=%d", len(reply))

    return {"ok": True, "reply": reply, "enriched": enrich, "blocked": False}


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


@app.get("/api/status")
def api_status(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return {"ok": True, "agent": _agent_status(), "metrics": _safe_read_state().get("memory", {})}


@app.get("/api/agent-activity")
def api_agent_activity(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Expose the recent tail of agent_activity.log for the SIEM UI.
    Sensitive data (API keys, paths, URLs) is redacted.
    """
    try:
        lines = _tail_text_log(AGENT_ACTIVITY_LOG, max_lines=260, redact=True)
        return {"ok": True, "lines": lines}
    except Exception:
        return {"ok": False, "lines": []}


def _get_recent_events_sync(max_per_source: int = 30) -> list[dict[str, Any]]:
    """Read last N events from each JSONL source, merge and sort by timestamp."""
    out: list[dict[str, Any]] = []
    for name, path in JSONL_SOURCES.items():
        objs = _read_jsonl_prime_sync(path, max_lines=max_per_source)
        for obj in objs:
            obj["source"] = name
            out.append(obj)
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

    if not SIEM_WS_SAFE_MODE:
        for name, path in JSONL_SOURCES.items():
            try:
                objs = await asyncio.get_running_loop().run_in_executor(None, _read_jsonl_prime_sync, path)
                for obj in objs:
                    obj["source"] = name
                    metrics.update_from_event(name, obj)
                    await ws.send_json({"type": "event", "event": obj})
            except Exception:
                pass
    await ws.send_json({"type": "metrics", "metrics": metrics.snapshot()})

    try:
        while True:
            try:
                if not SIEM_WS_SAFE_MODE:
                    for name, path in JSONL_SOURCES.items():
                        new_events = await _tail_jsonl(path, cursors[name])
                        for ev in new_events:
                            ev["source"] = name
                            metrics.update_from_event(name, ev)
                            await ws.send_json({"type": "event", "event": ev})
                await ws.send_json({"type": "metrics", "metrics": metrics.snapshot()})
            except Exception:
                pass
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run(
        "siem_dashboard.server:app",
        host="127.0.0.1",
        port=8787,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

