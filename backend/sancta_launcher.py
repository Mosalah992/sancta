"""
sancta_launcher.py — Single-entry launcher for the Sancta agent stack.

Manages:
  - Ollama (connect to existing or start if needed)
  - SIEM dashboard server (backend.siem_server via uvicorn)
  - Sancta main agent loop (sancta.py)
  - Curiosity run (on-demand)

Build to exe:
  pip install pyinstaller
  pyinstaller sancta_launcher.spec
"""

import tkinter as tk
from tkinter import font as tkfont
import shutil
import subprocess
import threading
import time
import sys
import os
import webbrowser
import requests
import signal
import queue
from datetime import datetime
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_URL      = "http://127.0.0.1:11434"
SIEM_URL        = "http://127.0.0.1:8787"
OLLAMA_MODEL    = "llama3.2"

# Paths: support both source (backend/) and frozen exe (backend/dist/sancta_launcher.exe)
def _get_paths() -> tuple[Path, Path]:
    """Return (ROOT, BACKEND_DIR). ROOT = project root, BACKEND_DIR = backend/."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        # exe in backend/dist/ -> backend = exe_dir.parent
        # exe in project_root/dist/ -> backend = exe_dir.parent / "backend"
        if (exe_dir.parent / "siem_server.py").exists():
            backend = exe_dir.parent
        elif (exe_dir.parent / "backend" / "siem_server.py").exists():
            backend = exe_dir.parent / "backend"
        else:
            backend = exe_dir.parent
        root = backend.parent if backend.name == "backend" else exe_dir.parent
        return root, backend
    _backend = Path(__file__).resolve().parent
    return _backend.parent, _backend

ROOT, BACKEND_DIR = _get_paths()
SANCTA_SCRIPT    = BACKEND_DIR / "sancta.py"
CURIOSITY_FLAG   = "--curiosity-run"
PHENOMENOLOGY_FLAG = "--phenomenology-battery"

# Python for subprocesses: when frozen, sys.executable is the exe — use system Python
def _python_exe() -> str:
    if getattr(sys, "frozen", False):
        for cand in ["python", "python3", "py"]:
            exe = shutil.which(cand)
            if exe:
                return exe
        return "python"
    return sys.executable

# SIEM runs via uvicorn from project root (backend.siem_server:app)
SIEM_CMD = [_python_exe(), "-m", "uvicorn", "backend.siem_server:app",
            "--host", "127.0.0.1", "--port", "8787"]

# Ollama: try common install paths, or use env
def _find_ollama_exe() -> str:
    cand = os.environ.get("OLLAMA_EXE")
    if cand and Path(cand).exists():
        return cand
    for p in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(r"C:\Program Files\Ollama\ollama.exe"),
    ]:
        if p.exists():
            return str(p)
    return r"C:\Users\bluem\AppData\Local\Programs\Ollama\ollama.exe"  # fallback

OLLAMA_EXE = _find_ollama_exe()

SANCTA_RESTART_DELAY = 10   # seconds before auto-restart
MAX_LOG_LINES        = 300
POLL_INTERVAL        = 500   # ms — UI redraw only, no network here
NET_CHECK_INTERVAL   = 3     # seconds — background thread cadence
REQUEST_TIMEOUT      = 1     # never block UI thread

# ─── Colors ──────────────────────────────────────────────────────────────────

C = {
    "bg":        "#030712",
    "bg2":       "#0a0f1e",
    "bg3":       "#111827",
    "border":    "#1e293b",
    "purple":    "#818cf8",
    "purple_dk": "#4338ca",
    "green":     "#10b981",
    "red":       "#ef4444",
    "amber":     "#f59e0b",
    "teal":      "#14b8a6",
    "text":      "#e5e7eb",
    "muted":     "#6b7280",
    "dim":       "#374151",
}

# ─── State ───────────────────────────────────────────────────────────────────

processes   = {}          # name -> subprocess.Popen
log_queue   = queue.Queue()
restart_counts = {}       # name -> int
curiosity_running = False

# Background-populated status cache — UI reads this, never makes network calls
_net_status = {
    "ollama":      False,
    "siem":        False,
    "ollama_model": None,
}
_net_status_lock = threading.Lock()


def _background_net_checker():
    """Runs in a daemon thread. Updates _net_status every NET_CHECK_INTERVAL s."""
    while True:
        try:
            ollama_ok   = is_ollama_running()
            siem_ok     = is_siem_running()
            model       = get_ollama_model() if ollama_ok else None
            with _net_status_lock:
                _net_status["ollama"]       = ollama_ok
                _net_status["siem"]         = siem_ok
                _net_status["ollama_model"] = model
        except Exception:
            pass
        time.sleep(NET_CHECK_INTERVAL)

# ─── Process management ──────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except:
        return False

def is_siem_running() -> bool:
    try:
        r = requests.get(SIEM_URL, timeout=REQUEST_TIMEOUT)
        return r.status_code < 500
    except:
        return False

def get_ollama_model() -> str | None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            for m in models:
                if OLLAMA_MODEL in m:
                    return m
    except:
        pass
    return None

def start_ollama():
    if is_ollama_running():
        log_queue.put(("ollama", "INFO", "Already running — connecting"))
        return True
    log_queue.put(("ollama", "INFO", f"Starting Ollama..."))
    try:
        proc = subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        processes["ollama"] = proc
        # Wait up to 15s for it to come up
        for _ in range(15):
            time.sleep(1)
            if is_ollama_running():
                log_queue.put(("ollama", "OK", "Ollama ready on :11434"))
                return True
        log_queue.put(("ollama", "ERROR", "Timeout waiting for Ollama"))
        return False
    except FileNotFoundError:
        log_queue.put(("ollama", "ERROR", f"Not found: {OLLAMA_EXE}"))
        return False

def start_process(name: str, script: Path, extra_args: list = None,
                  env_extra: dict = None, restart: bool = False,
                  args_override: list = None, cwd_override: Path = None):
    """Start a Python process. Use args_override+cwd_override for SIEM (uvicorn)."""
    py = _python_exe()
    args = (args_override or [py, str(script)] + (extra_args or []))
    cwd = str(cwd_override or BACKEND_DIR)
    env = os.environ.copy()
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        processes[name] = proc
        log_queue.put((name, "OK", f"Started (PID {proc.pid})"))

        # Stream logs
        def _stream():
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    level = "ERROR" if "ERROR" in line or "Traceback" in line \
                            else "WARN" if "WARNING" in line or "WARN" in line \
                            else "INFO"
                    log_queue.put((name, level, line[:200]))
            # Process exited
            code = proc.wait()
            log_queue.put((name, "WARN" if code != 0 else "INFO",
                           f"Exited (code {code})"))
            if restart and name in processes and not curiosity_running:
                log_queue.put((name, "WARN",
                               f"Restarting in {SANCTA_RESTART_DELAY}s..."))
                restart_counts[name] = restart_counts.get(name, 0) + 1
                time.sleep(SANCTA_RESTART_DELAY)
                if name in processes:  # wasn't manually stopped
                    start_process(name, script, extra_args, env_extra, restart)

        threading.Thread(target=_stream, daemon=True).start()
        return True
    except Exception as e:
        log_queue.put((name, "ERROR", str(e)))
        return False

def stop_process(name: str):
    proc = processes.pop(name, None)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
        log_queue.put((name, "INFO", "Stopped"))

def stop_all():
    for name in list(processes.keys()):
        stop_process(name)

# ─── UI ──────────────────────────────────────────────────────────────────────

class SanctaLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sancta Control Center")
        self.configure(bg=C["bg"])
        self.geometry("920x680")
        self.minsize(800, 580)
        self.resizable(True, True)

        # Fonts
        try:
            self.fn_head  = tkfont.Font(family="Consolas", size=11, weight="bold")
            self.fn_mono  = tkfont.Font(family="Consolas", size=9)
            self.fn_label = tkfont.Font(family="Consolas", size=9)
            self.fn_title = tkfont.Font(family="Consolas", size=13, weight="bold")
            self.fn_btn   = tkfont.Font(family="Consolas", size=9, weight="bold")
        except:
            self.fn_head  = tkfont.Font(size=11, weight="bold")
            self.fn_mono  = tkfont.Font(size=9)
            self.fn_label = tkfont.Font(size=9)
            self.fn_title = tkfont.Font(size=13, weight="bold")
            self.fn_btn   = tkfont.Font(size=9, weight="bold")

        # Status vars
        self.status_vars = {
            "ollama":    tk.StringVar(value="●"),
            "siem":      tk.StringVar(value="●"),
            "sancta":    tk.StringVar(value="●"),
            "curiosity": tk.StringVar(value="●"),
            "phenomenology": tk.StringVar(value="●"),
        }
        self.status_labels = {}

        self._build_ui()
        self._bind_close()

        # Start polling
        self.after(500, self._poll_status)
        self.after(100, self._drain_logs)

    # ─── UI Build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["bg2"], height=52)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡ SANCTA", font=self.fn_title,
                 fg=C["purple"], bg=C["bg2"]).pack(side="left", padx=20, pady=14)
        tk.Label(hdr, text="CONTROL CENTER", font=self.fn_label,
                 fg=C["muted"], bg=C["bg2"]).pack(side="left", pady=14)

        # Version badge
        tk.Label(hdr, text="v2.0", font=self.fn_label,
                 fg=C["purple_dk"], bg=C["bg2"],
                 relief="flat", padx=8, pady=2).pack(side="right", padx=20)

        # Divider
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Main area: left panel + log
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # ── Left control panel ────────────────────────────────────────────
        left = tk.Frame(main, bg=C["bg2"], width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Frame(main, bg=C["border"], width=1).pack(side="left", fill="y")

        # Services section
        self._section(left, "SERVICES")

        services = [
            ("ollama",  "Ollama",        "llama3.2 · :11434"),
            ("siem",    "SIEM Dashboard","FastAPI · :8787"),
            ("sancta",  "Sancta Agent",  "Main loop"),
        ]
        for key, name, sub in services:
            self._service_row(left, key, name, sub)

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=16, pady=8)

        # Actions section
        self._section(left, "ACTIONS")

        self.btn_start = self._btn(left, "▶  START ALL",
                                   C["green"], self._start_all)
        self.btn_stop  = self._btn(left, "■  STOP ALL",
                                   C["red"], self._stop_all)
        self._btn(left, "⊞  OPEN DASHBOARD",
                  C["purple"], lambda: webbrowser.open(SIEM_URL))

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=16, pady=8)

        # Curiosity run
        self._section(left, "CURIOSITY RUN")

        curiosity_frame = tk.Frame(left, bg=C["bg2"])
        curiosity_frame.pack(fill="x", padx=16, pady=2)

        status_row = tk.Frame(curiosity_frame, bg=C["bg2"])
        status_row.pack(fill="x")

        self.curiosity_status = tk.Label(
            status_row, textvariable=self.status_vars["curiosity"],
            font=self.fn_head, fg=C["muted"], bg=C["bg2"]
        )
        self.curiosity_status.pack(side="left")
        tk.Label(status_row, text=" Curiosity Run",
                 font=self.fn_label, fg=C["text"], bg=C["bg2"]).pack(side="left")

        self.btn_curiosity = self._btn(
            left, "◈  RUN CURIOSITY",
            C["amber"], self._toggle_curiosity
        )

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=16, pady=8)

        # Phenomenology research
        self._section(left, "PHENOMENOLOGY RESEARCH")

        phen_frame = tk.Frame(left, bg=C["bg2"])
        phen_frame.pack(fill="x", padx=16, pady=2)

        phen_status_row = tk.Frame(phen_frame, bg=C["bg2"])
        phen_status_row.pack(fill="x")

        self.phenomenology_status = tk.Label(
            phen_status_row, textvariable=self.status_vars["phenomenology"],
            font=self.fn_head, fg=C["muted"], bg=C["bg2"]
        )
        self.phenomenology_status.pack(side="left")
        tk.Label(phen_status_row, text=" Attack battery (11 vectors)",
                 font=self.fn_label, fg=C["text"], bg=C["bg2"]).pack(side="left")

        self.btn_phenomenology = self._btn(
            left, "◇  RUN PHENOMENOLOGY BATTERY",
            C["teal"], self._toggle_phenomenology
        )

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=16, pady=8)

        # Stats
        self._section(left, "SESSION")
        self.stats_frame = tk.Frame(left, bg=C["bg2"])
        self.stats_frame.pack(fill="x", padx=16)

        self.stat_vars = {
            "uptime":    tk.StringVar(value="00:00:00"),
            "restarts":  tk.StringVar(value="0"),
            "ollama_m":  tk.StringVar(value="—"),
        }
        stats = [
            ("Uptime",     self.stat_vars["uptime"]),
            ("Restarts",   self.stat_vars["restarts"]),
            ("Model",      self.stat_vars["ollama_m"]),
        ]
        for label, var in stats:
            row = tk.Frame(self.stats_frame, bg=C["bg2"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", font=self.fn_label,
                     fg=C["muted"], bg=C["bg2"], width=10,
                     anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=self.fn_label,
                     fg=C["text"], bg=C["bg2"]).pack(side="left")

        # ── Right: log panel ─────────────────────────────────────────────
        right = tk.Frame(main, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        log_header = tk.Frame(right, bg=C["bg3"], height=32)
        log_header.pack(fill="x")
        log_header.pack_propagate(False)
        tk.Label(log_header, text="LIVE LOG", font=self.fn_label,
                 fg=C["muted"], bg=C["bg3"]).pack(side="left", padx=14, pady=8)

        # Filter buttons
        self.log_filter = tk.StringVar(value="ALL")
        for f in ["ALL", "SANCTA", "SIEM", "OLLAMA", "PHENOMENOLOGY"]:
            tk.Radiobutton(
                log_header, text=f, variable=self.log_filter, value=f,
                font=self.fn_label, fg=C["muted"], bg=C["bg3"],
                selectcolor=C["bg"], activebackground=C["bg3"],
                command=self._apply_filter
            ).pack(side="left", padx=4)

        # Clear button
        tk.Button(log_header, text="CLR", font=self.fn_label,
                  fg=C["muted"], bg=C["bg3"], relief="flat",
                  bd=0, padx=8, command=self._clear_log,
                  cursor="hand2").pack(side="right", padx=12)

        # Log text widget
        log_frame = tk.Frame(right, bg=C["bg"])
        log_frame.pack(fill="both", expand=True, padx=1, pady=1)

        self.log_text = tk.Text(
            log_frame,
            bg=C["bg"], fg=C["text"],
            font=self.fn_mono,
            relief="flat", bd=0,
            state="disabled",
            wrap="none",
            insertbackground=C["purple"],
            selectbackground=C["purple_dk"],
        )
        sb = tk.Scrollbar(log_frame, orient="vertical",
                          command=self.log_text.yview,
                          bg=C["bg3"], troughcolor=C["bg"],
                          relief="flat")
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(fill="both", expand=True)

        # Log color tags
        self.log_text.tag_configure("ts",       foreground=C["dim"])
        self.log_text.tag_configure("src_sancta",  foreground=C["purple"])
        self.log_text.tag_configure("src_siem",    foreground=C["teal"])
        self.log_text.tag_configure("src_ollama",  foreground=C["amber"])
        self.log_text.tag_configure("src_curiosity", foreground=C["green"])
        self.log_text.tag_configure("src_phenomenology", foreground=C["teal"])
        self.log_text.tag_configure("src_launcher",  foreground=C["muted"])
        self.log_text.tag_configure("lvl_ERROR", foreground=C["red"])
        self.log_text.tag_configure("lvl_WARN",  foreground=C["amber"])
        self.log_text.tag_configure("lvl_OK",    foreground=C["green"])
        self.log_text.tag_configure("msg",       foreground=C["text"])

        # Status bar
        self.statusbar = tk.Frame(self, bg=C["bg3"], height=24)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)
        self.status_text = tk.StringVar(value="Ready")
        tk.Label(self.statusbar, textvariable=self.status_text,
                 font=self.fn_label, fg=C["muted"],
                 bg=C["bg3"]).pack(side="left", padx=10)

        self._start_time = time.time()

        # Start background network checker — never blocks UI
        threading.Thread(target=_background_net_checker,
                         daemon=True, name="net-checker").start()

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=self.fn_label,
                 fg=C["muted"], bg=C["bg2"]).pack(
            anchor="w", padx=16, pady=(12, 4))

    def _service_row(self, parent, key, name, sub):
        row = tk.Frame(parent, bg=C["bg2"])
        row.pack(fill="x", padx=16, pady=3)

        lbl = tk.Label(row, textvariable=self.status_vars[key],
                       font=self.fn_head, fg=C["muted"], bg=C["bg2"])
        lbl.pack(side="left")
        self.status_labels[key] = lbl

        col = tk.Frame(row, bg=C["bg2"])
        col.pack(side="left", padx=6)
        tk.Label(col, text=name, font=self.fn_label,
                 fg=C["text"], bg=C["bg2"]).pack(anchor="w")
        tk.Label(col, text=sub, font=self.fn_label,
                 fg=C["muted"], bg=C["bg2"]).pack(anchor="w")

    def _btn(self, parent, text, color, command):
        btn = tk.Button(
            parent, text=text,
            font=self.fn_btn, fg=color, bg=C["bg3"],
            relief="flat", bd=0,
            padx=14, pady=8,
            anchor="w", cursor="hand2",
            activebackground=C["border"],
            activeforeground=color,
            command=command
        )
        btn.pack(fill="x", padx=16, pady=2)
        return btn

    def _bind_close(self):
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Log management ──────────────────────────────────────────────────────

    def _drain_logs(self):
        """
        Batch log draining — accumulate up to 50 messages, then do
        ONE widget update. Prevents per-line redraws which stutter tkinter.
        """
        batch = []
        count = 0
        while not log_queue.empty() and count < 50:
            try:
                batch.append(log_queue.get_nowait())
                count += 1
            except queue.Empty:
                break

        if batch:
            self.log_text.configure(state="normal")

            # Trim old lines first (one operation)
            lines = int(self.log_text.index("end-1c").split(".")[0])
            excess = lines - MAX_LOG_LINES + len(batch)
            if excess > 0:
                self.log_text.delete("1.0", f"{excess}.0")

            filt = self.log_filter.get()
            for src, level, msg in batch:
                if filt != "ALL" and src.upper() != filt:
                    continue
                ts = datetime.now().strftime("%H:%M:%S")
                src_tag = f"src_{src.lower()}"
                valid_src_tags = {
                    "src_sancta", "src_siem", "src_ollama",
                    "src_curiosity", "src_phenomenology", "src_launcher"
                }
                self.log_text.insert("end", f"{ts} ", "ts")
                self.log_text.insert(
                    "end", f"[{src.upper():<9}] ",
                    src_tag if src_tag in valid_src_tags else "src_launcher"
                )
                lvl_tag = f"lvl_{level}" if level in ("ERROR", "WARN", "OK") else "msg"
                self.log_text.insert("end", f"{msg}\n", lvl_tag)

            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(100, self._drain_logs)

    def _append_log(self, src: str, level: str, msg: str):
        filt = self.log_filter.get()
        if filt != "ALL" and src.upper() != filt:
            return

        self.log_text.configure(state="normal")

        # Trim old lines
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{lines - MAX_LOG_LINES}.0")

        ts = datetime.now().strftime("%H:%M:%S")
        src_tag = f"src_{src.lower()}"

        self.log_text.insert("end", f"{ts} ", "ts")
        self.log_text.insert("end", f"[{src.upper():<9}] ", src_tag
                             if src_tag in ("src_sancta", "src_siem",
                                            "src_ollama", "src_curiosity",
                                            "src_phenomenology")
                             else "src_launcher")

        lvl_tag = f"lvl_{level}" if level in ("ERROR", "WARN", "OK") else "msg"
        self.log_text.insert("end", f"{msg}\n", lvl_tag)

        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _apply_filter(self):
        pass  # filter applies on next message; could rebuild here if needed

    # ─── Status polling ──────────────────────────────────────────────────────

    def _poll_status(self):
        """
        UI-only poll. Reads from _net_status cache populated by background thread.
        No network calls here — never blocks the UI thread.
        """
        with _net_status_lock:
            ollama_ok = _net_status["ollama"]
            siem_ok   = _net_status["siem"]
            model     = _net_status["ollama_model"]

        # Ollama
        self._set_status("ollama", "green" if ollama_ok else "red")
        self.stat_vars["ollama_m"].set(model or ("running, no model" if ollama_ok else "—"))

        # SIEM
        self._set_status("siem", "green" if siem_ok else "red")

        # Sancta — check process object only (no network)
        sancta_proc = processes.get("sancta")
        if sancta_proc and sancta_proc.poll() is None:
            self._set_status("sancta", "green")
        elif "sancta" in processes:
            self._set_status("sancta", "amber")
        else:
            self._set_status("sancta", "red")

        # Curiosity
        curiosity_proc = processes.get("curiosity")
        if curiosity_proc and curiosity_proc.poll() is None:
            self._set_status("curiosity", "green")
            self.btn_curiosity.configure(text="◈  STOP CURIOSITY", fg=C["red"])
        else:
            if "curiosity" in processes:
                processes.pop("curiosity", None)
            self._set_status("curiosity", "muted")
            self.btn_curiosity.configure(text="◈  RUN CURIOSITY", fg=C["amber"])

        # Phenomenology
        phen_proc = processes.get("phenomenology")
        if phen_proc and phen_proc.poll() is None:
            self._set_status("phenomenology", "green")
            self.btn_phenomenology.configure(
                text="◇  STOP PHENOMENOLOGY BATTERY", fg=C["red"]
            )
        else:
            if "phenomenology" in processes:
                processes.pop("phenomenology", None)
            self._set_status("phenomenology", "muted")
            self.btn_phenomenology.configure(
                text="◇  RUN PHENOMENOLOGY BATTERY", fg=C["teal"]
            )

        # Uptime
        elapsed = int(time.time() - self._start_time)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        self.stat_vars["uptime"].set(f"{h:02d}:{m:02d}:{s:02d}")

        # Restarts
        self.stat_vars["restarts"].set(str(sum(restart_counts.values())))

        self.after(POLL_INTERVAL, self._poll_status)

    def _set_status(self, key: str, color: str):
        colors = {
            "green": C["green"],
            "red":   C["red"],
            "amber": C["amber"],
            "muted": C["muted"],
        }
        c = colors.get(color, C["muted"])
        self.status_vars[key].set("●")
        if key in self.status_labels:
            self.status_labels[key].configure(fg=c)
        if key == "curiosity":
            self.curiosity_status.configure(fg=c)
        if key == "phenomenology":
            self.phenomenology_status.configure(fg=c)

    # ─── Actions ─────────────────────────────────────────────────────────────

    def _start_all(self):
        self.btn_start.configure(state="disabled")
        self.status_text.set("Starting services...")

        def _run():
            log_queue.put(("launcher", "INFO", "─── Starting Sancta stack ───"))

            # 1. Ollama
            log_queue.put(("launcher", "INFO", "Checking Ollama..."))
            if not start_ollama():
                log_queue.put(("launcher", "ERROR",
                               "Ollama failed — check installation"))
                self.after(0, lambda: self.btn_start.configure(state="normal"))
                return

            time.sleep(1)

            # 2. SIEM server (uvicorn from project root)
            if not is_siem_running():
                log_queue.put(("launcher", "INFO", "Starting SIEM server..."))
                start_process(
                    "siem",
                    BACKEND_DIR / "siem_server.py",  # dummy for restart; not used
                    restart=False,
                    args_override=SIEM_CMD,
                    cwd_override=ROOT,
                    env_extra={
                        "SIEM_METRICS_SAFE_MODE": "false",
                        "SIEM_WS_SAFE_MODE": "false",
                    },
                )
                # Wait for it
                for _ in range(10):
                    time.sleep(1)
                    if is_siem_running():
                        log_queue.put(("siem", "OK", "Dashboard ready on :8787"))
                        break
                else:
                    log_queue.put(("siem", "WARN", "SIEM slow to start — continuing"))
            else:
                log_queue.put(("siem", "INFO", "Already running on :8787"))

            # 3. Sancta
            log_queue.put(("launcher", "INFO", "Starting Sancta agent..."))
            start_process("sancta", SANCTA_SCRIPT, restart=True)
            time.sleep(2)

            # 4. Open browser
            log_queue.put(("launcher", "INFO", "Opening dashboard..."))
            time.sleep(1)
            webbrowser.open(SIEM_URL)

            log_queue.put(("launcher", "OK", "All services started ✓"))
            self.after(0, lambda: self.status_text.set("Running"))
            self.after(0, lambda: self.btn_start.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _stop_all(self):
        self.status_text.set("Stopping...")
        log_queue.put(("launcher", "INFO", "─── Stopping all services ───"))

        def _run():
            stop_process("phenomenology")
            stop_process("curiosity")
            stop_process("sancta")
            stop_process("siem")
            # Don't stop Ollama — it's shared, let user manage it
            log_queue.put(("launcher", "OK",
                           "Services stopped (Ollama left running)"))
            self.after(0, lambda: self.status_text.set("Stopped"))

        threading.Thread(target=_run, daemon=True).start()

    def _toggle_curiosity(self):
        global curiosity_running
        proc = processes.get("curiosity")
        if proc and proc.poll() is None:
            # Stop
            log_queue.put(("curiosity", "INFO", "Stopping curiosity run..."))
            stop_process("curiosity")
            curiosity_running = False
        else:
            # Start
            if not is_ollama_running():
                log_queue.put(("curiosity", "ERROR",
                               "Ollama must be running first"))
                return
            log_queue.put(("curiosity", "INFO",
                           "─── Starting curiosity run ───"))
            curiosity_running = True
            start_process(
                "curiosity", SANCTA_SCRIPT,
                extra_args=[CURIOSITY_FLAG],
                restart=False
            )

    def _toggle_phenomenology(self):
        proc = processes.get("phenomenology")
        if proc and proc.poll() is None:
            # Stop
            log_queue.put(("phenomenology", "INFO",
                           "Stopping phenomenology battery..."))
            stop_process("phenomenology")
        else:
            # Start
            log_queue.put(("phenomenology", "INFO",
                           "─── Running phenomenology attack battery (11 vectors) ───"))
            start_process(
                "phenomenology", SANCTA_SCRIPT,
                extra_args=[PHENOMENOLOGY_FLAG],
                restart=False,
            )

    def _on_close(self):
        log_queue.put(("launcher", "INFO", "Shutting down..."))
        stop_all()
        self.after(800, self.destroy)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = SanctaLauncher()
    log_queue.put(("launcher", "OK",
                   "Sancta Control Center ready — click START ALL"))
    app.mainloop()


if __name__ == "__main__":
    main()
