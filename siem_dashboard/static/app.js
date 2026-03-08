/* ══════════════════════════════════════════════
   Auth — Bearer token when SIEM_AUTH_TOKEN is set
   ══════════════════════════════════════════════ */
const AUTH_STORAGE_KEY = "siem_auth_token";

function getAuthToken() {
  return sessionStorage.getItem(AUTH_STORAGE_KEY);
}

function setAuthToken(t) {
  if (t) sessionStorage.setItem(AUTH_STORAGE_KEY, t);
  else sessionStorage.removeItem(AUTH_STORAGE_KEY);
}

function authHeaders() {
  const t = getAuthToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function authFetchWithRetry(url, opts = {}, maxRetries = 3) {
  const headers = { ...opts.headers, ...authHeaders() };
  let lastErr;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const res = await fetch(url, { ...opts, headers });
      if (res.status === 401 && getAuthToken()) {
        setAuthToken(null);
        showAuthOverlay();
        if (wsLive) {
          wsLive.close();
          wsLive = null;
        }
      }
      return res;
    } catch (e) {
      lastErr = e;
      if (attempt < maxRetries - 1) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

async function authFetch(url, opts = {}) {
  const headers = { ...opts.headers, ...authHeaders() };
  const res = await fetch(url, { ...opts, headers });
  if (res.status === 401 && getAuthToken()) {
    setAuthToken(null);
    showAuthOverlay();
    if (wsLive) {
      wsLive.close();
      wsLive = null;
    }
  }
  return res;
}

/* ══════════════════════════════════════════════
   Audio Manager — browser-side sound playback
   ══════════════════════════════════════════════ */
let audioEnabled = true;
let soundManifest = null;
let lastSoundTime = 0;
const SOUND_COOLDOWN_MS = 800;

async function initAudio() {
  try {
    const res = await authFetchWithRetry("/api/sounds/manifest");
    const data = await res.json();
    if (data.ok && data.manifest) {
      soundManifest = data.manifest;
    }
  } catch (_) {}
}

function mapEventToCategory(ev) {
  const src = ev.source || "";
  const name = ev.event || "";
  if (src === "security" && /inject|block|reject|suspicious/.test(name)) return "security.alert";
  if (src === "redteam" && /escalat|reward/.test(name)) return "security.alert";
  if (/published|complete|cycle_done/.test(name)) return "task.complete";
  if (/mood_shift|mood_change|epistemic_state/.test(name) && ev.mood) return "mood.change";
  return null;
}

function playEventSound(ev) {
  if (!audioEnabled || !soundManifest) return;
  const now = Date.now();
  if (now - lastSoundTime < SOUND_COOLDOWN_MS) return;

  const category = mapEventToCategory(ev);
  if (!category) return;

  const packName = soundManifest.default_pack || "office_peon";
  const pack = (soundManifest.packs || {})[packName];
  if (!pack) return;

  const sounds = pack[category];
  if (!sounds || !sounds.length) return;

  const file = sounds[Math.floor(Math.random() * sounds.length)];
  const audio = new Audio("/sounds/" + file);
  audio.volume = 0.55;
  audio.play().catch(() => {});
  lastSoundTime = now;
}

function playSessionSound() {
  if (!audioEnabled || !soundManifest) return;
  const packName = soundManifest.default_pack || "office_peon";
  const pack = (soundManifest.packs || {})[packName];
  if (!pack) return;
  const sounds = pack["session.start"];
  if (!sounds || !sounds.length) return;
  const file = sounds[Math.floor(Math.random() * sounds.length)];
  const audio = new Audio("/sounds/" + file);
  audio.volume = 0.55;
  audio.play().catch(() => {});
}

/* ══════════════════════════════════════════════
   DOM References
   ══════════════════════════════════════════════ */
const panes = {
  feed: document.getElementById("pane-feed"),
  alerts: document.getElementById("pane-alerts"),
  system: document.getElementById("pane-system"),
  activity: document.getElementById("pane-activity"),
};

const els = {
  agent: document.getElementById("m-agent"),
  pid: document.getElementById("m-pid"),
  mood: document.getElementById("m-mood"),
  inj: document.getElementById("m-inj"),
  sanitized: document.getElementById("m-sanitized"),
  reward: document.getElementById("m-reward"),
  fp: document.getElementById("m-fp"),
  belief: document.getElementById("m-belief"),
};

const modeSelect = document.getElementById("mode");
const scrollLock = document.getElementById("scrollLock");
const tabs = Array.from(document.querySelectorAll(".tab"));
const feedToggle = document.getElementById("feed-toggle");
const feedStatusIcon = document.getElementById("feed-status-icon");
const feedStatusLabel = document.getElementById("feed-status-label");
const timeFrom = document.getElementById("time-from");
const timeTo = document.getElementById("time-to");
const timeApply = document.getElementById("time-apply");
const timeClear = document.getElementById("time-clear");

const injEls = {
  hidden: document.getElementById("inj-hidden"),
  unicode: document.getElementById("inj-unicode"),
  patterns: document.getElementById("inj-patterns"),
  rate: document.getElementById("inj-rate"),
};

const epiEls = {
  conf: document.getElementById("epi-conf"),
  ent: document.getElementById("epi-ent"),
  anth: document.getElementById("epi-anth"),
  over: document.getElementById("epi-over"),
};

const footerEls = {
  total: document.getElementById("f-total"),
  sanitizations: document.getElementById("f-sanitizations"),
  conf: document.getElementById("f-conf"),
  ent: document.getElementById("f-ent"),
  threat: document.getElementById("f-threat"),
};

let activeTab = "all";
let feedPaused = false;
let wsLive = null;
let bufferedEvents = [];
const MAX_BUFFER = 1200;
const renderedEventIds = new Set();
const MAX_RENDERED_IDS = 500;

const counters = {
  totalEvents: 0,
  injectionAttempts: 0,
  sanitizations: 0,
  unicodeCleanEvents: 0,
  hiddenCharsStripped: 0,
  patternMatches: 0,
  epistemicN: 0,
  sumConf: 0,
  sumEntropy: 0,
  sumAnth: 0,
};

let lastStatus = {
  running: null,
  suspended: null,
  pid: null,
};

let lastActivitySnapshot = "";

/* ══════════════════════════════════════════════
   Helpers
   ══════════════════════════════════════════════ */
function esc(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function levelClass(level) {
  const l = (level || "").toUpperCase();
  if (l === "INFO") return "lvl-info";
  if (l === "DEBUG") return "lvl-debug";
  if (l === "WARN" || l === "WARNING") return "lvl-warn";
  if (l === "ERROR") return "lvl-error";
  if (l === "ALERT" || l === "CRITICAL") return "lvl-alert";
  return "lvl-info";
}

function appendLine(pane, htmlLine) {
  pane.insertAdjacentHTML("afterbegin", htmlLine + "\n");
  if (scrollLock.checked) {
    pane.scrollTop = 0;
  }
}

/* ══════════════════════════════════════════════
   Tabs
   ══════════════════════════════════════════════ */
function setTab(tab) {
  activeTab = tab;
  for (const b of tabs) b.classList.toggle("is-active", b.dataset.tab === tab);
  appendLine(panes.system, `<span class="lvl-debug">[SYS] tab=${esc(tab)}</span>`);
}

for (const b of tabs) {
  b.addEventListener("click", () => setTab(b.dataset.tab));
}

window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) return;
  const map = { "1": "all", "2": "security", "3": "redteam", "4": "philosophy", "5": "alerts", "6": "system" };
  if (map[e.key]) setTab(map[e.key]);
});

/* ══════════════════════════════════════════════
   Event Processing
   ══════════════════════════════════════════════ */
function summarizeEvent(ev) {
  const ts = ev.ts ? ev.ts.replace("T", " ").replace("Z", "") : "";
  const src = (ev.source || "unknown").toUpperCase();
  const name = ev.event || "";
  const cid = ev.correlation_id || (ev.data && ev.data.correlation_id) || null;

  let msg = ev.message || "";
  if (!msg) {
    const author = ev.author || (ev.data && ev.data.author) || "";
    const preview = ev.preview || (ev.data && ev.data.preview) || "";
    if (author && preview) msg = `author=${author} | ${preview}`;
    else if (preview) msg = preview;
  }

  let extra = "";
  const ac = ev.attack_complexity || (ev.data && ev.data.attack_complexity);
  if (ac && ac.complexity_label) extra += ` | complexity=${ac.complexity_label}(${ac.pattern_count}/${ac.class_count})`;
  const pers = ev.attacker_persistence || (ev.data && ev.data.attacker_persistence);
  if (typeof pers === "number") extra += ` | persistence=${pers.toFixed(3)}`;
  if (cid) extra += ` | corr=${cid}`;

  return `${ts} [${src}] ${name} ${msg}${extra}`.trim();
}

function matchesActiveTab(ev) {
  if (activeTab === "all") return true;
  if (activeTab === "alerts") return maybeAlert(ev);
  return ev.source === activeTab;
}

function maybeAlert(ev) {
  const src = ev.source;
  const name = ev.event || "";
  if (src === "security" && (name === "injection_blocked" || name === "suspicious_block")) return true;
  if (src === "redteam" && name === "redteam_escalate") return true;
  const ac = ev.attack_complexity || (ev.data && ev.data.attack_complexity);
  if (ac && ac.complexity_label === "multi_pattern_chained_intent") return true;
  return false;
}

function eventId(ev) {
  const ts = ev.ts || ev.timestamp || "";
  const src = ev.source || "";
  const name = ev.event || "";
  const preview = (ev.preview || (ev.data && ev.data.preview) || "").slice(0, 80);
  return `${ts}_${src}_${name}_${preview}`;
}

function renderEvent(ev) {
  const id = eventId(ev);
  if (renderedEventIds.has(id)) return;
  renderedEventIds.add(id);
  if (renderedEventIds.size > MAX_RENDERED_IDS) {
    const arr = [...renderedEventIds];
    arr.splice(0, 100);
    renderedEventIds.clear();
    arr.forEach((x) => renderedEventIds.add(x));
  }
  playEventSound(ev);
  counters.totalEvents += 1;
  bufferedEvents.push(ev);
  if (bufferedEvents.length > MAX_BUFFER) bufferedEvents = bufferedEvents.slice(-MAX_BUFFER);

  if (ev.source === "security") {
    if (ev.event === "unicode_clean") {
      counters.unicodeCleanEvents += 1;
      const n = ev.stripped_hidden_chars ?? (ev.data && ev.data.stripped_hidden_chars) ?? 0;
      counters.hiddenCharsStripped += Number(n) || 0;
    }
    if (ev.event === "input_reject" || ev.event === "injection_blocked" || ev.event === "suspicious_block") {
      counters.injectionAttempts += 1;
    }
    if (ev.event === "input_reject" || ev.event === "injection_blocked" || ev.event === "output_redact") {
      counters.sanitizations += 1;
    }
    const pm = ev.patterns_matched ?? (ev.data && ev.data.patterns_matched);
    if (typeof pm === "number") counters.patternMatches += pm;
  }

  if (ev.source === "philosophy" && ev.event === "epistemic_state") {
    const es = ev.epistemic_state || (ev.data && ev.data.epistemic_state);
    if (es) {
      const c = Number(es.confidence_score);
      const ent = Number(es.uncertainty_entropy);
      const a = Number(es.anthropomorphism_index);
      if (!Number.isNaN(c) && !Number.isNaN(ent) && !Number.isNaN(a)) {
        counters.epistemicN += 1;
        counters.sumConf += c;
        counters.sumEntropy += ent;
        counters.sumAnth += a;
      }
    }
  }

  const lvl = ev.level || "INFO";
  const cls = levelClass(lvl);
  const text = summarizeEvent(ev);
  const line = `<span class="${cls} dim">${esc(text)}</span>`;
  if (!feedPaused && matchesActiveTab(ev)) appendLine(panes.feed, line);
  if (maybeAlert(ev)) {
    appendLine(panes.alerts, `<span class="lvl-alert">${esc(text)}</span>`);
  }

  injEls.hidden.textContent = String(counters.hiddenCharsStripped);
  injEls.unicode.textContent = String(counters.unicodeCleanEvents);
  injEls.patterns.textContent = String(counters.patternMatches);
  injEls.rate.textContent = counters.injectionAttempts > 0
    ? `${Math.round((counters.sanitizations / counters.injectionAttempts) * 100)}%`
    : "\u2014";

  const n = counters.epistemicN;
  const meanConf = n ? (counters.sumConf / n) : null;
  const meanEnt = n ? (counters.sumEntropy / n) : null;
  const meanAnth = n ? (counters.sumAnth / n) : null;
  epiEls.conf.textContent = meanConf == null ? "\u2014" : meanConf.toFixed(3);
  epiEls.ent.textContent = meanEnt == null ? "\u2014" : meanEnt.toFixed(3);
  epiEls.anth.textContent = meanAnth == null ? "\u2014" : meanAnth.toFixed(3);
  epiEls.over.textContent = meanConf == null ? "\u2014" : (1 - meanConf).toFixed(3);

  footerEls.total.textContent = String(counters.totalEvents);
  footerEls.sanitizations.textContent = String(counters.sanitizations);
  footerEls.conf.textContent = meanConf == null ? "\u2014" : meanConf.toFixed(3);
  footerEls.ent.textContent = meanEnt == null ? "\u2014" : meanEnt.toFixed(3);

  const threat = counters.injectionAttempts >= 10 ? "HIGH" : (counters.injectionAttempts >= 3 ? "MED" : "LOW");
  footerEls.threat.textContent = threat;
  footerEls.threat.className = "v " + (threat === "HIGH" ? "threat-high" : threat === "MED" ? "threat-med" : "threat-low");
}

/* ══════════════════════════════════════════════
   Metrics
   ══════════════════════════════════════════════ */
function renderMetrics(m) {
  els.agent.textContent = m.running ? (m.suspended ? "PAUSED" : "RUNNING") : "STOPPED";
  els.pid.textContent = m.pid || "\u2014";
  els.mood.textContent = m.agent_mood || "\u2014";
  els.inj.textContent = m.injection_attempts_detected ?? 0;
  els.sanitized.textContent = m.sanitized_payload_count ?? 0;
  els.reward.textContent = m.reward_score_rolling_sum ?? 0;
  els.fp.textContent = (m.false_positive_rate == null) ? "\u2014" : String(m.false_positive_rate);
  els.belief.textContent = (m.belief_confidence == null) ? "\u2014" : m.belief_confidence.toFixed(3);

  const changed =
    lastStatus.running !== m.running ||
    lastStatus.suspended !== m.suspended ||
    lastStatus.pid !== (m.pid || null);

  if (changed) {
    if (m.running && !m.suspended && !lastStatus.running) playSessionSound();
    const sys = `agent_running=${m.running} suspended=${m.suspended} pid=${m.pid || "\u2014"}`;
    appendLine(panes.system, `<span class="lvl-debug">[SYS] ${esc(sys)}</span>`);
    lastStatus = {
      running: m.running,
      suspended: m.suspended,
      pid: m.pid || null,
    };
  }
}

/* ══════════════════════════════════════════════
   Agent Activity
   ══════════════════════════════════════════════ */
let agentActivityFailCount = 0;
const AGENT_ACTIVITY_BASE_MS = 4000;
const AGENT_ACTIVITY_MAX_MS = 60000;

async function refreshAgentActivity() {
  try {
    if (!panes.activity) return;

    const res = await authFetchWithRetry("/api/agent-activity");
    if (!res.ok) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_http_${res.status}</span>`);
      return;
    }
    const data = await res.json();
    if (!data || data.ok === false || !Array.isArray(data.lines)) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_invalid_response</span>`);
      return;
    }

    const joined = data.lines.join("\n");
    if (joined === lastActivitySnapshot) return;
    lastActivitySnapshot = joined;

    panes.activity.textContent = "";
    if (!data.lines.length) {
      panes.activity.textContent = "(no recent agent activity)";
      return;
    }
    const reversed = data.lines.slice().reverse();
    for (const line of reversed) {
      panes.activity.insertAdjacentHTML("beforeend", esc(line) + "\n");
    }
    panes.activity.scrollTop = 0;
    agentActivityFailCount = 0;
  } catch (_) {
    agentActivityFailCount++;
    if (panes.system) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_fetch_error</span>`);
    }
  }
}

function scheduleAgentActivityRefresh() {
  const backoff = Math.min(AGENT_ACTIVITY_BASE_MS * Math.pow(2, Math.min(agentActivityFailCount, 4)), AGENT_ACTIVITY_MAX_MS);
  setTimeout(async () => {
    await refreshAgentActivity();
    scheduleAgentActivityRefresh();
  }, backoff);
}

/* ══════════════════════════════════════════════
   Actions
   ══════════════════════════════════════════════ */
async function post(path, body = null) {
  const res = await authFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  });
  return await res.json();
}

async function runAction(btnId, path, body, level) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = true;
  btn.classList.remove("btn-pulse-ok", "btn-pulse-err");
  try {
    const res = await post(path, body);
    const ok = res && res.ok !== false && !res.error;
    const cls = ok ? "btn-pulse-ok" : "btn-pulse-err";
    btn.classList.add(cls);
    setTimeout(() => btn.classList.remove(cls), 260);
    const lvl = ok ? level : "lvl-error";
    appendLine(panes.system, `<span class="${lvl}">[CMD] ${esc(JSON.stringify(res))}</span>`);
  } catch (e) {
    btn.classList.add("btn-pulse-err");
    setTimeout(() => btn.classList.remove("btn-pulse-err"), 260);
    appendLine(panes.system, `<span class="lvl-error">[ERR] ${esc(String(e))}</span>`);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("start").addEventListener("click", async () => {
  const mode = modeSelect.value;
  await runAction("start", "/api/agent/start", { mode }, "lvl-info");
});
document.getElementById("pause").addEventListener("click", async () => {
  await runAction("pause", "/api/agent/pause", {}, "lvl-warn");
});
document.getElementById("resume").addEventListener("click", async () => {
  await runAction("resume", "/api/agent/resume", {}, "lvl-info");
});
document.getElementById("restart").addEventListener("click", async () => {
  const mode = modeSelect.value;
  await runAction("restart", "/api/agent/restart", { mode }, "lvl-alert");
});
document.getElementById("kill").addEventListener("click", async () => {
  await runAction("kill", "/api/agent/kill", {}, "lvl-error");
});

/* ══════════════════════════════════════════════
   Feed Controls
   ══════════════════════════════════════════════ */
function updateFeedStatus(kind) {
  feedStatusIcon.classList.remove("status-live", "status-paused", "status-off", "status-polling");
  if (kind === "live") {
    feedStatusIcon.classList.add("status-live");
    feedStatusLabel.textContent = "LIVE";
  } else if (kind === "paused") {
    feedStatusIcon.classList.add("status-paused");
    feedStatusLabel.textContent = "PAUSED";
  } else if (kind === "polling") {
    feedStatusIcon.classList.add("status-polling");
    feedStatusLabel.textContent = "POLLING";
  } else {
    feedStatusIcon.classList.add("status-off");
    feedStatusLabel.textContent = "STOPPED";
  }
}

function applyTimeFilter() {
  const from = timeFrom.value ? new Date(timeFrom.value).getTime() : null;
  const to = timeTo.value ? new Date(timeTo.value).getTime() : null;
  panes.feed.textContent = "";
  const filtered = [];
  for (const ev of bufferedEvents) {
    if (!ev.ts) continue;
    const ts = Date.parse(ev.ts);
    if (Number.isNaN(ts)) continue;
    if (from !== null && ts < from) continue;
    if (to !== null && ts > to) continue;
    if (!matchesActiveTab(ev)) continue;
    filtered.push(ev);
  }
  for (let i = filtered.length - 1; i >= 0; i--) {
    const ev = filtered[i];
    const lvl = ev.level || "INFO";
    const cls = levelClass(lvl);
    const text = summarizeEvent(ev);
    const line = `<span class="${cls} dim">${esc(text)}</span>`;
    panes.feed.insertAdjacentHTML("beforeend", line + "\n");
  }
  panes.feed.scrollTop = 0;
}

timeApply.addEventListener("click", () => {
  feedPaused = true;
  feedToggle.textContent = "\u25b9 LIVE";
  updateFeedStatus("paused");
  applyTimeFilter();
});

timeClear.addEventListener("click", () => {
  timeFrom.value = "";
  timeTo.value = "";
  panes.feed.textContent = "";
  feedPaused = false;
  feedToggle.textContent = "\u23f8 PAUSE";
  updateFeedStatus(wsLive ? "live" : "off");
  for (let i = bufferedEvents.length - 1; i >= 0; i--) {
    const ev = bufferedEvents[i];
    if (!matchesActiveTab(ev)) continue;
    const lvl = ev.level || "INFO";
    const cls = levelClass(lvl);
    const text = summarizeEvent(ev);
    const line = `<span class="${cls} dim">${esc(text)}</span>`;
    panes.feed.insertAdjacentHTML("beforeend", line + "\n");
  }
  panes.feed.scrollTop = 0;
});

feedToggle.addEventListener("click", () => {
  feedPaused = !feedPaused;
  if (feedPaused) {
    feedToggle.textContent = "\u25b9 LIVE";
    updateFeedStatus("paused");
  } else {
    feedToggle.textContent = "\u23f8 PAUSE";
    updateFeedStatus(wsLive ? "live" : "off");
  }
});

/* ══════════════════════════════════════════════
   WebSocket
   ══════════════════════════════════════════════ */
let wsReconnectCount = 0;
const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;

function connect() {
  const token = getAuthToken();
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`ws://${location.host}/ws/live${qs}`);
  wsLive = ws;
  ws.onopen = () => {
    wsReconnectCount = 0;
    appendLine(panes.system, `<span class="lvl-info">[WS] connected</span>`);
    if (!feedPaused) updateFeedStatus("live");
  };
  ws.onclose = (ev) => {
    if (ev.code === 4001) {
      setAuthToken(null);
      showAuthOverlay();
      wsLive = null;
      updateFeedStatus("off");
      return;
    }
    wsReconnectCount++;
    const delay = Math.min(WS_RECONNECT_BASE_MS * Math.pow(2, Math.min(wsReconnectCount - 1, 4)), WS_RECONNECT_MAX_MS);
    appendLine(panes.system, `<span class="lvl-warn">[WS] disconnected — reconnecting in ${Math.round(delay / 1000)}s</span>`);
    wsLive = null;
    updateFeedStatus("off");
    setTimeout(connect, delay);
  };
  ws.onerror = () => {};
  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      if (data.type === "event") renderEvent(data.event);
      if (data.type === "metrics") renderMetrics(data.metrics);
    } catch (_) {}
  };
}

/* ══════════════════════════════════════════════
   Chat — Talk to the agent
   ══════════════════════════════════════════════ */
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatEnrich = document.getElementById("chat-enrich");
const CHAT_SESSION_KEY = "sancta_chat_session_id";

function appendChatMessage(role, text) {
  if (!chatMessages) return;
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  const label = role === "user" ? "You" : "Sancta";
  div.innerHTML = `<span class="chat-label">${esc(label)}</span><br>${esc(text)}`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatMessage() {
  if (!chatInput || !chatSend) return;
  const msg = (chatInput.value || "").trim();
  if (!msg) return;

  chatSend.disabled = true;
  chatInput.value = "";
  appendChatMessage("user", msg);

  try {
    const res = await authFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        message: msg,
        enrich: chatEnrich ? chatEnrich.checked : true,
        session_id: localStorage.getItem(CHAT_SESSION_KEY) || undefined,
      }),
    });

    let data;
    try {
      data = await res.json();
    } catch (_) {
      data = { error: `HTTP ${res.status}`, detail: "Invalid response" };
    }

    if (res.status === 404) {
      appendChatMessage("agent", "Chat endpoint not found (404). Restart the SIEM server to load the chat feature.");
      if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[CHAT] 404 — restart SIEM: python -m uvicorn siem_dashboard.server:app</span>`);
      return;
    }
    if (res.status === 401) {
      appendChatMessage("agent", "Unauthorized. Enter the SIEM token to use chat.");
      return;
    }
    if (res.status >= 400) {
      if (data?.session_id) localStorage.setItem(CHAT_SESSION_KEY, data.session_id);
      const err = data?.error || data?.detail || `HTTP ${res.status}`;
      appendChatMessage("agent", `Error: ${err}`);
      if (panes.system) appendLine(panes.system, `<span class="lvl-error">[CHAT] ${res.status}: ${esc(err)}</span>`);
      return;
    }

    if (data?.session_id) {
      localStorage.setItem(CHAT_SESSION_KEY, data.session_id);
    }
    if (data && data.ok && data.reply) {
      appendChatMessage("agent", data.reply);
      if (data.enriched && panes.system) {
        appendLine(panes.system, `<span class="lvl-info">[CHAT] Exchange added to knowledge base</span>`);
      }
    } else {
      const err = data?.error || data?.detail || "Failed to get reply";
      appendChatMessage("agent", `Error: ${err}`);
      if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[CHAT] ${esc(err)}</span>`);
    }
  } catch (e) {
    appendChatMessage("agent", `Error: ${esc(String(e))}`);
    if (panes.system) appendLine(panes.system, `<span class="lvl-error">[CHAT] ${esc(String(e))}</span>`);
  } finally {
    chatSend.disabled = false;
    chatInput?.focus();
  }
}

if (chatSend) {
  chatSend.addEventListener("click", sendChatMessage);
}
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
}

/* ══════════════════════════════════════════════
   Sound Toggle
   ══════════════════════════════════════════════ */
const soundBtn = document.getElementById("sound-toggle");
if (soundBtn) {
  soundBtn.addEventListener("click", () => {
    audioEnabled = !audioEnabled;
    soundBtn.textContent = audioEnabled ? "\uD83D\uDD0A SOUND" : "\uD83D\uDD07 MUTED";
    soundBtn.classList.toggle("btn-dim", !audioEnabled);
  });
}

/* ══════════════════════════════════════════════
   Auth overlay
   ══════════════════════════════════════════════ */
let initRan = false;

function showAuthOverlay() {
  const el = document.getElementById("auth-overlay");
  if (el) {
    el.style.display = "flex";
    document.getElementById("auth-token")?.focus();
  }
}

function hideAuthOverlay() {
  const el = document.getElementById("auth-overlay");
  if (el) el.style.display = "none";
}

function setAuthError(msg) {
  const el = document.getElementById("auth-error");
  if (el) el.textContent = msg || "";
}

/* ══════════════════════════════════════════════
   Matrix rain — low-opacity background for live feed
   ══════════════════════════════════════════════ */
const MATRIX_CHARS = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン";
const MATRIX_FONT_SIZE = 12;
const MATRIX_COLUMN_SPACING = 18;

function initMatrixRain() {
  const canvas = document.getElementById("matrix-rain");
  const container = canvas?.closest(".panel.feed");
  if (!canvas || !container) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let rafId = null;
  let columns = [];
  let lastResize = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) return;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);

    const colCount = Math.ceil(w / MATRIX_COLUMN_SPACING);
    columns = columns.slice(0, colCount);
    while (columns.length < colCount) {
      columns.push({
        y: Math.random() * h,
        speed: 0.16 + Math.random() * 0.24,
        chars: Array.from({ length: 8 }, () => MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)]),
      });
    }
  }

  function draw() {
    if (document.hidden) {
      rafId = requestAnimationFrame(draw);
      return;
    }
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) {
      rafId = requestAnimationFrame(draw);
      return;
    }

    ctx.fillStyle = "rgba(0, 2, 1, 0.06)";
    ctx.fillRect(0, 0, w, h);

    ctx.font = `${MATRIX_FONT_SIZE}px 'Courier New', monospace`;
    const colCount = Math.ceil(w / MATRIX_COLUMN_SPACING);

    for (let i = 0; i < colCount; i++) {
      const col = columns[i] || { y: 0, speed: 0.05, chars: [] };
      col.y += col.speed;
      if (col.y > h + 50) col.y = -30;

      for (let j = 0; j < col.chars.length; j++) {
        const y = col.y - j * MATRIX_FONT_SIZE;
        if (y < -MATRIX_FONT_SIZE || y > h) continue;
        const alpha = 1 - j / col.chars.length;
        ctx.fillStyle = `rgba(0, 255, 65, ${alpha * 0.7})`;
        ctx.fillText(col.chars[j], i * MATRIX_COLUMN_SPACING, y);
      }
    }
    rafId = requestAnimationFrame(draw);
  }

  resize();
  draw();

  const ro = new ResizeObserver(() => {
    const now = Date.now();
    if (now - lastResize < 100) return;
    lastResize = now;
    resize();
  });
  ro.observe(container);
}

let wsSafeMode = false;

async function runInit() {
  if (initRan) {
    if (!wsSafeMode) connect();
    return;
  }
  initRan = true;
  await new Promise((r) => setTimeout(r, 150));
  initMatrixRain();
  await initAudio();
  // Fetch metrics immediately so MOOD, INJ, REWARD show before WebSocket connects
  try {
    const statusRes = await authFetchWithRetry("/api/status", {}, 1);
    if (statusRes.ok) {
      const d = await statusRes.json();
      if (d.metrics) renderMetrics(d.metrics);
      wsSafeMode = !!d.ws_safe_mode;
    }
  } catch (_) {}
  if (wsSafeMode) {
    appendLine(panes.system, `<span class="lvl-info">[SYS] polling mode (WebSocket disabled for stability)</span>`);
    updateFeedStatus("polling");
    pollStatusForMetrics();
  } else {
    connect();
  }
  await refreshAgentActivity();
  scheduleAgentActivityRefresh();
  pollLiveEvents();
}

function pollStatusForMetrics() {
  const poll = async () => {
    try {
      const res = await authFetchWithRetry("/api/status", {}, 1);
      if (!res.ok) return;
      const d = await res.json();
      if (d.metrics) renderMetrics(d.metrics);
    } catch (_) {}
    setTimeout(poll, 1500);
  };
  poll();
}

async function pollLiveEvents() {
  const poll = async () => {
    try {
      const res = await authFetchWithRetry("/api/live-events", {}, 1);
      if (!res.ok) return;
      const data = await res.json();
      if (data.ok && Array.isArray(data.events)) {
        for (const ev of data.events) {
          renderEvent(ev);
        }
      }
    } catch (_) {}
    setTimeout(poll, 2500);
  };
  poll();
}

async function onAuthSubmit() {
  const input = document.getElementById("auth-token");
  const token = (input?.value || "").trim();
  if (!token) {
    setAuthError("Enter a token");
    return;
  }
  setAuthError("");
  const res = await fetch("/api/auth/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  const data = await res.json();
  if (data && data.ok) {
    setAuthToken(token);
    hideAuthOverlay();
    runInit();
  } else {
    setAuthError("Invalid token");
  }
}

async function bootstrap() {
  setTab("all");

  document.getElementById("auth-submit")?.addEventListener("click", onAuthSubmit);
  document.getElementById("auth-token")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onAuthSubmit();
  });

  const statusRes = await fetch("/api/auth/status");
  let authRequired = false;
  try {
    const status = await statusRes.json();
    authRequired = !!(status && status.auth_required);
  } catch (_) {}

  if (authRequired && !getAuthToken()) {
    showAuthOverlay();
    return;
  }

  runInit();
}

/* ══════════════════════════════════════════════
   Init
   ══════════════════════════════════════════════ */
bootstrap();

