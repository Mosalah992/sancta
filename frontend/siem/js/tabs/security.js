/* ─── Security Tab ───────────────────────────────────────── */
import { S, evSeverity } from '../state.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _safeStr(v) {
  if (v == null) return '';
  return typeof v === 'string' ? v : (typeof v === 'object' ? '' : String(v));
}

function _inner(ev) {
  return (ev && typeof ev.event === 'object') ? ev.event : ev;
}

function renderEvent(ev) {
  if (!ev || typeof ev !== 'object') return '';
  const inner = _inner(ev);
  const sev = evSeverity(ev);
  const cls = sev === 'block' ? 'ev-block' : sev === 'warn' ? 'ev-warn' : 'ev-info';
  const ts = _safeStr(inner?.ts ?? ev.ts ?? inner?.timestamp ?? ev.timestamp).replace('T', ' ').slice(0, 19);
  const tag = _safeStr(inner?.source ?? ev.source ?? inner?.event ?? ev.event ?? 'sec').toUpperCase().slice(0, 8);
  const msg = _safeStr(inner?.message ?? ev.message ?? inner?.preview ?? ev.preview ?? inner?.event ?? ev.event);
  const tagCls = sev === 'block' ? 'tag-block' : sev === 'warn' ? 'tag-warn' : 'tag-info';
  return `<div class="term-event ${cls}">
    <span class="term-ts">${esc(ts)}</span>
    <span class="term-tag ${tagCls}">${esc(tag)}</span>
    <span class="term-msg">${esc(msg)}</span>
  </div>`;
}

export function init() {}

export function refresh(adversaryData) {
  _updateBars();
  _renderThreats(adversaryData);
  _renderPatterns(adversaryData);
  _renderRedTeam(adversaryData);
  _renderInjectors(adversaryData);
}

function _updateBars() {
  const secDefenseFill = document.getElementById('sec-defense-fill');
  const secDefenseVal  = document.getElementById('sec-defense-val');
  const secFpFill      = document.getElementById('sec-fp-fill');
  const secFpVal       = document.getElementById('sec-fp-val');

  const dr = S.defenseRate != null ? Math.min(100, Math.max(0, S.defenseRate * 100)) : 0;
  const fp = S.fpRate     != null ? Math.min(100, Math.max(0, S.fpRate * 100)) : 0;

  if (secDefenseFill) secDefenseFill.style.width = `${dr.toFixed(1)}%`;
  if (secDefenseVal)  secDefenseVal.textContent  = S.defenseRate != null ? `${Math.round(dr)}%` : '—';
  if (secFpFill)      secFpFill.style.width      = `${fp.toFixed(1)}%`;
  if (secFpVal)       secFpVal.textContent        = S.fpRate != null ? `${fp.toFixed(1)}%` : '—';
}

function _renderThreats(adv) {
  const feed  = document.getElementById('sec-threats-feed');
  const count = document.getElementById('sec-threat-count');
  if (!feed) return;

  // Prefer adversary recent_attacks, fall back to event buffer
  const recent = adv?.recent_attacks;
  if (recent?.length) {
    const html = recent.slice(0, 40).map(ev => {
      const ts  = esc((ev.ts || '').slice(0, 19));
      const tag = ev.action === 'blocked' ? 'BLOCK' : 'IOC';
      const msg = esc(ev.preview || ev.event || '—');
      const cls = ev.action === 'blocked' ? 'ev-block tag-block' : 'ev-warn tag-warn';
      return `<div class="term-event ${cls.split(' ')[0]}">
        <span class="term-ts">${ts}</span>
        <span class="term-tag ${cls.split(' ')[1]}">${tag}</span>
        <span class="term-msg">${msg}</span>
      </div>`;
    }).join('');
    feed.innerHTML = html || '<div class="term-event ev-info"><span class="term-msg">No threats</span></div>';
    if (count) count.textContent = String(adv?.total_attacks ?? recent.length);
  } else {
    // Fall back to event buffer
    const threats = (S.events || []).filter(e => evSeverity(e) === 'block' || evSeverity(e) === 'warn');
    feed.innerHTML = threats.slice(0, 50).map(renderEvent).join('') ||
      '<div class="term-event ev-info"><span class="term-msg">No threats</span></div>';
    if (count) count.textContent = String(threats.length);
  }
}

function _renderPatterns(adv) {
  const el = document.getElementById('sec-patterns-body');
  if (!el) return;

  const stats = adv?.defense_stats;
  if (stats) {
    const total = Object.values(stats).reduce((a, b) => a + b, 0) || 1;
    const items = [
      { k: 'blocked',           color: 'var(--red)',     val: stats.blocked || 0 },
      { k: 'ioc_detected',      color: 'var(--amber)',   val: stats.ioc_detected || 0 },
      { k: 'unicode_sanitized', color: 'var(--purple)',  val: stats.unicode_sanitized || 0 },
      { k: 'normal',            color: 'var(--green)',   val: stats.normal || 0 },
    ];
    el.innerHTML = items.map(it => {
      const pct = Math.round(it.val / total * 100);
      return `<div class="prog-row">
        <span class="prog-label" style="min-width:130px;font-size:10px">${esc(it.k)}</span>
        <div class="prog-track"><div class="prog-fill" style="width:${pct}%;background:${it.color}"></div></div>
        <span class="prog-val">${it.val}</span>
      </div>`;
    }).join('');

    if (adv.unique_fingerprints) {
      el.innerHTML += `<div style="margin-top:8px;font-family:var(--font-terminal);font-size:10px;color:var(--text-muted)">
        ${adv.unique_fingerprints} unique fingerprints · ${adv.high_risk_count || 0} high-risk
      </div>`;
    }
    return;
  }
  // Fall back to event-derived patterns
  const counts = {};
  for (const ev of S.events) {
    const d = ev.data || {};
    const vec = d.social_engineering_vector || d.vector || d.attack_type;
    if (vec && vec !== 'none_detected') counts[vec] = (counts[vec] || 0) + 1;
  }
  if (!Object.keys(counts).length) {
    el.innerHTML = '<div style="color:var(--text-muted);font-family:var(--font-terminal);font-size:11px;padding:8px 0">No attack patterns detected.</div>';
    return;
  }
  const max = Math.max(...Object.values(counts), 1);
  el.innerHTML = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, v]) => {
    const pct = Math.round(v / max * 100);
    return `<div class="prog-row">
      <span class="prog-label" style="min-width:140px;font-size:10px">${esc(k)}</span>
      <div class="prog-track"><div class="prog-fill pf-magenta" style="width:${pct}%"></div></div>
      <span class="prog-val">${v}</span>
    </div>`;
  }).join('');
}

function _renderRedTeam(adv) {
  const el  = document.getElementById('sec-rt-body');
  const bdg = document.getElementById('sec-rt-badge');
  if (!el) return;

  if (adv?.ok) {
    const level = adv.threat_level || 'green';
    const levelColor = level === 'red' ? 'var(--red)' : level === 'orange' ? 'var(--amber)' : level === 'yellow' ? 'var(--amber)' : 'var(--green)';
    if (bdg) { bdg.textContent = adv.total_attacks ?? '—'; }
    el.innerHTML = `<div style="font-family:var(--font-terminal);font-size:11px;display:flex;flex-direction:column;gap:5px">
      <div><span style="color:var(--text-muted)">threat level: </span><span style="color:${levelColor};font-weight:600">${esc(level.toUpperCase())}</span></div>
      <div><span style="color:var(--text-muted)">total attacks: </span><span>${esc(adv.total_attacks ?? 0)}</span></div>
      <div><span style="color:var(--text-muted)">high risk: </span><span style="color:var(--red)">${esc(adv.high_risk_count ?? 0)}</span></div>
      <div><span style="color:var(--text-muted)">fingerprints: </span><span>${esc(adv.unique_fingerprints ?? 0)}</span></div>
    </div>`;
  } else {
    if (bdg) bdg.textContent = '—';
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px;font-family:var(--font-terminal)">No red team data. Run a simulation in Lab tab.</span>';
  }
}

function _renderInjectors(adv) {
  const feed = document.getElementById('sec-injectors-feed');
  if (!feed) return;

  const attackers = adv?.known_attackers;
  if (attackers?.length) {
    feed.innerHTML = attackers.slice(0, 20).map(a =>
      `<div class="term-event ev-warn">
        <span class="term-tag tag-block">INJECTOR</span>
        <span class="term-msg">${esc(a.author)} <span style="color:var(--text-muted)">(${a.count} attacks)</span></span>
      </div>`
    ).join('');
    return;
  }
  // Fall back to event-derived
  const injectors = new Set();
  for (const ev of S.events) {
    const d = ev.data || {};
    const src = d.source_id || d.user_id || d.sender || d.origin || d.author;
    if (src && /block|inject|suspicious/.test(String(ev.event || ev.type || '').toLowerCase())) {
      injectors.add(String(src));
    }
  }
  if (!injectors.size) {
    feed.innerHTML = '<div class="term-event ev-info"><span class="term-msg" style="color:var(--text-muted)">No known injectors.</span></div>';
    return;
  }
  feed.innerHTML = [...injectors].slice(0, 20).map(id =>
    `<div class="term-event ev-warn"><span class="term-tag tag-block">INJECTOR</span><span class="term-msg">${esc(id)}</span></div>`
  ).join('');
}
