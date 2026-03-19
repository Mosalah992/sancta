/* ─── Lab Tab ─────────────────────────────────────────────── */
import { S, pushEvent } from '../state.js';
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function init() {
  const labRtInput = document.getElementById('lab-rt-input');
  const labRtRun = document.getElementById('lab-rt-run');
  const labPhenoInput = document.getElementById('lab-pheno-input');
  const labPhenoRun = document.getElementById('lab-pheno-run');
  const labClear = document.getElementById('lab-clear');
  const labPresets = document.querySelectorAll('.lab-rt-preset');

  const addResult = (text, tag = 'info') => {
    S.labResults = S.labResults || [];
    S.labResults.unshift({ ts: new Date().toISOString(), text, tag });
    if (S.labResults.length > 50) S.labResults.pop();
    refresh();
  };

  labRtRun?.addEventListener('click', async () => {
    const payload = (labRtInput?.value || '').trim();
    if (!payload) return;
    labRtRun.disabled = true;
    addResult(`→ Sending injection payload: "${esc(payload.slice(0, 80))}"`, 'block');
    try {
      // Send the payload through the real chat endpoint — this exercises the full security pipeline
      const data = await api.sendChatMessage(payload, 'lab_redteam');
      if (data?.reply) {
        addResult(`← Agent replied (not blocked): ${esc(data.reply.slice(0, 120))}`, 'warn');
        addResult('⚠ Injection PASSED — security filter did not block this payload', 'warn');
      } else if (data?.error) {
        addResult(`✓ Blocked/error: ${esc(data.error)}`, 'ok');
      } else {
        addResult(`Result: ${esc(JSON.stringify(data).slice(0, 200))}`, 'info');
      }
      pushEvent({ source: 'redteam', event: 'lab_run', message: payload.slice(0, 60), ts: new Date().toISOString() });
    } catch (e) {
      addResult(`✓ Request blocked (exception): ${esc(String(e?.message || e))}`, 'ok');
    } finally {
      labRtRun.disabled = false;
    }
  });

  labPresets?.forEach(btn => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.p;
      if (p && labRtInput) labRtInput.value = p;
    });
  });

  labPhenoRun?.addEventListener('click', async () => {
    const msg = (labPhenoInput?.value || '').trim();
    if (!msg) return;
    labPhenoRun.disabled = true;
    addResult(`Phenomenology: sending "${esc(msg.slice(0, 60))}"`, 'soul');
    try {
      // Record pre-state snapshot
      const { S: _S } = await import('../state.js');
      const pre = { mood: _S.mood, cycle: _S.cycle, coherence: _S.epistemic?.coherence };
      const data = await api.sendChatMessage(msg, 'lab_pheno');
      // Check post-state (may differ after response if agent logs epistemic shift)
      addResult(`Pre: mood="${pre.mood}" coherence=${(pre.coherence ?? 0).toFixed(3)} cycle=${pre.cycle}`, 'info');
      if (data?.reply) {
        addResult(`Response: ${esc(data.reply.slice(0, 150))}`, 'soul');
      }
      addResult('Phenomenology record logged. Check Soul tab for state changes.', 'info');
      pushEvent({ source: 'philosophy', event: 'lab_pheno', message: msg.slice(0, 60), ts: new Date().toISOString() });
    } catch (e) {
      addResult(`Failed: ${esc(String(e?.message || e))}`, 'block');
    } finally {
      labPhenoRun.disabled = false;
    }
  });

  labClear?.addEventListener('click', () => {
    S.labResults = [];
    refresh();
  });
}

export function refresh() {
  const feed = document.getElementById('lab-results-feed');
  if (!feed) return;
  const results = S.labResults || [];
  feed.innerHTML = results.map(r => {
    const ts = (r.ts || '').toString().replace('T', ' ').slice(0, 19);
    const tagCls = r.tag === 'block' ? 'tag-block' : r.tag === 'soul' ? 'tag-soul' : r.tag === 'ok' ? 'tag-ok' : 'tag-info';
    return `<div class="term-event ev-${r.tag || 'info'}">
      <span class="term-ts">${esc(ts)}</span>
      <span class="term-tag ${tagCls}">LAB</span>
      <span class="term-msg">${esc(r.text)}</span>
    </div>`;
  }).join('') || '<div class="term-event ev-info"><span class="term-msg">No results yet</span></div>';
}
