/* ─── Control Tab ─────────────────────────────────────────── */
import { S } from '../state.js';
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function init() {
  const ctrlModeSelect = document.getElementById('ctrl-mode-select');
  const ctrlStart = document.getElementById('ctrl-start');
  const ctrlPause = document.getElementById('ctrl-pause');
  const ctrlResume = document.getElementById('ctrl-resume');
  const ctrlStop = document.getElementById('ctrl-stop');
  const ctrlRestart = document.getElementById('ctrl-restart');

  const runAction = async (fn) => {
    try {
      const data = await fn();
      if (data?.ok === false) {
        const msg = data?.error ?? data?.detail ?? 'Action failed';
        if (typeof window.showToast === 'function') window.showToast(msg, 'error');
        else console.warn('[Control]', msg);
      }
      refresh();
    } catch (e) {
      const msg = e?.message || String(e);
      if (typeof window.showToast === 'function') window.showToast(msg, 'error');
      else console.error('[Control]', e);
    }
  };

  const getMode = () => (ctrlModeSelect?.value || 'passive').trim() || 'passive';
  ctrlStart?.addEventListener('click', () => runAction(() => api.startAgent(getMode())));
  ctrlPause?.addEventListener('click', () => runAction(api.pauseAgent));
  ctrlResume?.addEventListener('click', () => runAction(api.resumeAgent));
  ctrlStop?.addEventListener('click', () => {
    if (S.agentRunning && !confirm('Stop the Sancta agent? This will terminate the process.')) return;
    runAction(api.killAgent);
  });
  ctrlRestart?.addEventListener('click', () => {
    if (S.agentRunning && !confirm('Restart the agent? This will stop and start it again.')) return;
    runAction(() => api.restartAgent(getMode()));
  });
}

export function refresh() {
  const piStatus = document.getElementById('pi-status');
  const piPid = document.getElementById('pi-pid');
  const piCycle = document.getElementById('pi-cycle');
  const piKarma = document.getElementById('pi-karma');
  const piMood = document.getElementById('pi-mood');
  const piModel = document.getElementById('pi-model');
  const ctrlActivityFeed = document.getElementById('ctrl-activity-feed');
  const ctrlLogCount = document.getElementById('ctrl-log-count');
  const ctrlKnowledge = document.getElementById('ctrl-knowledge');
  const ctrlCommunity = document.getElementById('ctrl-community');
  const ctrlStart = document.getElementById('ctrl-start');
  const ctrlPause = document.getElementById('ctrl-pause');
  const ctrlResume = document.getElementById('ctrl-resume');
  const ctrlStop = document.getElementById('ctrl-stop');
  const ctrlRestart = document.getElementById('ctrl-restart');

  const running = S.agentRunning && !S.agentSuspended;
  const suspended = S.agentRunning && S.agentSuspended;

  if (ctrlStart) ctrlStart.disabled = !!running;
  if (ctrlPause) ctrlPause.disabled = !running || !!suspended;
  if (ctrlResume) ctrlResume.disabled = !suspended;
  if (ctrlStop) ctrlStop.disabled = !running;
  if (ctrlRestart) ctrlRestart.disabled = !running;

  if (piStatus) {
    piStatus.textContent = S.agentRunning ? (S.agentSuspended ? 'PAUSED' : 'ONLINE') : 'OFFLINE';
    piStatus.className = 'proc-val ' + (S.agentRunning && !S.agentSuspended ? 'pv-online' : 'pv-offline');
  }
  if (piPid) piPid.textContent = S.agentPid ?? '—';
  if (piCycle) piCycle.textContent = S.cycle ?? '—';
  if (piKarma) piKarma.textContent = S.karma ?? '—';
  if (piMood) piMood.textContent = S.mood ?? '—';
  if (piModel) piModel.textContent = S.modelInfo ?? '—';

  const lines = S.activityLines || [];
  if (ctrlActivityFeed) {
    ctrlActivityFeed.innerHTML = lines.slice(0, 80).map(line => {
      const escaped = esc(line);
      return `<div class="term-event ev-info"><span class="term-msg">${escaped}</span></div>`;
    }).join('') || '<div class="term-event ev-info"><span class="term-msg">No activity yet</span></div>';
  }
  if (ctrlLogCount) ctrlLogCount.textContent = String(lines.length);

  if (ctrlKnowledge) {
    ctrlKnowledge.innerHTML = `
      <div class="ks-item"><span class="ks-key">entries</span><span class="ks-val">—</span></div>
      <div class="ks-item"><span class="ks-key">patterns</span><span class="ks-val">—</span></div>
    `;
  }
  if (ctrlCommunity) {
    ctrlCommunity.innerHTML = `
      <div class="ks-item"><span class="ks-key">inner circle</span><span class="ks-val">${S.innerCircle ?? '—'}</span></div>
      <div class="ks-item"><span class="ks-key">recruited</span><span class="ks-val">${S.recruited ?? '—'}</span></div>
    `;
  }
}
