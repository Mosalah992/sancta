/* ─── Chat Tab ─────────────────────────────────────────────── */
import { S } from '../state.js';
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const CHAT_SESSION_KEY = 'sancta_chat_session_id';

export function init() {
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const chatClear = document.getElementById('chat-clear');

  const scrollChat = () => {
    const wrap = document.getElementById('chat-messages-wrap');
    if (wrap) wrap.scrollTop = wrap.scrollHeight;
  };

  const send = async () => {
    const msg = (chatInput?.value || '').trim();
    if (!msg) return;
    if (chatSend) chatSend.disabled = true;
    if (chatInput) chatInput.value = '';

    const messagesEl = document.getElementById('chat-messages');
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.innerHTML = `<div class="chat-bubble">${esc(msg)}</div><div class="chat-meta">You</div>`;
    messagesEl?.appendChild(userDiv);

    const typing = document.createElement('div');
    typing.className = 'chat-msg agent chat-typing';
    typing.innerHTML = '<span></span><span></span><span></span>';
    messagesEl?.appendChild(typing);
    scrollChat();

    try {
      const sessionId = localStorage.getItem(CHAT_SESSION_KEY) || undefined;
      const data = await api.sendChatMessage(msg, sessionId);
      typing.remove();

      if (data?.session_id) localStorage.setItem(CHAT_SESSION_KEY, data.session_id);
      const reply = data?.reply ?? data?.error ?? 'No response';
      const interactionId = data?.interaction_id;

      const agentDiv = document.createElement('div');
      agentDiv.className = 'chat-msg agent';
      agentDiv.innerHTML = `<div class="chat-bubble">${esc(reply)}</div><div class="chat-meta">Sancta</div>`;
      if (messagesEl) messagesEl.appendChild(agentDiv);

      if (interactionId) {
        const fb = document.createElement('div');
        fb.className = 'chat-meta';
        fb.innerHTML = `
          <button class="btn btn-sm" data-fb="1" title="Good">+</button>
          <button class="btn btn-sm" data-fb="0" title="Neutral">?</button>
          <button class="btn btn-sm" data-fb="-1" title="Bad">−</button>
        `;
        fb.querySelectorAll('[data-fb]').forEach(btn => {
          btn.addEventListener('click', () => {
            api.sendChatFeedback(interactionId, parseInt(btn.dataset.fb, 10));
            btn.disabled = true;
          });
        });
        agentDiv.appendChild(fb);
      }
      scrollChat();
    } catch (e) {
      typing.remove();
      const errDiv = document.createElement('div');
      errDiv.className = 'chat-msg agent';
      errDiv.innerHTML = `<div class="chat-bubble">Error: ${esc(String(e?.message || e))}</div>`;
      messagesEl?.appendChild(errDiv);
      scrollChat();
    } finally {
      if (chatSend) chatSend.disabled = false;
      chatInput?.focus();
    }
  };

  chatSend?.addEventListener('click', send);
  chatInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  chatClear?.addEventListener('click', () => {
    const el = document.getElementById('chat-messages');
    if (el && (!el.children.length || confirm('Clear all chat messages?'))) el.innerHTML = '';
  });
}

export function refresh() {
  const messagesEl = document.getElementById('chat-messages');
  if (!messagesEl) return;
  const history = S.chatHistory || [];
  if (history.length && !messagesEl.querySelector('.chat-msg')) {
    history.forEach(m => {
      const div = document.createElement('div');
      div.className = `chat-msg ${m.role || 'agent'}`;
      div.innerHTML = `<div class="chat-bubble">${esc(m.text || m.content)}</div><div class="chat-meta">${m.role === 'user' ? 'You' : 'Sancta'}</div>`;
      messagesEl.appendChild(div);
    });
  }
}
