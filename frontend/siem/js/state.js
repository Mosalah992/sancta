/* ─── Shared State Singleton ─────────────────────────────── */
export const S = {
  // Auth
  token: '',
  authRequired: false,

  // Agent
  cycle: 0,
  karma: 0,
  mood: '—',
  innerCircle: 0,
  recruited: 0,
  agentRunning: false,
  agentPid: null,
  agentSuspended: false,
  modelInfo: '—',

  // Security
  injections: 0,
  defenseRate: null,
  fpRate: null,
  recentThreats: 0,

  // Epidemic
  seir: 'SUSCEPTIBLE',
  driftScore: 0,
  alertLevel: 'clear',
  epidSignals: {},
  epidParams: {},
  epidSimData: null,

  // Epistemic
  epistemic: {
    coherence: 0.7,
    dissonance: 0.1,
    curiosity: 0.6,
    confidence: 0.75,
  },

  // Soul
  beliefs: {},
  journalEntries: [],

  // Events buffer (rolling 200)
  events: [],

  // Chat
  chatHistory: [],

  // Lab
  labResults: [],

  // Control
  activityLines: [],

  // Active tab
  activeTab: 'dashboard',

  // WS state
  wsConnected: false,

  // Cached adversary data from /api/security/adversary
  _adversaryData: null,
};

/** Push event into rolling buffer (max 80). Skip non-event objects (e.g. metrics). */
export function pushEvent(ev) {
  if (!ev || typeof ev !== 'object') return;
  if (ev.type === 'metrics' || ev.metrics !== undefined) return;
  S.events.unshift(ev);
  if (S.events.length > 80) S.events.pop();
}

function _str(v) {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'object') return '';  // unwrap nested: { event: "foo" } stays ""
  return String(v);
}

/** Classify event severity. Handles WS wrapper { type, event } and flat events. */
export function evSeverity(ev) {
  if (!ev || typeof ev !== 'object') return 'info';
  let e = ev.event ?? ev.type ?? '';
  if (typeof e === 'object' && e !== null) e = e.event ?? e.type ?? '';
  e = _str(e).toLowerCase();
  const src = _str(ev.source ?? '').toLowerCase();
  if (/block|inject|suspicious|compromise|critical/.test(e)) return 'block';
  if (/warn|attack|anomal|drift/.test(e)) return 'warn';
  if (/ok|pass|clean|susceptible|recovered/.test(e)) return 'ok';
  if (src === 'security' || src === 'redteam') return 'warn';
  if (src === 'philosophy') return 'soul';
  return 'info';
}
