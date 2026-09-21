const state = {
  meta: null,
  evaluation: null,
  rows: null,
  website: null,
  dayIndex: 0,
  gameIndex: 0,
  gameSeries: null,
  detailOpen: null,
  graphMetric: 'Stakes',
  view: 'dashboard',
  sidebarCollapsed: false,
};

const $ = (id) => document.getElementById(id);
const fmtInt = (x) => Number.isFinite(Number(x)) ? Math.round(Number(x)).toLocaleString() : '—';
const fmtPct = (x) => Number.isFinite(Number(x)) ? `${(Number(x) * 100).toFixed(1)}%` : '—';
const fmtScore = (x) => Number.isFinite(Number(x)) ? Number(x).toFixed(1) : '—';
const fmtSignal = (x) => Number.isFinite(Number(x)) ? Number(x).toFixed(2) : '—';
const fmtMoney = (x) => Number.isFinite(Number(x)) ? `R ${Math.round(Number(x)).toLocaleString()}` : '—';
const hasNumber = (x) => Number.isFinite(Number(x));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));

const explanations = {
  performance: { title:'Current performance', body:"The current-performance signal is calculated from the latest available 7-day history before the selected evaluation day. It is a ranking signal, not the selected day raw result.", formula:'25% Clients + 20% Bet Count + 30% Stakes + 25% GGR (missing metrics are excluded and weights are re-normalised)' },
  momentum: { title:'Momentum', body:'Momentum measures whether the game is improving or weakening. The engine compares recent periods with their preceding periods rather than relying on one day.', formula:'Multi-window directional change across Clients, Bet Count, Stakes and GGR' },
  acceleration: { title:'Acceleration', body:'Acceleration asks whether momentum itself is strengthening or cooling. A positive value means the direction is becoming more positive.', formula:'Shorter-window momentum compared with longer-window momentum' },
  stability: { title:'Stability', body:'Stability measures how consistent recent performance is. Consistent activity gets more credibility than a highly erratic series.', formula:'Higher consistency → higher stability score' },
  persistence: { title:'Persistence', body:'Persistence is not a ranking score. It measures how many consecutive evaluation days the current direction has continued. Longer streaks allow stronger movement when the evidence remains consistent.', formula:'Consecutive days supporting the same direction; movement uses a 1–5 day maturity scale' },
};

function currentDate() { return state.meta.dates[state.dayIndex]; }

function spark(values, positive=true) {
  const a = values.map(Number).filter(Number.isFinite);
  if (a.length < 2) return '<svg class="spark" viewBox="0 0 46 18"><line x1="2" y1="9" x2="44" y2="9" stroke="#5f6880"/></svg>';
  let mn = Math.min(...a), mx = Math.max(...a); if (mx === mn) mx = mn + 1;
  const pts = a.map((x,i) => `${(i/(a.length-1))*44+1},${17-((x-mn)/(mx-mn))*14}`).join(' ');
  const c = positive ? '#2bd576' : '#ff5c6c';
  return `<svg class="spark" viewBox="0 0 46 18"><polyline fill="none" stroke="${c}" stroke-width="2" points="${pts}"/></svg>`;
}

function fullSpark(values, positive=true) {
  const a = values.map(Number).filter(Number.isFinite);
  if (a.length < 2) return '<svg viewBox="0 0 304 100"><line x1="2" y1="50" x2="302" y2="50" stroke="#5f6880"/></svg>';
  let mn = Math.min(...a), mx = Math.max(...a); if (mx === mn) mx = mn + 1;
  const pts = a.map((x,i) => `${(i/(a.length-1))*300+2},${92-((x-mn)/(mx-mn))*84}`).join(' ');
  const c = positive ? '#2bd576' : '#ff5c6c';
  return `<svg viewBox="0 0 304 100" preserveAspectRatio="none"><polyline fill="none" stroke="${c}" stroke-width="3" points="${pts}"/></svg>`;
}

function signalTone(value, neutral=1) {
  const v = Number(value);
  if (!Number.isFinite(v) || Math.abs(v) < neutral) return 'neutral';
  return v > 0 ? 'positive' : 'negative';
}

function signalMiniSpark(values, currentValue, neutral=1) {
  const a = values.map(Number).filter(Number.isFinite);
  if (a.length < 2) return '<span class="signal-mini-empty">—</span>';
  let mn = Math.min(...a), mx = Math.max(...a); if (mx === mn) mx = mn + 1;
  const pts = a.map((x,i) => `${(i/(a.length-1))*56+1},${19-((x-mn)/(mx-mn))*16}`).join(' ');
  const tone = signalTone(currentValue, neutral);
  const c = tone === 'positive' ? '#2bd576' : tone === 'negative' ? '#ff5c6c' : '#9aa3b7';
  return `<svg class="signal-mini-svg" viewBox="0 0 58 20" aria-hidden="true"><polyline fill="none" stroke="${c}" stroke-width="1.8" points="${pts}"/></svg>`;
}

async function getJson(url) {
  const r = await fetch(url, { cache:'no-store' });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

async function loadDay(index) {
  state.dayIndex = Math.max(0, Math.min(state.meta.dates.length - 1, index));
  $('dateSelect').value = String(state.dayIndex);
  $('rankMeta').textContent = 'Loading selected day...';
  state.rows = await getJson(`/api/ranking?date=${encodeURIComponent(currentDate())}`);
  state.website = null;
  renderDashboard();
  if (state.view === 'website') await renderWebsite();
  if (state.gameIndex < state.meta.games.length) await selectGame(state.gameIndex, false);
}

function renderSummary() {
  const o = state.evaluation?.overall_evaluation || {};
  const c = state.meta.config;
  const cards = [
    ['Evaluation date', currentDate()],
    ['Games ranked', fmtInt(state.rows.rows.length)],
    ['Mean rank correlation', fmtScore(o.mean_algorithm_spearman)],
    ['Mean Top-10 overlap', fmtPct(o.mean_algorithm_top10_overlap)],
    ['Mean rank error', fmtInt(o.mean_algorithm_rank_error)],
    ['Forecast', 'Excluded in this test'],
  ];
  $('summaryGrid').innerHTML = cards.map(([label,value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
  $('subtitle').textContent = `Calibration: ${c.calibration_range[0]} → ${c.calibration_range[1]} · Evaluation: ${c.evaluation_range[0]} → ${c.evaluation_range[1]}`;
  $('rankMeta').textContent = `${currentDate()} · ${fmtInt(state.rows.rows.length)} games · website position is the controlled placement`;
}

function renderDashboardTable() {
  const q = $('rankSearch').value.trim().toLowerCase();
  const sf = $('rankFilter').value;
  const rows = state.rows.rows
    .filter(r => {
      const matchesText = !q || r.name.toLowerCase().includes(q) || r.system.toLowerCase().includes(q);
      return matchesText && (sf === 'ALL' || r.state === sf);
    })
    .sort((a,b) => Number(a.website_position) - Number(b.website_position));

  $('rankTable').innerHTML = rows.length ? rows.map(r => {
    const move = Number(r.movement) || 0;
    const mom = Number(r.momentum) || 0;
    const trend = mom >= 15 ? [-15,-6,3,10,16] : mom <= -15 ? [15,8,1,-8,-16] : [-3,-1,0,2,3];
    return `<tr data-gid="${r.game_idx}">
      <td class="rank">${r.website_position}</td>
      <td><div class="game-name"><span>${esc(r.name)}</span><small>${esc(r.system)}</small></div></td>
      <td class="${move>0?'movement-up':move<0?'movement-down':''}">${move>0?'↑ '+move:move<0?'↓ '+Math.abs(move):'—'}</td>
      <td>${spark(trend, mom >= 0)}</td>
      <td>${r.state === 'No Data' ? '—' : fmtScore(r.score)}</td>
      <td><span class="state ${stateClass(r.state)}">${esc(r.state)}</span></td>
    </tr>`;
  }).join('') : '<tr><td colspan="6" class="muted">No games match the selected filter.</td></tr>';

  document.querySelectorAll('#rankTable tr[data-gid]').forEach(tr => {
    tr.onclick = () => {
      navigate('dashboard');
      selectGame(Number(tr.dataset.gid));
    };
  });
}

function stateClass(value) {
  return String(value || '').toLowerCase().replace(/\s*&\s*/g,'-and-').replace(/\s+/g,'-');
}

function renderGameOptions() {
  $('gameList').innerHTML = state.meta.games.map(g => `<option value="${esc(g.name)}"></option>`).join('');
  $('gameSelect').innerHTML = state.meta.games.map((g,i) => `<option value="${i}">${esc(g.name)} · ${esc(g.system)}</option>`).join('');
  $('gameSelect').value = String(state.gameIndex);
}

function setSearchToGame() {
  const g = state.meta.games[state.gameIndex];
  if (!g) return;
  $('gameSearch').value = g.name;
  $('gameSelect').value = String(state.gameIndex);
}

async function selectGame(index, renderNow=true) {
  if (!Number.isInteger(index) || index < 0 || index >= state.meta.games.length) return;
  state.gameIndex = index;
  setSearchToGame();
  if (renderNow) $('gameDetail').innerHTML = '<div class="detail-empty">Loading game history...</div>';
  try {
    state.gameSeries = await getJson(`/api/game?id=${index}`);
    renderDetail();
  } catch (e) {
    $('gameDetail').innerHTML = `<div class="detail-empty">Unable to load this game: ${esc(e.message)}</div>`;
  }
}

function toggleSignal(key) {
  state.detailOpen = state.detailOpen === key ? null : key;
  renderDetail();
}

function renderDetail() {
  const g = state.meta.games[state.gameIndex];
  const r = state.rows?.rows.find(x => Number(x.game_idx) === Number(state.gameIndex));
  const s = state.gameSeries?.series || {};
  if (!g || !r) return;

  const dates = s.dates || [];
  const pos = dates.indexOf(currentDate());
  const start = Math.max(0, pos - 14);
  const metric = state.graphMetric || 'Stakes';
  const values = (s[metric] || []).slice(start, pos + 1);
  const signalHistory = s.signal_history || [];
  const currentSignalPoint = signalHistory.find(x => x.date === currentDate());
  const previousSignalPoint = pos > 0 ? signalHistory.find(x => x.date === dates[pos - 1]) : null;
  const ex = explanations[state.detailOpen];
  const hasRawData = r.data_status === 'Available';
  const signalAsOf = r.signal_as_of || (pos > 0 ? dates[pos - 1] : null);
  const movementReason = r.movement_reason || (Number(r.movement) === 0 ? 'Position held' : 'Moving toward ideal position');
  const signalUnavailable = r.state === 'No Data' && !hasNumber(r.current_performance);
  const signalDeltas = previousSignalPoint && !signalUnavailable ? {
    current_performance: Number(r.current_performance) - Number(previousSignalPoint.current_performance),
    momentum: Number(r.momentum) - Number(previousSignalPoint.momentum),
    acceleration: Number(r.acceleration_raw) - Number(previousSignalPoint.acceleration),
    stability: Number(r.stability) - Number(previousSignalPoint.stability),
    score: Number(r.score) - Number(previousSignalPoint.score),
  } : null;
  const scoreUnchanged = signalDeltas && Number.isFinite(signalDeltas.score) && Math.abs(signalDeltas.score) < 0.0001;
  const movementContext = !signalUnavailable && Number(r.movement) !== 0 && movementReason === 'Moving toward ideal position' && scoreUnchanged
    ? 'Score unchanged; position is continuing toward the existing ideal rank.'
    : movementReason;
  const deltaText = (key) => {
    if (!signalDeltas || !Number.isFinite(signalDeltas[key])) return '';
    const d = signalDeltas[key];
    if (Math.abs(d) < 0.005) return 'No change';
    return `${d > 0 ? '+' : ''}${d.toFixed(2)} vs prior day`;
  };
  const momentumSeries = signalHistory.map(x => x.momentum);
  const accelerationSeries = signalHistory.map(x => x.acceleration);
  const cards = [
    ['performance','Current performance',signalUnavailable ? 'No Data' : fmtSignal(r.current_performance), signalUnavailable ? 'muted-signal' : '', ''],
    ['momentum','Momentum',signalUnavailable ? 'No Data' : `${Number(r.momentum)>0?'+':''}${fmtSignal(r.momentum)}`, signalUnavailable ? 'muted-signal' : signalTone(r.momentum, 1), signalUnavailable ? '' : signalMiniSpark(momentumSeries, r.momentum, 1)],
    ['acceleration','Acceleration',signalUnavailable ? 'No Data' : `${Number(r.acceleration_raw)>0?'+':''}${fmtSignal(r.acceleration_raw)}`, signalUnavailable ? 'muted-signal' : signalTone(r.acceleration_raw, 1), signalUnavailable ? '' : signalMiniSpark(accelerationSeries, r.acceleration_raw, 1)],
    ['stability','Stability',signalUnavailable ? 'No Data' : fmtSignal(r.stability), signalUnavailable ? 'muted-signal' : '', ''],
  ];

  $('gameDetail').innerHTML = `
    <div class="detail-title">${esc(g.name)}</div>
    <div class="detail-sub">${esc(g.system)}</div>
    <div class="detail-status-row">
      <span class="data-status ${hasRawData ? 'available' : 'missing'}">${hasRawData ? `Raw data available for ${esc(currentDate())}` : `No Data for ${esc(currentDate())}`}</span>
      <span class="signal-asof">Ranking signals as of ${esc(signalAsOf || 'N/A')}</span>
    </div>
    <div class="detail-grid rank-mini-grid">
      <div class="mini-card"><div class="k">Website position</div><div class="v">#${r.website_position}</div></div>
      <div class="mini-card"><div class="k">Ranking score</div><div class="v">${signalUnavailable ? '—' : fmtScore(r.score)}</div></div>
      <div class="mini-card"><div class="k">Game state</div><div class="v"><span class="state ${stateClass(r.state)}">${esc(r.state)}</span></div></div>
      <div class="mini-card"><div class="k">Daily movement</div><div class="v">${Number(r.movement)>0?'↑ '+r.movement:Number(r.movement)<0?'↓ '+Math.abs(r.movement):'—'}</div></div>
    </div>
    <div class="movement-reason"><span class="k">Movement reason</span><span>${esc(movementContext)}</span></div>
    <div class="signal-delta-row evidence-row">
      <span>Direction persistence: <b>${Number(r.persistence_days) || 0} day${Number(r.persistence_days) === 1 ? '' : 's'}</b></span>
      <span>Strong-performance streak: <b>${Number(r.strong_performance_days) || 0} day${Number(r.strong_performance_days) === 1 ? '' : 's'}</b></span>
      <span>Movement confidence: <b>${hasNumber(r.movement_confidence) ? fmtPct(r.movement_confidence) : '—'}</b></span>
    </div>
    <div class="signal-delta-row">${!signalUnavailable && previousSignalPoint ? `
      <span>vs prior day: <b>${esc(deltaText('current_performance'))}</b></span>
      <span><b>${esc(deltaText('momentum'))}</b></span>
      <span><b>${esc(deltaText('acceleration'))}</b></span>
      <span><b>${esc(deltaText('stability'))}</b></span>
    ` : (signalUnavailable ? '<span>Signals unavailable because there is no selected-day data.</span>' : '<span>No prior evaluation day available.</span>')}</div>
    <div class="signal-grid">
      ${cards.map(([k,l,v,c,graph]) => `<button class="signal-card ${state.detailOpen===k?'active':''}" data-signal="${k}"><span><span class="k">${l}</span><span class="signal-value ${c}">${v}</span></span>${graph ? `<span class="signal-mini">${graph}</span>` : ''}<span class="signal-chevron">${state.detailOpen===k?'−':'+'}</span></button>`).join('')}
    </div>
    ${ex ? `<div class="explain-box"><div class="explain-title">${ex.title}</div><div class="explain-body">${ex.body}</div><div class="formula">${ex.formula}</div></div>` : ''}
    <div class="metric-head"><div class="k">${metric === 'Betcount' ? 'Bet count' : metric} · recent history</div><select id="graphMetric"><option value="Stakes">Stakes</option><option value="Clients">Clients</option><option value="Betcount">Bet count</option><option value="GGR">GGR</option></select></div>
    <div class="trend-card">${values.filter(hasNumber).length >= 2 ? fullSpark(values, Number(r.momentum) >= 0) : '<div class="chart-empty">No historical data available</div>'}</div>
    <div class="raw-title">Raw performance for ${esc(currentDate())}</div>
    <div class="raw-grid">
      <div class="raw-card ${hasRawData && hasNumber(r.Clients) ? '' : 'no-data'}"><span>Clients</span><strong>${hasRawData && hasNumber(r.Clients) ? fmtInt(r.Clients) : 'No Data'}</strong></div>
      <div class="raw-card ${hasRawData && hasNumber(r.Betcount) ? '' : 'no-data'}"><span>Bet count</span><strong>${hasRawData && hasNumber(r.Betcount) ? fmtInt(r.Betcount) : 'No Data'}</strong></div>
      <div class="raw-card ${hasRawData && hasNumber(r.Stakes) ? '' : 'no-data'}"><span>Stakes</span><strong>${hasRawData && hasNumber(r.Stakes) ? fmtMoney(r.Stakes) : 'No Data'}</strong></div>
      <div class="raw-card ${hasRawData && hasNumber(r.GGR) ? '' : 'no-data'}"><span>GGR</span><strong>${hasRawData && hasNumber(r.GGR) ? fmtMoney(r.GGR) : 'No Data'}</strong></div>
    </div>`;

  $('gameDetail').querySelectorAll('[data-signal]').forEach(b => b.onclick = () => toggleSignal(b.dataset.signal));
  $('graphMetric').value = state.graphMetric;
  $('graphMetric').onchange = (e) => { state.graphMetric = e.target.value; renderDetail(); };
}

function renderEvaluation() {
  const o = state.evaluation?.overall_evaluation || {};
  const methods = [
    ['Algorithm','algorithm'],['Stakes baseline','stakes'],['GGR baseline','ggr'],['Clients baseline','clients'],['Bet Count baseline','bet_count']
  ];
  $('baselineTable').innerHTML = methods.map(([label,k]) => `<tr><td>${label}</td><td>${fmtScore(o[`mean_${k}_spearman`])}</td><td>${fmtPct(o[`mean_${k}_top10_overlap`])}</td><td>${fmtScore(o[`mean_${k}_top10_actual_score`])}</td></tr>`).join('');
  const ab = state.evaluation?.ablation_summary || {};
  const abLabels = { stakes:'Stakes only', current:'Current performance only', current_momentum:'Current + Momentum', current_momentum_acceleration:'Current + Momentum + Acceleration', full:'Full calibrated score' };
  if ($('ablationTable')) {
    $('ablationTable').innerHTML = Object.entries(abLabels).map(([k,label]) => {
      const x = ab[k] || {};
      return `<tr><td>${label}</td><td>${fmtScore(x.mean_spearman)}</td><td>${fmtPct(x.mean_top10)}</td><td>${fmtPct(x.mean_top20)}</td><td>${fmtPct(x.mean_top100)}</td><td>${fmtInt(x.mean_rank_error)}</td></tr>`;
    }).join('');
  }
  $('evalCards').innerHTML = [
    ['Algorithm correlation', fmtScore(o.mean_algorithm_spearman)],
    ['Algorithm Top-10 overlap', fmtPct(o.mean_algorithm_top10_overlap)],
    ['Algorithm Top-20 overlap', fmtPct(o.mean_algorithm_top20_overlap)],
    ['Algorithm Top-100 overlap', fmtPct(o.mean_algorithm_top100_overlap)],
    ['Algorithm rank error', fmtInt(o.mean_algorithm_rank_error)],
    ['Calibrated weights', state.evaluation?.calibration_weights ? Object.values(state.evaluation.calibration_weights).map(x => `${Math.round(Number(x)*100)}%`).join(' / ') : '—']
  ].map(([l,v]) => `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');
}

async function renderWebsite() {
  $('websiteGrid').innerHTML = '<div class="website-loading">Loading website placement...</div>';
  try {
    state.website = await getJson(`/api/website?date=${encodeURIComponent(currentDate())}&limit=300`);
    $('websiteMeta').textContent = `${currentDate()} · showing positions 1–${Math.min(300, state.website.total_games)} of ${fmtInt(state.website.total_games)} games`;
    $('websiteGrid').innerHTML = state.website.cards.map(card => {
      const movement = Number(card.movement) || 0;
      const movementHtml = movement > 0 ? `<span class="card-move up">↑ ${movement}</span>` : movement < 0 ? `<span class="card-move down">↓ ${Math.abs(movement)}</span>` : '';
      const image = card.image ? `<img src="${esc(card.image)}" alt="${esc(card.name)}" loading="lazy">` : `<div class="art-placeholder"><span>${esc((card.name || 'Game').slice(0,1).toUpperCase())}</span></div>`;
      return `<button class="game-tile" data-gid="${card.game_idx}" title="Open ${esc(card.name)} in Dashboard">
        <div class="tile-art">${image}<span class="tile-position">#${card.website_position}</span>${movementHtml}</div>
        <div class="tile-info"><span class="tile-name">${esc(card.name)}</span><span class="tile-provider">${esc(card.system)}</span></div>
      </button>`;
    }).join('');
    document.querySelectorAll('.game-tile').forEach(tile => tile.onclick = () => {
      navigate('dashboard');
      selectGame(Number(tile.dataset.gid));
    });
  } catch (e) {
    $('websiteGrid').innerHTML = `<div class="website-loading">Unable to load website placement: ${esc(e.message)}</div>`;
  }
}

function renderDashboard() {
  renderSummary();
  renderDashboardTable();
  renderDetail();
  $('prevBtn').disabled = state.dayIndex === 0;
  $('nextBtn').disabled = state.dayIndex === state.meta.dates.length - 1;
}

function navigate(view) {
  state.view = view;
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.toggle('active', btn.dataset.view === view));
  $('dashboardView').classList.toggle('hidden', view !== 'dashboard');
  $('websiteView').classList.toggle('hidden', view !== 'website');
  $('pageTitle').textContent = view === 'dashboard' ? 'Dynamic Game Placement Simulation' : 'Website Placement Simulation';
  if (view === 'website') renderWebsite();
}

function applySidebar() {
  document.body.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  try { localStorage.setItem('category_x_sidebar_collapsed', state.sidebarCollapsed ? '1' : '0'); } catch (_) {}
}

async function pollStatus() {
  try {
    const s = await getJson('/api/status');
    const box = $('pipelineStatus');
    box.className = `pipeline-status ${s.status || 'unknown'}`;
    box.querySelector('.status-dot').className = `status-dot ${s.status || 'unknown'}`;
    box.querySelector('.status-text').textContent = s.message || s.status || 'Unknown';
  } catch (_) {}
}

async function init() {
  state.meta = await getJson('/api/bootstrap');
  state.evaluation = await getJson('/api/evaluation');
  $('dateSelect').innerHTML = state.meta.dates.map((d,i) => `<option value="${i}">${d}</option>`).join('');
  renderGameOptions();
  renderEvaluation();
  document.querySelectorAll('.nav-item').forEach(btn => btn.onclick = () => navigate(btn.dataset.view));
  $('sidebarToggle').onclick = () => { state.sidebarCollapsed = !state.sidebarCollapsed; applySidebar(); };
  $('prevBtn').onclick = () => loadDay(state.dayIndex - 1);
  $('nextBtn').onclick = () => loadDay(state.dayIndex + 1);
  $('dateSelect').onchange = e => loadDay(Number(e.target.value));
  $('rankSearch').oninput = renderDashboardTable;
  $('rankFilter').onchange = renderDashboardTable;
  $('gameSelect').onchange = e => selectGame(Number(e.target.value));
  $('gameSearch').onchange = () => {
    const target = state.meta.games.findIndex(g => g.name.toLowerCase() === $('gameSearch').value.trim().toLowerCase());
    if (target >= 0) selectGame(target);
  };
  $('gameSearch').addEventListener('keydown', e => { if (e.key === 'Enter') $('gameSearch').dispatchEvent(new Event('change')); });
  try { state.sidebarCollapsed = localStorage.getItem('category_x_sidebar_collapsed') === '1'; } catch (_) {}
  applySidebar();
  state.dayIndex = Math.max(0, state.meta.dates.length - 1);
  $('dateSelect').value = String(state.dayIndex);
  await loadDay(state.dayIndex);
  await pollStatus();
  setInterval(pollStatus, 5000);
}

init().catch(e => {
  document.body.innerHTML = `<div class="fatal">Unable to load simulation: ${esc(e.message)}</div>`;
});
