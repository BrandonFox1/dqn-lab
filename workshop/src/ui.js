(() => {
  'use strict';
  const E = Engine;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const MINUS = '\u2212';
  const sgn = (x, d = 2) => (x < 0 ? MINUS : '+') + Math.abs(x).toFixed(d);
  const num = (x, d = 2) => (x < -0.5 * Math.pow(10, -d) ? MINUS : '') + Math.abs(x).toFixed(d);
  const pts = (x) => (x < 0 ? MINUS : '+') + Math.abs(Math.round(x));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const MOVE = ['up', 'down', 'left', 'right'];
  const GLYPH = ['↑', '↓', '←', '→'];
  const PAC_ROT = [270, 90, 180, 0];
  const ARR_ROT = [0, 180, 270, 90];
  const TRAINED_B64 = '__BRAIN_B64__';
  const TRAINED_META = __BRAIN_META__;
  const HIDDEN = [64, 64];
  const SCALE = 0.01;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(node, s) { node.innerHTML = s; return node; }
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  function heat(t) {
    const a = Math.round(clamp(Math.abs(t), 0, 1) * 80);
    if (a < 2) return '';
    return `color-mix(in srgb, var(${t >= 0 ? '--heat-pos' : '--heat-neg'}) ${a}%, transparent)`;
  }
  function tieArgmax(q, rng) {
    let best = -Infinity, ties = [];
    for (let i = 0; i < q.length; i++) { if (q[i] > best + 1e-9) { best = q[i]; ties = [i]; } else if (Math.abs(q[i] - best) <= 1e-9) ties.push(i); }
    return ties.length === 1 ? ties[0] : ties[rng.int(ties.length)];
  }
  function newTrainedBrain() { const m = new E.MLP([116, ...HIDDEN, 4], 1); m.loadBase64(TRAINED_B64); return m; }

  // ---------------------------------------------------------- controls
  function bindSeg(root, onChange) {
    root.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-v]');
      if (!b || b.disabled || !root.contains(b)) return;
      setSeg(root, b.dataset.v); onChange(b.dataset.v);
    });
  }
  function setSeg(root, v) { $$('button[data-v]', root).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === String(v)))); }

  function makeKnob(parent, spec) {
    const wrap = el('label', 'knob', parent);
    el('span', 'k-name', wrap, spec.label);
    const out = el('output', '', wrap);
    const input = el('input', '', wrap);
    input.type = 'range'; input.min = 0; input.max = 1000; input.step = 1;
    if (spec.hint) el('span', 'k-hint', wrap, spec.hint);
    const R = 1000;
    const toPos = (v) => spec.log ? Math.round(R * Math.log(v / spec.min) / Math.log(spec.max / spec.min)) : Math.round(R * (v - spec.min) / (spec.max - spec.min));
    const fromPos = (p) => {
      let v = spec.log ? spec.min * Math.pow(spec.max / spec.min, p / R) : spec.min + (spec.max - spec.min) * (p / R);
      if (spec.step) v = Math.round(v / spec.step) * spec.step;
      if (spec.sig) v = +v.toPrecision(spec.sig);
      return clamp(v, spec.min, spec.max);
    };
    let value = spec.value;
    const show = () => { out.textContent = spec.fmt ? spec.fmt(value) : value.toFixed(spec.digits ?? 2); };
    input.value = toPos(value); show();
    input.addEventListener('input', () => { value = fromPos(+input.value); show(); spec.onInput && spec.onInput(value); });
    input.addEventListener('change', () => { spec.onChange && spec.onChange(value); });
    return {
      get: () => value,
      set(v, silent) { value = v; input.value = toPos(v); show(); if (!silent) { spec.onInput && spec.onInput(v); } },
      wrap, input,
    };
  }

  function fitCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 300, h = cv.clientHeight || 150;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  function haloText(ctx, text, x, y) {
    ctx.save(); ctx.lineWidth = 3.5; ctx.lineJoin = 'round'; ctx.strokeStyle = cssVar('--sheet-2'); ctx.setLineDash([]);
    ctx.strokeText(text, x, y); ctx.restore(); ctx.fillText(text, x, y);
  }
  function niceTicks(lo, hi, n = 4) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  // ------------------------------------------------------------ Q bars
  class QBars {
    constructor(root) {
      this.root = root; root.textContent = ''; this.cols = [];
      for (let a = 0; a < 4; a++) {
        const qb = el('div', 'qb', root), tr = el('div', 'qb-track', qb);
        const zero = el('div', 'qb-zero', tr), bar = el('div', 'qb-bar', tr), val = el('div', 'qb-val num', tr);
        const name = el('div', 'qb-name', qb);
        this.cols.push({ zero, bar, val, name });
      }
    }
    render(q, { hi = -1, star = -1, cls = 'guess', range = null } = {}) {
      let lo = Math.min(0, ...q), top = Math.max(0, ...q);
      if (range) { lo = Math.min(lo, range[0]); top = Math.max(top, range[1]); }
      const span = (top - lo) || 1, pad = span * 0.26;
      const T = top + pad, B = lo < 0 ? lo - pad : 0;
      const y = (v) => ((T - v) / (T - B)) * 100;
      for (let a = 0; a < 4; a++) {
        const c = this.cols[a], v = q[a];
        const y1 = y(Math.max(v, 0)), y2 = y(Math.min(v, 0));
        c.bar.style.top = y1 + '%'; c.bar.style.height = `max(2px, ${y2 - y1}%)`;
        c.bar.className = 'qb-bar' + (a === hi ? ' ' + cls : '');
        c.zero.style.top = y(0) + '%';
        c.val.textContent = num(v);
        c.val.style.top = v >= 0 ? `calc(${y1}% - 17px)` : `calc(${y2}% + 2px)`;
        c.name.className = 'qb-name' + (a === hi ? ' hi' : '');
        c.name.innerHTML = `<span class="g">${GLYPH[a]}</span><span class="w"> ${MOVE[a]}</span>${a === star ? ' <span class="star">★</span>' : ''}`;
      }
    }
  }

  // --------------------------------------------------------- maze view
  class MazeView {
    constructor(root, env, opts = {}) {
      this.root = root; this.env = env; this.opts = opts;
      root.style.setProperty('--cols', env.W); root.style.setProperty('--rows', env.H);
      root.textContent = '';
      this.floor = new Array(env.N);
      for (let g = 0; g < env.W * env.H; g++) {
        const c = env.cellOf[g];
        if (c < 0) {
          const d = el('div', 'cell wall', root);
          const x = g % env.W, yy = Math.floor(g / env.W);
          if (x === 0 || yy === 0 || x === env.W - 1 || yy === env.H - 1) d.classList.add('edge');
          continue;
        }
        const d = el(opts.onTap ? 'button' : 'div', 'cell floor', root);
        if (opts.onTap) {
          d.type = 'button';
          const [x, yy] = env.xy(c);
          d.setAttribute('aria-label', `Square in column ${x}, row ${yy}`);
          d.addEventListener('click', () => opts.onTap(c));
        }
        const tint = el('div', 'tint', d), pel = el('div', 'pellet', d);
        const val = opts.mini ? null : el('div', 'cval', d);
        const arr = opts.mini ? null : el('div', 'carr', d);
        if (arr) arr.hidden = true;
        this.floor[c] = { d, tint, pel, val, arr };
      }
      this.ring = el('div', 'pick-ring'); this.ring.hidden = true;
      this.trail = el('div', 'actor ghost trail', root);
      this.ghost = el('div', 'actor ghost', root);
      this.pac = el('div', 'actor pac', root);
    }
    place(actor, c) { const [x, y] = this.env.xy(c); actor.style.transform = `translate(${x * 100}%, ${y * 100}%)`; }
    instant(on) { this.root.classList.toggle('instant', !!on); }
    setRing(c) { if (c == null || c < 0) { this.ring.hidden = true; return; } this.ring.hidden = false; this.floor[c].d.appendChild(this.ring); }
    render(s, overlay = null) {
      const N = this.env.N;
      for (let c = 0; c < N; c++) {
        const f = this.floor[c];
        f.pel.hidden = !s.pellets[c] || c === s.player;
        f.pel.style.opacity = overlay ? '.35' : '';
        if (!overlay || overlay.norm[c] == null) {
          f.tint.style.background = '';
          if (f.val) f.val.textContent = '';
          if (f.arr) f.arr.hidden = true;
          continue;
        }
        f.tint.style.background = heat(overlay.norm[c]);
        if (f.val) f.val.textContent = overlay.labels ? overlay.labels[c] : '';
        if (f.arr) {
          const a = overlay.arrows ? overlay.arrows[c] : -1;
          f.arr.hidden = !(a >= 0);
          if (a >= 0) f.arr.style.setProperty('--rot', ARR_ROT[a] + 'deg');
        }
      }
      this.place(this.pac, s.player);
      this.pac.style.setProperty('--rot', PAC_ROT[s.lastAction ?? 3] + 'deg');
      this.pac.classList.toggle('dead', !!s.dead);
      this.place(this.ghost, s.ghost);
      const gp = s.ghostPrev ?? s.ghost;
      this.trail.hidden = gp === s.ghost;
      if (gp !== s.ghost) this.place(this.trail, gp);
    }
  }

  // =================================================================== ARCADE
  const arcade = (() => {
    const DEF = { lr: 0.001, gamma: 0.97, epsDecay: 15000, targetSync: 500, batch: 32, bufferSize: 20000, useReplay: true, useTarget: true, double: true, epsStart: 1, epsEnd: 0.05 };
    const cfg = Object.assign({}, DEF);
    const env = new E.MiniPac({ seed: 11 });
    const testEnv = new E.MiniPac({ seed: 999 });
    let agent, obs, testObs;
    const EXAM_EVERY = 2500, EXAM_GAMES = 10;
    const examEnv = new E.MiniPac({ seed: 31337 });
    const st = { running: false, mode: 'watch', testing: false, scores: [], scoreSteps: [], wins: [], events: [], exams: [], nextExam: 0, lastStep: 0, lastBoard: 0, lastCurve: 0, lastStats: 0, baseline: 0, testLast: null, testGames: 0, loadedTrained: false };
    let view, qbars, knobs = {};
    const panel = () => $('#panel-arcade');

    function randomBaseline() {
      const e = new E.MiniPac({ seed: 4242 }), rng = E.makeRng(77); let s = 0; const n = 300;
      for (let i = 0; i < n; i++) { e.reset(); while (!e.done) e.step(rng.int(4)); s += e.score; }
      return s / n;
    }
    function newAgent(extra = {}) {
      agent = new E.DQN(116, 4, Object.assign({ hidden: HIDDEN, seed: (Math.random() * 1e9) | 0 }, cfg, extra));
      obs = env.reset((Math.random() * 1e9) | 0);
      st.scores = []; st.scoreSteps = []; st.wins = []; st.events = []; st.exams = []; st.loadedTrained = false;
      runExam();
    }
    // Exam: 10 fixed games of best play (no random moves, no learning). This is what the agent has really learned.
    function runExam(brk = false) {
      let sc = 0, wins = 0;
      for (let i = 0; i < EXAM_GAMES; i++) {
        let o = examEnv.reset(1000 + i);
        while (!examEnv.done) o = examEnv.step(E.argmax(agent.live.forward(o))).obs;
        sc += examEnv.score; wins += examEnv.won ? 1 : 0;
      }
      st.exams.push({ step: agent.steps, score: sc / EXAM_GAMES, wins, brk });
      st.nextExam = agent.steps - (agent.steps % EXAM_EVERY) + EXAM_EVERY;
    }
    function addEvent(label) { st.events.push({ step: agent.steps, label }); drawCurve(); }

    function trainStep() {
      const a = agent.act(obs);
      const r = env.step(a);
      agent.observe(obs, a, r.reward, r.obs, r.terminal);
      obs = r.obs;
      if (env.done) { st.scores.push(env.score); st.scoreSteps.push(agent.steps); st.wins.push(env.won ? 1 : 0); obs = env.reset(); }
      if (agent.steps >= st.nextExam) runExam();
    }
    function testStep() {
      if (!testObs || testEnv.done) { testObs = testEnv.reset(); st.testGames++; }
      const r = testEnv.step(agent.act(testObs, 0));
      testObs = r.obs;
      if (testEnv.done) st.testLast = { score: testEnv.score, won: testEnv.won, dead: testEnv.dead };
    }

    function renderBoard(e, o, instant) {
      view.instant(instant || reduceMotion);
      view.render(e);
      $('#arcade-q').classList.toggle('instant', !!instant || (st.running && st.mode !== 'watch'));
      const q = agent.q(o);
      qbars.render(q, { hi: E.argmax(q), cls: 'live' });
      const status = $('#arcade-status');
      if (st.testing) {
        const last = st.testLast ? ` Last test game: ${pts(st.testLast.score)}${st.testLast.won ? ', cleared the board' : st.testLast.dead ? ', caught' : ''}.` : '';
        status.innerHTML = `<span>Test drive, no learning, no random moves.${last}</span><span class="num">score ${pts(e.score)}</span>`;
      } else {
        status.innerHTML = `<span>Game ${st.scores.length + 1}, move ${e.steps}</span><span class="num">score ${pts(e.score)}</span>`;
      }
    }
    function avgLast(arr, n) { if (!arr.length) return NaN; const s = arr.slice(-n); return s.reduce((a, b) => a + b, 0) / s.length; }
    function renderStats() {
      const dl = $('#arc-stats');
      const avg = avgLast(st.scores, 20), ex = st.exams[st.exams.length - 1];
      const rows = [
        ['Games played', st.scores.length.toLocaleString()],
        ['Steps', agent.steps.toLocaleString()],
        ['Random-move chance (ε)', agent.epsilon().toFixed(2)],
        ['Replay memory', agent.cfg.useReplay ? `${agent.size.toLocaleString()} / ${agent.cap.toLocaleString()}` : 'off'],
        ['Weight updates', agent.updates.toLocaleString()],
        ['Frozen-copy syncs', agent.cfg.useTarget ? agent.syncs.toLocaleString() : 'off'],
        ['Practice avg, last 20', isNaN(avg) ? '—' : pts(avg)],
        ['Last loss', isNaN(agent.lastLoss) ? '—' : agent.lastLoss.toFixed(4)],
        ['Latest exam score', ex ? pts(ex.score) : '—'],
        ['Exam boards cleared', ex ? `${ex.wins} of ${EXAM_GAMES}` : '—'],
      ];
      if (!dl._built) { dl.textContent = ''; dl._dd = rows.map(([k]) => { const d = el('div', '', dl); el('dt', '', d, k); return el('dd', '', d); }); dl._built = true; }
      rows.forEach(([, v], i) => { if (dl._dd[i].textContent !== v) dl._dd[i].textContent = v; });
    }
    function drawCurve() {
      const cv = $('#arc-chart');
      if (!cv || panel().hidden) return;
      const { ctx, w, h } = fitCanvas(cv);
      const ink = cssVar('--ink'), ink2 = cssVar('--ink-3'), live = cssVar('--live'), frozen = cssVar('--frozen');
      const L = 42, Rm = 12, T = 12, B = 26;
      const sc = st.scores, ss = st.scoreSteps, ex = st.exams;
      ctx.font = '11px ' + cssVar('--font');
      if (!sc.length && ex.length < 2) {
        ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('Press Start training. Practice games show up as dots,', w / 2, h / 2 - 8);
        ctx.fillText('exam scores as a line.', w / 2, h / 2 + 8);
        return;
      }
      const x0 = ex.length ? ex[0].step : (ss[0] || 0), x1 = Math.max(agent.steps, x0 + 1000);
      let lo = st.baseline, hi = 0;
      for (const v of sc) { if (v < lo) lo = v; if (v > hi) hi = v; }
      for (const e of ex) { if (e.score < lo) lo = e.score; if (e.score > hi) hi = e.score; }
      const pad = (hi - lo) * 0.08 || 10; lo -= pad; hi += pad;
      const X = (step) => L + ((step - x0) / (x1 - x0)) * (w - L - Rm);
      const Y = (v) => T + ((hi - v) / (hi - lo)) * (h - T - B);
      ctx.lineWidth = 1; ctx.fillStyle = ink2; ctx.strokeStyle = ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const t of niceTicks(lo, hi, 4)) { ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - Rm, Y(t)); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(String(t).replace('-', MINUS), L - 6, Y(t)); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const t of niceTicks(x0, x1, 4)) { if (t < x0) continue; ctx.fillText(t >= 1000 ? `${+(t / 1000).toFixed(1)}k` : String(t), X(t), h - B + 6); }
      ctx.textAlign = 'right'; ctx.fillText('steps', w - Rm, h - B + 6 + 0);
      // random baseline
      ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(L, Y(st.baseline)); ctx.lineTo(w - Rm, Y(st.baseline)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; haloText(ctx, 'random play', L + 4, Y(st.baseline) - 3);
      // events
      ctx.textBaseline = 'top';
      st.events.forEach((ev, k) => {
        const x = X(clamp(ev.step, x0, x1));
        ctx.strokeStyle = frozen; ctx.globalAlpha = 0.8; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        const right = x > w * 0.62; ctx.fillStyle = frozen; ctx.textAlign = right ? 'right' : 'left';
        haloText(ctx, ev.label, x + (right ? -4 : 4), T + (k % 3) * 13);
      });
      // practice games (with random moves)
      ctx.fillStyle = ink; ctx.globalAlpha = sc.length > 400 ? 0.16 : 0.28;
      const stride = Math.max(1, Math.floor(sc.length / (w * 1.5)));
      for (let i = 0; i < sc.length; i += stride) { ctx.beginPath(); ctx.arc(X(ss[i]), Y(sc[i]), 1.6, 0, 6.3); ctx.fill(); }
      ctx.globalAlpha = 1;
      // exam line (best play)
      if (ex.length) {
        ctx.strokeStyle = live; ctx.fillStyle = live; ctx.lineWidth = 2.5; ctx.beginPath();
        ex.forEach((e, i) => (i && !e.brk ? ctx.lineTo(X(e.step), Y(e.score)) : ctx.moveTo(X(e.step), Y(e.score))));
        ctx.stroke();
        if (ex.length < 60) for (const e of ex) { ctx.beginPath(); ctx.arc(X(e.step), Y(e.score), 2.6, 0, 6.3); ctx.fill(); }
        const last = ex[ex.length - 1];
        ctx.font = '600 11px ' + cssVar('--font'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        haloText(ctx, 'exam', Math.min(X(last.step), w - Rm) - 4, Y(last.score) - 4);
      }
    }
    function setRunning(on) {
      st.running = on;
      if (on && st.testing) setTesting(false);
      $('#arc-train').textContent = on ? 'Pause training' : (agent.steps ? 'Resume training' : 'Start training');
    }
    function setTesting(on) {
      if (on && st.running) setRunning(false);
      st.testing = on; testObs = null; st.testLast = null;
      $('#arc-test').textContent = on ? 'Stop test drive' : 'Test drive';
      if (!on) renderBoard(env, obs, true);
    }

    function knobSpecs() {
      return [
        ['lr', { label: 'Learning rate', hint: 'How big each nudge is. Too big and it overshoots.', min: 0.0001, max: 0.02, log: true, value: cfg.lr, sig: 2, fmt: (v) => v.toPrecision(2), ev: (v) => `lr ${v.toPrecision(2)}` }],
        ['gamma', { label: 'Future discount (γ)', hint: 'How much a reward counts if it comes later. 0 = only the next step matters.', min: 0, max: 0.995, value: cfg.gamma, step: 0.005, fmt: (v) => v.toFixed(3), ev: (v) => `γ ${v.toFixed(2)}` }],
        ['epsDecay', { label: 'Exploration fades over', hint: 'Steps until random moves drop from 100% to 5%.', min: 1000, max: 60000, log: true, value: cfg.epsDecay, step: 500, fmt: (v) => `${Math.round(v).toLocaleString()} steps`, ev: (v) => `explore ${Math.round(v / 1000)}k` }],
        ['targetSync', { label: 'Sync frozen copy every', hint: 'How long the target stays still before it catches up with the live network.', min: 20, max: 5000, log: true, value: cfg.targetSync, step: 10, fmt: (v) => `${Math.round(v)} steps`, ev: (v) => `sync ${Math.round(v)}` }],
        ['batch', { label: 'Memories per update', hint: 'How many random memories are graded in each nudge.', min: 1, max: 64, value: cfg.batch, step: 1, fmt: (v) => String(Math.round(v)), ev: (v) => `batch ${Math.round(v)}` }],
        ['bufferSize', { label: 'Replay memory size', hint: 'Changing this empties the memory.', min: 500, max: 20000, log: true, value: cfg.bufferSize, step: 500, fmt: (v) => Math.round(v).toLocaleString(), ev: (v) => `memory ${Math.round(v / 1000)}k` }],
      ];
    }
    function applyCfg(key, v) {
      cfg[key] = v;
      if (key === 'bufferSize') agent.resizeBuffer(Math.round(v));
      else agent.cfg[key] = (key === 'batch' || key === 'targetSync') ? Math.round(v) : v;
    }
    function syncKnobUI() {
      for (const [k, spec] of knobSpecs()) knobs[k].set(cfg[k] ?? spec.value, true);
      $('#arc-replay').checked = cfg.useReplay; $('#arc-frozen').checked = cfg.useTarget; $('#arc-double').checked = cfg.double;
    }
    function restoreDefaults() {
      for (const k of Object.keys(DEF)) {
        if (k === 'bufferSize' && cfg.bufferSize === DEF.bufferSize) continue;
        cfg[k] = DEF[k];
        if (k === 'bufferSize') agent.resizeBuffer(DEF.bufferSize); else agent.cfg[k] = DEF[k];
      }
      syncKnobUI(); addEvent('defaults');
    }

    const EXPERIMENTS = [
      { title: 'Take away the frozen copy', body: 'The target now comes from the same network you are nudging, so every nudge also moves the goalposts.', watch: '__NOTARGET__', apply() { cfg.useTarget = false; agent.cfg.useTarget = false; return 'no frozen copy'; } },
      { title: 'Make it short-sighted', body: 'Set the future discount to 0.5. A reward five steps away now counts for about 3% of its value.', watch: '__GAMMA__', apply() { cfg.gamma = 0.5; agent.cfg.gamma = 0.5; return 'γ 0.5'; } },
      { title: 'Give it amnesia', body: 'Turn off replay memory. It learns from each moment once, in order, and then forgets it.', watch: '__NOREPLAY__', apply() { cfg.useReplay = false; agent.cfg.useReplay = false; return 'no replay'; } },
      { title: 'Take huge steps', body: 'Raise the learning rate to 0.01, ten times the default.', watch: '__LR__', apply() { cfg.lr = 0.01; agent.cfg.lr = 0.01; return 'lr 0.01'; } },
      { title: 'Never explore', body: 'Start over with a fresh brain that never makes a random move. It always does whatever it currently thinks is best.', watch: '__EPS0__', apply() { cfg.epsStart = 0; cfg.epsEnd = 0; newAgent(); setRunning(true); return 'fresh, no exploring'; } },
      { title: 'Watch a trained brain keep learning', body: 'Load the trained brain and keep training it with the default knobs.', watch: 'The exam line jumps near the top and stays there. The practice dots sit far lower: with 5% random moves, one wrong step next to the ghost costs 300 points. In our tests the same brain won 99% of games playing its best, but only 36% with random moves mixed in.', apply() { loadTrained(); setRunning(true); return null; } },
    ];

    function loadTrained() {
      agent.live.loadBase64(TRAINED_B64);
      agent.live.m = null; agent.live.v = null; agent.live.t = 0;
      agent.frozen.copyFrom(agent.live);
      agent.steps = Math.max(agent.steps, Math.round(agent.cfg.epsDecay));
      st.loadedTrained = true;
      obs = env.reset();
      runExam(true);
      addEvent('trained brain');
      renderBoard(env, obs, true); renderStats();
    }

    function init() {
      st.baseline = randomBaseline();
      newAgent();
      view = new MazeView($('#arcade-board'), env);
      qbars = new QBars($('#arcade-q'));
      const kroot = $('#arc-knobs');
      for (const [k, spec] of knobSpecs()) {
        knobs[k] = makeKnob(kroot, Object.assign({}, spec, { onInput: (v) => applyCfg(k, v), onChange: (v) => addEvent(spec.ev(v)) }));
      }
      const tog = (id, key, label) => $(id).addEventListener('change', (e) => { cfg[key] = e.target.checked; agent.cfg[key] = e.target.checked; addEvent(`${label} ${e.target.checked ? 'on' : 'off'}`); });
      tog('#arc-replay', 'useReplay', 'replay'); tog('#arc-frozen', 'useTarget', 'frozen copy'); tog('#arc-double', 'double', 'double');
      $('#arc-defaults').addEventListener('click', restoreDefaults);
      $('#arc-train').addEventListener('click', () => setRunning(!st.running));
      $('#arc-test').addEventListener('click', () => setTesting(!st.testing));
      bindSeg($('#arc-speed'), (v) => { st.mode = v; });
      $('#arc-load').addEventListener('click', loadTrained);
      $('#arc-reset').addEventListener('click', () => { setRunning(false); setTesting(false); cfg.epsStart = DEF.epsStart; cfg.epsEnd = DEF.epsEnd; newAgent(); renderBoard(env, obs, true); renderStats(); drawCurve(); $('#arc-train').textContent = 'Start training'; });
      $('#arc-send').addEventListener('click', () => { bench.useArcadeBrain(); selectTab('tab-bench'); $('#panel-bench').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' }); });
      const list = $('#arc-experiments');
      EXPERIMENTS.forEach((x) => {
        const d = el('div', 'exp', list);
        el('h4', '', d, x.title); el('p', '', d, x.body); el('p', 'watch', d, x.watch);
        const b = el('button', 'btn', d, 'Try it'); b.type = 'button';
        b.addEventListener('click', () => { const label = x.apply(); syncKnobUI(); if (label) addEvent(label); if (!st.running && !st.testing) setRunning(true); });
      });
      renderBoard(env, obs, true); renderStats();
    }
    function tick(ts) {
      const vis = !panel().hidden;
      if (st.testing) {
        if (ts - st.lastStep >= 130) { testStep(); st.lastStep = ts; if (vis) renderBoard(testEnv, testObs); }
      } else if (st.running) {
        if (st.mode === 'watch') { if (ts - st.lastStep >= 120) { trainStep(); st.lastStep = ts; if (vis) renderBoard(env, obs); } }
        else if (st.mode === 'fast') { for (let k = 0; k < 4; k++) trainStep(); if (vis) renderBoard(env, obs, true); }
        else {
          const t0 = performance.now();
          do { for (let k = 0; k < 6; k++) trainStep(); } while (performance.now() - t0 < 12);
          if (vis && ts - st.lastBoard > 120) { renderBoard(env, obs, true); st.lastBoard = ts; }
        }
      }
      if (vis && ts - st.lastStats > 150) { renderStats(); st.lastStats = ts; }
      if (vis && ts - st.lastCurve > 500 && (st.running || st.testing)) { drawCurve(); st.lastCurve = ts; }
    }
    function onShow() { renderBoard(st.testing ? testEnv : env, st.testing ? (testObs || testEnv.obs()) : obs, true); renderStats(); drawCurve(); }
    return { init, tick, onShow, drawCurve, get agent() { return agent; }, get state() { return st; }, loadTrained, setRunning, cfg, _train(n) { for (let i = 0; i < n; i++) trainStep(); renderStats(); drawCurve(); } };
  })();

  // ==================================================================== BENCH
  const bench = (() => {
    const env = new E.MiniPac({ aggression: 1 });
    const cellAt = (x, y) => env.cellOf[y * env.W + x];
    const st = {
      brainKind: 'trained', live: null, frozen: null, baseline: null, situation: null, action: 2, tool: 'pac', overlay: 'game',
      useFrozen: true, useDouble: true, gamma: 0.97, lr: 0.003, history: [], syncs: [], scenario: 'pellet', arcadeSteps: 0, flash: '',
    };
    let view, nextView, barsGuess, barsPick, barsValue, cur = null;
    const SCEN = [
      { id: 'pellet', label: 'Pellet ahead', pac: [4, 5], ghost: [4, 1], a: 2, note: 'An ordinary step: eat a pellet, pay 1 point for the move.' },
      { id: 'ghost', label: 'Walk into the ghost', pac: [3, 3], ghost: [4, 3], a: 3, note: 'Game over. The target is just the penalty, because there is no future to add.' },
      { id: 'flee', label: 'Run from the ghost', pac: [3, 3], ghost: [4, 3], a: 2, note: 'You survive this step, but the ghost follows. Any danger shows up through v, the value of where you land.' },
      { id: 'wall', label: 'Bump a wall', pac: [4, 5], ghost: [4, 1], a: 1, note: 'Walking into a wall costs a move and gains nothing, while the ghost gets closer.' },
      { id: 'last', label: 'Last pellet', pac: [5, 5], ghost: [1, 1], a: 3, pellets: [[6, 5]], note: 'Clearing the board ends the game with a +200 bonus.' },
      { id: 'hall', label: 'Empty hallway', pac: [4, 3], ghost: [1, 5], a: 3, rows: [1, 2, 4, 5], note: 'No pellet on this square, so the only reward is the step cost. Almost all of the target comes from the future.' },
    ];
    const NOTES = {
      trained: () => `Trained in this page's engine for ${TRAINED_META.step.toLocaleString()} steps. In ${(TRAINED_META.heldOutGames || 50).toLocaleString()} fresh test games it cleared the board ${Math.round(TRAINED_META.win * 100)}% of the time. Its guesses are usually close, so look for gaps in unusual situations.`,
      fresh: () => 'Random weights. Every guess is close to zero, so the reward dominates the target and the gaps are large.',
      arcade: () => `A copy of the Arcade's brain after ${st.arcadeSteps.toLocaleString()} steps of training.`,
    };

    function situationFor(sc) {
      const pellets = new Uint8Array(env.N);
      if (sc.pellets) for (const [x, y] of sc.pellets) pellets[cellAt(x, y)] = 1;
      else if (sc.rows) { for (let c = 0; c < env.N; c++) if (sc.rows.includes(env.xy(c)[1])) pellets[c] = 1; }
      else pellets.set(env.pelletStart);
      const s = { pellets, player: cellAt(...sc.pac), ghost: cellAt(...sc.ghost), ghostDir: -1, lastAction: sc.a };
      s.ghostPrev = s.ghost; s.pellets[s.player] = 0;
      return s;
    }
    function setBrain(kind) {
      st.brainKind = kind;
      if (kind === 'trained') st.live = newTrainedBrain();
      else if (kind === 'fresh') st.live = new E.MLP([116, ...HIDDEN, 4], 7);
      else { st.live = arcade.agent.live.clone(); st.arcadeSteps = arcade.agent.steps; }
      st.frozen = st.live.clone(); st.baseline = st.live.clone();
      st.history = []; st.syncs = [];
      setSeg($('#bench-brain'), kind);
      $('#bench-brain-note').textContent = NOTES[kind]();
      render();
    }
    function loadScenario(id) {
      const sc = SCEN.find((s) => s.id === id);
      st.scenario = id; st.situation = situationFor(sc); st.action = sc.a; st.history = []; st.syncs = []; st.flash = '';
      render();
    }

    function compute() {
      const s = st.situation, a = st.action;
      const oS = env.encode(s);
      env.setState({ ...s, pellets: s.pellets, steps: 0, score: 0, done: false, dead: false, won: false });
      const res = env.step(a, { deterministicGhost: true });
      const s2 = env.getState();
      const r = res.reward * SCALE;
      const qS = st.live.forward(oS);
      let qPick = null, qVal = null, aStar = -1, v = 0;
      if (!res.terminal) {
        qPick = st.live.forward(res.obs);
        qVal = st.useFrozen ? st.frozen.forward(res.obs) : qPick;
        aStar = st.useDouble ? E.argmax(qPick) : E.argmax(qVal);
        v = qVal[aStar];
      }
      const q = qS[a], target = r + st.gamma * v, gap = q - target;
      return { oS, s2, res, r, qS, q, qPick, qVal, aStar, v, target, gap, loss: E.huber(gap) };
    }

    function nudge(n) {
      let c = compute();
      if (!st.history.length) st.history.push({ q: c.q, t: c.target });
      const d = new Float32Array(4);
      for (let k = 0; k < n; k++) {
        st.live.forward(c.oS);
        d.fill(0); d[st.action] = E.huberGrad(c.gap);
        st.live.backward(d); st.live.sgdStep(st.lr, 0);
        c = compute();
        st.history.push({ q: c.q, t: c.target });
      }
      if (st.history.length > 600) { const drop = st.history.length - 600; st.history.splice(0, drop); st.syncs = st.syncs.map((i) => i - drop).filter((i) => i >= 0); }
      render();
    }

    function hypothetical(c) { const h = { ...st.situation, pellets: Uint8Array.from(st.situation.pellets), player: c }; h.pellets[c] = 0; return h; }
    function overlayData() {
      const N = env.N, s = st.situation;
      if (st.overlay === 'value') {
        const V = new Array(N).fill(null), arrows = new Array(N).fill(-1);
        let lo = Infinity, hi = -Infinity;
        for (let c = 0; c < N; c++) {
          if (c === s.ghost) continue;
          const q = st.live.forward(env.encode(hypothetical(c)));
          V[c] = Math.max(...q); arrows[c] = E.argmax(q);
          lo = Math.min(lo, V[c]); hi = Math.max(hi, V[c]);
        }
        const norm = V.map((v) => v == null ? null : (hi - lo < 1e-6 ? 0 : ((v - lo) / (hi - lo)) * 2 - 1));
        return { norm, arrows, labels: V.map((v) => v == null ? '' : num(v, 2)), lo, hi };
      }
      if (st.overlay === 'ripple') {
        const D = new Array(N).fill(null); let m = 0, ext = 0;
        for (let c = 0; c < N; c++) {
          if (c === s.ghost) continue;
          const o = env.encode(hypothetical(c));
          D[c] = st.live.forward(o)[st.action] - st.baseline.forward(o)[st.action];
          if (Math.abs(D[c]) > m) { m = Math.abs(D[c]); ext = D[c]; }
        }
        const norm = D.map((d) => d == null ? null : (m < 1e-4 ? 0 : d / m));
        return { norm, labels: D.map((d) => d == null ? '' : (m < 1e-4 ? '' : sgn(d, 2))), maxAbs: m, ext };
      }
      return null;
    }

    function drawChart() {
      const cv = $('#bench-chart');
      if (!cv || $('#panel-bench').hidden) return;
      const { ctx, w, h } = fitCanvas(cv);
      const H = st.history, ink2 = cssVar('--ink-3');
      ctx.font = '11px ' + cssVar('--font');
      if (H.length < 2) { ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('The guess and the target will be plotted here as you nudge.', w / 2, h / 2); return; }
      const L = 40, R = 10, T = 12, B = 22;
      let lo = Infinity, hi = -Infinity;
      for (const p of H) { lo = Math.min(lo, p.q, p.t); hi = Math.max(hi, p.q, p.t); }
      const pad = (hi - lo) * 0.15 || 0.1; lo -= pad; hi += pad;
      const X = (i) => L + (i / (H.length - 1)) * (w - L - R), Y = (v) => T + ((hi - v) / (hi - lo)) * (h - T - B);
      ctx.fillStyle = ink2; ctx.strokeStyle = ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
      for (const t of niceTicks(lo, hi, 3)) { ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(num(t, Math.abs(hi - lo) < 0.5 ? 2 : 1), L - 5, Y(t)); }
      ctx.strokeStyle = cssVar('--frozen'); ctx.setLineDash([3, 3]);
      for (const i of st.syncs) { ctx.beginPath(); ctx.moveTo(X(i), T); ctx.lineTo(X(i), h - B); ctx.stroke(); }
      ctx.setLineDash([]);
      const line = (key, color, width) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); H.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p[key])) : ctx.moveTo(X(i), Y(p[key])))); ctx.stroke(); };
      line('t', cssVar('--target'), 2.5);
      line('q', cssVar('--guess'), 2.5);
      ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(`nudges (${H.length - 1})`, L + (w - L - R) / 2, h - 5);
    }

    const PART_LABEL = { step: 'step', pellet: 'pellet', caught: 'caught', cleared: 'board cleared' };
    function render() {
      if (!st.live || !st.situation) return;
      cur = compute();
      const c = cur, s = st.situation, a = st.action;
      // boards
      const ov = overlayData();
      view.render({ ...s, lastAction: a, dead: false }, ov);
      const dest = env.nbr[s.player][a];
      view.setRing(st.overlay === 'game' && dest >= 0 ? dest : -1);
      nextView.instant(true);
      nextView.render(c.s2);
      const note = $('#bench-overlay-note');
      if (st.flash) note.textContent = st.flash;
      else if (st.overlay === 'value') note.textContent = `Value map: the best q the live network gives each square if Pac-Man stood there (ghost and pellets as shown). Greener means worth more than the other squares, redder means worth less. Arrows show the move it would pick. Range ${num(ov.lo)} to ${num(ov.hi)}.`;
      else if (st.overlay === 'ripple') note.textContent = ov.maxAbs < 1e-4 ? 'Ripple: nothing has changed yet. Nudge the weights, then come back here to see which other squares moved.' : `Ripple: how much q(${MOVE[a]}) changed since this brain was loaded, for every square Pac-Man could stand on. You only trained on one square, yet others moved: the weights are shared. Largest change ${sgn(ov.ext)}.`;
      else { const sc = SCEN.find((x) => x.id === st.scenario); note.textContent = sc ? sc.note : 'Your own situation. Pick a move with the arrows.'; }
      $$('#bench-scenarios .chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === st.scenario)));
      $$('#bench-pad button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.a === a)));
      // reward parts
      const parts = $('#bench-parts'); parts.textContent = '';
      for (const [k, v] of c.res.parts) el('span', 'rpart ' + (v < 0 ? 'neg' : 'pos'), parts, `${PART_LABEL[k]} ${pts(v)}`);
      if (c.res.bumped) el('span', 'rpart', parts, 'hit a wall');
      if (c.res.terminal) el('span', 'rpart end', parts, 'Game over');
      html($('#bench-rline'), `Reward: <b class="num">${pts(c.res.reward)}</b> points × 0.01 = <b class="num c-reward">r = ${sgn(c.r)}</b>`);
      // step 1
      barsGuess.render(c.qS, { hi: a, cls: 'guess' });
      $('#ro-move').textContent = MOVE[a];
      $('#ro-q').textContent = num(c.q);
      // step 2
      const pair = $('#future-pair'), over = $('#future-over'), ft = $('#future-text');
      if (c.res.terminal) {
        pair.hidden = true; over.hidden = false;
        ft.textContent = 'Normally we would ask the network what board S′ is worth.';
      } else {
        pair.hidden = false; over.hidden = true;
        const rng = [Math.min(...c.qPick, ...c.qVal), Math.max(...c.qPick, ...c.qVal)];
        if (!st.useFrozen) {
          pair.classList.add('single'); $('#bars-value').parentElement.hidden = true;
          $('#cap-picker').textContent = 'Live network picks the best next move and scores it';
          barsPick.render(c.qPick, { hi: c.aStar, star: c.aStar, cls: 'live', range: rng });
          ft.textContent = 'No frozen copy: the live network scores board S′ itself. That means every nudge you make will move the target as well.';
        } else {
          pair.classList.remove('single'); $('#bars-value').parentElement.hidden = false;
          if (st.useDouble) {
            $('#cap-picker').textContent = 'Live network picks the best next move';
            $('#cap-valuer').textContent = 'Frozen copy says what it is worth';
            barsPick.render(c.qPick, { hi: c.aStar, star: c.aStar, cls: 'live', range: rng });
            ft.innerHTML = 'From board S′ the <span class="hl hl-live">live network</span> picks the move that looks best (★). The <span class="hl hl-frozen">frozen copy</span>, an older snapshot of the same network, says what that move is worth. Splitting the jobs keeps one network\u2019s optimism from feeding on itself.';
          } else {
            $('#cap-picker').textContent = 'Live network (not used for v)';
            $('#cap-valuer').textContent = 'Frozen copy picks the best move and scores it';
            barsPick.render(c.qPick, { hi: -1, range: rng });
            ft.innerHTML = 'Plain DQN: the <span class="hl hl-frozen">frozen copy</span> both picks the best move from S′ (★) and scores it. Taking the biggest of several noisy guesses tends to overestimate, which is why Double DQN splits the jobs.';
          }
          barsValue.render(c.qVal, { hi: c.aStar, star: st.useDouble ? -1 : c.aStar, cls: 'frozen', range: rng });
        }
      }
      $('#ro-v').textContent = num(c.v);
      // step 3
      const g = st.gamma.toFixed(2);
      html($('#bench-formula'), `
        <span class="lhs">target =</span><span class="rhs"><span class="c-reward">r</span> + γ × <span class="c-frozen">v</span></span>
        <span class="lhs">=</span><span class="rhs"><span class="c-reward">${sgn(c.r)}</span> + ${g} × <span class="c-frozen">${c.res.terminal ? '0' : num(c.v)}</span>${c.res.terminal ? ' <span class="dim">(game over)</span>' : ''}</span>
        <span class="lhs">=</span><span class="rhs total">${num(c.target)}</span>`);
      // step 4
      drawGap(c);
      $('#ro-gap').textContent = sgn(c.gap);
      $('#ro-loss').textContent = c.loss.toFixed(4);
      const ag = Math.abs(c.gap);
      let words = ag < 0.02 ? 'Almost no gap: the brain already agrees with the evidence, so this memory would teach it very little.'
        : ag < 0.3 ? (c.gap < 0 ? 'The guess is a little too low. A nudge will raise it slightly.' : 'The guess is a little too high. A nudge will lower it slightly.')
        : (c.gap < 0 ? 'A big gap: the brain underrated this move. Surprises like this are where most learning happens.' : 'A big gap: the brain overrated this move. Surprises like this are where most learning happens.');
      if (ag > 1) words += ' Because the gap is bigger than 1, the loss grows in a straight line instead of squaring, so one surprising memory can\u2019t yank the weights too hard.';
      $('#gap-words').textContent = words;
      // step 5
      $('#nudge-words').textContent = nudgeWords(c);
      drawChart();
    }
    function nudgeWords(c) {
      const H = st.history;
      if (H.length < 2) return 'Press a nudge button to watch the guess move toward the target.';
      const n = H.length - 1, q0 = H[0].q, t0 = H[0].t;
      const lastSync = st.syncs.length ? st.syncs[st.syncs.length - 1] : -1;
      const recent = H.slice(-8).map((p) => p.q - p.t);
      let flips = 0; for (let i = 1; i < recent.length; i++) if (Math.sign(recent[i]) !== Math.sign(recent[i - 1]) && Math.abs(recent[i]) > 0.01) flips++;
      if (flips >= 3 || Math.abs(c.gap) > Math.abs(H[0].q - H[0].t) * 1.5 + 0.05) return `The steps are too big: q is overshooting the target and bouncing back and forth. Lower the learning rate and press Undo all nudges.`;
      if (lastSync === H.length - 1) return 'Syncing copied the live weights into the frozen copy, so the target jumped to reflect everything the live network has learned. In real training this happens every few hundred steps.';
      if (Math.abs(c.target - t0) > 0.02 && !st.useFrozen) return `After ${n} nudges q went from ${num(q0)} to ${num(c.q)}, but the target moved too, from ${num(t0)} to ${num(c.target)}. With no frozen copy the network is chasing its own tail.`;
      if (Math.abs(c.target - t0) > 0.02) return `After ${n} nudges q went from ${num(q0)} to ${num(c.q)}. The target moved from ${num(t0)} to ${num(c.target)} because the live network changed its mind about which next move is best, or because of a sync.`;
      return `After ${n} nudges q went from ${num(q0)} to ${num(c.q)}. The target stayed at ${num(c.target)} because the frozen copy didn\u2019t change.${Math.abs(c.gap) < 0.01 ? ' The gap is closed: this memory has nothing left to teach.' : ''} Switch the board to Ripple to see what else moved.`;
    }
    function drawGap(c) {
      const root = $('#bench-gapviz');
      if (!root._b) {
        root._b = {
          gap: el('div', 'gv-gap', root), axis: el('div', 'gv-axis', root), axl: el('div', 'gv-axis-label num', root, '0'),
          g: el('div', 'gv-bar g', root), t: el('div', 'gv-bar t', root), gl: el('div', 'gv-lab', root), tl: el('div', 'gv-lab', root),
        };
      }
      const b = root._b;
      let lo = Math.min(0, c.q, c.target), hi = Math.max(0, c.q, c.target);
      const pad = (hi - lo) * 0.3 || 0.2; lo -= pad; hi += pad;
      const X = (v) => ((v - lo) / (hi - lo)) * 100;
      const x0 = X(0);
      b.axis.style.left = x0 + '%'; b.axl.style.left = x0 + '%';
      const bar = (node, v) => { const xv = X(v); node.style.left = Math.min(x0, xv) + '%'; node.style.width = `max(2px, ${Math.abs(xv - x0)}%)`; };
      bar(b.g, c.q); bar(b.t, c.target);
      const lab = (node, v, text, top, cls) => {
        const xv = X(v), inside = v >= 0 ? xv > 70 : xv < 30;
        node.className = 'gv-lab ' + cls + (inside ? ' inside' : ''); node.innerHTML = text; node.style.top = top + 'px';
        if (v >= 0) { node.style.left = `calc(${xv}% + 6px)`; node.style.transform = xv > 70 ? 'translateX(calc(-100% - 12px))' : ''; }
        else { node.style.left = `calc(${xv}% - 6px)`; node.style.transform = xv < 30 ? 'translateX(12px)' : 'translateX(-100%)'; }
      };
      lab(b.gl, c.q, `guess <span class="num">${num(c.q)}</span>`, 4, 'c-guess');
      lab(b.tl, c.target, `target <span class="num">${num(c.target)}</span>`, 34, 'c-target');
      const xa = X(Math.min(c.q, c.target)), xb = X(Math.max(c.q, c.target));
      b.gap.style.left = xa + '%'; b.gap.style.width = `max(2px, ${xb - xa}%)`;
    }

    function tap(c) {
      const s = st.situation; st.flash = '';
      if (st.tool === 'pac') {
        if (c === s.ghost) { st.flash = 'Pac-Man can\u2019t start on the ghost\u2019s square. Move the ghost first.'; render(); return; }
        s.player = c; s.pellets[c] = 0;
      } else if (st.tool === 'ghost') {
        if (c === s.player) { st.flash = 'The ghost can\u2019t start on Pac-Man\u2019s square.'; render(); return; }
        s.ghost = c; s.ghostPrev = c;
      } else {
        if (c === s.player) { st.flash = 'Pac-Man has already eaten whatever was on his square.'; render(); return; }
        s.pellets[c] ^= 1;
      }
      st.scenario = null; st.history = []; st.syncs = [];
      render();
    }
    function setAction(a) { st.action = a; st.history = []; st.syncs = []; st.flash = ''; render(); }

    function init() {
      view = new MazeView($('#bench-board'), env, { onTap: tap });
      view.instant(true);
      nextView = new MazeView($('#bench-next'), env, { mini: true });
      barsGuess = new QBars($('#bars-guess')); barsPick = new QBars($('#bars-pick')); barsValue = new QBars($('#bars-value'));
      const chips = $('#bench-scenarios');
      for (const sc of SCEN) { const b = el('button', 'chip', chips, sc.label); b.type = 'button'; b.dataset.id = sc.id; b.addEventListener('click', () => loadScenario(sc.id)); }
      bindSeg($('#bench-brain'), setBrain);
      bindSeg($('#bench-overlay'), (v) => { st.overlay = v; st.flash = ''; render(); });
      bindSeg($('#bench-tool'), (v) => { st.tool = v; });
      $$('#bench-pad button').forEach((b) => b.addEventListener('click', () => setAction(+b.dataset.a)));
      $('#panel-bench').addEventListener('keydown', (e) => {
        if (e.target.matches('input, textarea, select')) return;
        const k = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 }[e.key];
        if (k != null && e.target.closest('#bench-board, #bench-pad')) { e.preventDefault(); setAction(k); }
      });
      $('#bench-frozen').addEventListener('change', (e) => { st.useFrozen = e.target.checked; $('#bench-double').disabled = !st.useFrozen; render(); });
      $('#bench-double').addEventListener('change', (e) => { st.useDouble = e.target.checked; render(); });
      makeKnob($('#bench-gamma-knob'), { label: 'Future discount (γ)', hint: 'How much “where you landed” counts. At 0, only the immediate reward matters.', min: 0, max: 0.995, value: st.gamma, step: 0.005, fmt: (v) => v.toFixed(3), onInput: (v) => { st.gamma = v; render(); } });
      makeKnob($('#bench-lr-knob'), { label: 'Learning rate', hint: 'Size of each nudge. Try 0.01 or more to see what overshooting looks like.', min: 0.0003, max: 0.05, log: true, value: st.lr, sig: 2, fmt: (v) => v.toPrecision(2), onInput: (v) => { st.lr = v; } });
      $$('[data-nudge]').forEach((b) => b.addEventListener('click', () => nudge(+b.dataset.nudge)));
      $('#bench-sync').addEventListener('click', () => { if (!st.history.length) st.history.push({ q: cur.q, t: cur.target }); st.frozen.copyFrom(st.live); const c = compute(); st.history.push({ q: c.q, t: c.target }); st.syncs.push(st.history.length - 1); render(); });
      $('#bench-undo').addEventListener('click', () => { st.live.copyFrom(st.baseline); st.frozen.copyFrom(st.baseline); st.history = []; st.syncs = []; render(); });
      $('#param-count').textContent = new E.MLP([116, ...HIDDEN, 4], 1).paramCount.toLocaleString();
      const rk = $('#bench-reward-knobs'), rknobs = {};
      const R = [['pellet', 'Pellet', 0, 50, 1], ['step', 'Each move', -10, 0, 0.5], ['death', 'Caught by the ghost', -1000, 0, 10], ['clear', 'Board cleared', 0, 500, 10]];
      for (const [k, label, min, max, step] of R) rknobs[k] = makeKnob(rk, { label, min, max, step, value: E.DEFAULT_REWARDS[k], fmt: (v) => pts(v), onInput: (v) => { env.rewards[k] = v; render(); } });
      $('#bench-reward-reset').addEventListener('click', () => { for (const [k] of R) { env.rewards[k] = E.DEFAULT_REWARDS[k]; rknobs[k].set(E.DEFAULT_REWARDS[k], true); } render(); });
      st.situation = situationFor(SCEN[0]);
      setBrain('trained');
    }
    function useArcadeBrain() { setBrain('arcade'); }
    return { init, render, onShow: () => { render(); }, useArcadeBrain, get state() { return st; }, compute, nudge, loadScenario, setAction, setBrain };
  })();

  // ===================================================================== TANK
  const tank = (() => {
    const world = new E.GridWorld(7, 6);
    const NS = world.W * world.H;
    const st = {
      kind: 'table', table: new E.TableQ(NS), net: null, frozen: null, buffer: [], agent: -1, epSteps: 0, episodes: 0, updates: 0, sweeps: 0,
      running: false, gamma: 0.9, eps: 0.3, alpha: 0.5, lrNet: 0.01, speed: 6, showNums: false, tool: '1', last: null, acc: 0, lastT: 0,
      rng: E.makeRng(21), sinceSync: 0,
    };
    let cells = [], pac, knobs = {};
    function resetNet() { st.net = new E.NetQ(world, 32, 5 + ((Math.random() * 1000) | 0)); st.frozen = st.net.snapshot(); st.buffer = []; st.sinceSync = 0; }
    resetNet();
    const brain = () => (st.kind === 'table' ? st.table : st.net);

    function build() {
      const root = $('#tank-board'); root.textContent = '';
      root.style.setProperty('--cols', world.W); root.style.setProperty('--rows', world.H);
      cells = [];
      for (let s = 0; s < NS; s++) {
        const b = el('button', 'tcell', root); b.type = 'button';
        b.setAttribute('aria-label', `Square in column ${(s % world.W) + 1}, row ${Math.floor(s / world.W) + 1}`);
        const tint = el('div', 'tint', b), icon = el('div', 'icon', b), val = el('div', 'cval', b), arr = el('div', 'carr', b);
        arr.hidden = true;
        b.addEventListener('click', () => tap(s));
        cells.push({ b, tint, icon, val, arr });
      }
      pac = el('div', 'actor pac', root); pac.hidden = true;
    }
    function place(s) { const x = s % world.W, y = Math.floor(s / world.W); pac.style.transform = `translate(calc(${x * 100}% + ${x * 3}px), calc(${y * 100}% + ${y * 3}px))`; }

    function tap(s) {
      const t = +st.tool;
      if (world.type[s] === t) return;
      world.type[s] = t;
      if (st.agent === s || !world.isState(st.agent)) st.agent = -1;
      render();
    }
    function knownCount() {
      let known = 0, total = 0;
      for (const s of world.states()) { total++; const q = brain().values(s); if (Math.max(...q) > 0.1) known++; }
      return [known, total];
    }
    function stepOnce() {
      if (st.agent < 0 || !world.isState(st.agent)) {
        st.agent = world.randomStart(); st.epSteps = 0;
        if (st.agent < 0) return false;
        st.episodes++;
      }
      const s = st.agent, q = brain().values(s);
      const a = st.rng() < st.eps ? st.rng.int(4) : tieArgmax(q, st.rng);
      const tr = world.transition(s, a);
      let guess, target, after, v;
      if (st.kind === 'table') {
        v = tr.terminal ? 0 : Math.max(...st.table.values(tr.next));
        target = tr.reward + st.gamma * v;
        guess = st.table.update(s, a, target, st.alpha);
        after = st.table.values(s)[a];
      } else {
        st.buffer.push({ s, a, next: tr.next, reward: tr.reward, terminal: tr.terminal });
        if (st.buffer.length > 2000) st.buffer.shift();
        v = tr.terminal ? 0 : Math.max(...st.frozen.values(tr.next));
        target = tr.reward + st.gamma * v; guess = q[a];
        const batch = [{ s, a, target }];
        for (let k = 0; k < 15; k++) {
          const b = st.buffer[st.rng.int(st.buffer.length)];
          batch.push({ s: b.s, a: b.a, target: b.reward + (b.terminal ? 0 : st.gamma * Math.max(...st.frozen.values(b.next))) });
        }
        st.net.train(batch, st.lrNet);
        after = st.net.values(s)[a];
        if (++st.sinceSync >= 30) { st.frozen = st.net.snapshot(); st.sinceSync = 0; }
      }
      st.updates++; st.epSteps++;
      st.last = { s, a, next: tr.next, r: tr.reward, v, guess, target, after, terminal: tr.terminal };
      st.agent = tr.terminal || st.epSteps >= world.maxSteps ? -1 : tr.next;
      return true;
    }
    function sweep() {
      const states = world.states();
      if (!states.length) return;
      if (st.kind === 'table') {
        const snap = st.table.snapshot();
        for (const s of states) for (let a = 0; a < 4; a++) {
          const tr = world.transition(s, a);
          st.table.update(s, a, tr.reward + (tr.terminal ? 0 : st.gamma * Math.max(...snap.values(tr.next))), 1);
        }
        st.updates += states.length * 4;
      } else {
        const snap = st.net.snapshot(), batch = [];
        for (const s of states) for (let a = 0; a < 4; a++) {
          const tr = world.transition(s, a);
          batch.push({ s, a, target: tr.reward + (tr.terminal ? 0 : st.gamma * Math.max(...snap.values(tr.next))) });
        }
        for (let k = 0; k < 15; k++) st.net.train(batch, st.lrNet);
        st.frozen = st.net.snapshot(); st.updates += 15;
      }
      st.sweeps++; st.agent = -1;
      st.last = { sweep: true };
    }
    const where = (s) => `column ${(s % world.W) + 1}, row ${Math.floor(s / world.W) + 1}`;
    function render() {
      const scale = Math.max(Math.abs(world.rewards.cherry), Math.abs(world.rewards.ghost), 1);
      const b = brain();
      for (let s = 0; s < NS; s++) {
        const c = cells[s], t = world.type[s];
        c.b.className = 'tcell' + (t === E.WALL ? ' wall' : t === E.CHERRY ? ' cherry' : t === E.GHOST ? ' ghostc' : '');
        if (t !== E.EMPTY) { c.tint.style.background = ''; c.val.textContent = ''; c.arr.hidden = true; continue; }
        const q = b.values(s), V = Math.max(...q), spread = V - Math.min(...q);
        c.tint.style.background = heat(V / scale);
        c.val.textContent = st.showNums ? num(V, 1) : '';
        const show = spread > 0.05;
        c.arr.hidden = !show;
        if (show) c.arr.style.setProperty('--rot', ARR_ROT[E.argmax(q)] + 'deg');
      }
      if (st.last && !st.last.sweep) { cells[st.last.s].b.classList.add('flash-guess'); if (!st.last.terminal && st.last.next !== st.last.s) cells[st.last.next].b.classList.add('flash-next'); }
      pac.hidden = st.agent < 0; if (st.agent >= 0) place(st.agent);
      // readout
      const L = st.last, lu = $('#tank-last');
      if (!L) lu.innerHTML = 'Nothing learned yet. Press <b>Explore</b> to let Pac-Man wander, or <b>Sweep</b> to grade every square at once.';
      else if (L.sweep) {
        const [k, tot] = knownCount();
        lu.innerHTML = `Sweep ${st.sweeps}: every square graded once${st.kind === 'table' ? ', each against a frozen snapshot of the map from before the sweep' : ' against a frozen copy of the network, then 15 nudges'}. <b>${k} of ${tot}</b> squares now know about the cherry.`;
      } else {
        lu.innerHTML = `From ${where(L.s)}, moved ${GLYPH[L.a]}${L.terminal ? (L.r > 0 ? ' onto the cherry' : ' into the ghost') : ''}.<br>
          target = <span class="num c-reward">${sgn(L.r)}</span> + ${st.gamma.toFixed(2)} × <span class="num c-frozen">${num(L.v)}</span> = <span class="num c-target">${num(L.target)}</span>${L.terminal ? ' (game over, v = 0)' : ''}<br>
          guess was <span class="num c-guess">${num(L.guess)}</span>, now <span class="num c-guess">${num(L.after)}</span>`;
      }
      const [k, tot] = knownCount();
      const dl = $('#tank-stats');
      html(dl, `<div><dt>Games</dt><dd>${st.episodes.toLocaleString()}</dd></div><div><dt>Updates</dt><dd>${st.updates.toLocaleString()}</dd></div>
        <div><dt>Squares that know about the cherry</dt><dd>${k} of ${tot}</dd></div>
        <div><dt>${st.kind === 'table' ? 'Numbers in the table' : 'Weights in the network'}</dt><dd>${st.kind === 'table' ? (tot * 4).toLocaleString() : st.net.net.paramCount.toLocaleString()}</dd></div>`);
    }
    function clearFlashes() { cells.forEach((c) => c.b.classList.remove('flash-guess', 'flash-next')); }
    function setRunning(on) { st.running = on; $('#tank-run').textContent = on ? 'Pause' : 'Explore'; st.lastT = 0; st.acc = 0; }
    function brainNote() {
      $('#tank-brain-note').textContent = st.kind === 'table'
        ? 'One number per square per move. Updating one square changes only that square.'
        : `A ${st.net.net.paramCount}-weight network that sees only the column and row. Squares share weights, so one update tints many squares. It also uses replay memory and a frozen copy, like the real DQN.`;
      knobs.alpha.wrap.hidden = st.kind !== 'table'; knobs.lrNet.wrap.hidden = st.kind === 'table';
    }
    function init() {
      build();
      bindSeg($('#tank-tool'), (v) => { st.tool = v; });
      bindSeg($('#tank-brain'), (v) => { st.kind = v; st.last = null; clearFlashes(); brainNote(); render(); });
      $('#tank-run').addEventListener('click', () => setRunning(!st.running));
      $('#tank-step').addEventListener('click', () => { setRunning(false); clearFlashes(); stepOnce(); render(); });
      $('#tank-sweep').addEventListener('click', () => { setRunning(false); clearFlashes(); sweep(); render(); });
      $('#tank-reset').addEventListener('click', () => {
        setRunning(false);
        if (st.kind === 'table') st.table = new E.TableQ(NS); else resetNet();
        st.episodes = 0; st.updates = 0; st.sweeps = 0; st.agent = -1; st.last = null; clearFlashes(); render();
      });
      $('#tank-nums').addEventListener('change', (e) => { st.showNums = e.target.checked; render(); });
      const K = $('#tank-knobs');
      knobs.speed = makeKnob(K, { label: 'Exploring speed', min: 1, max: 300, log: true, value: st.speed, sig: 2, fmt: (v) => `${Math.round(v)} moves/s`, onInput: (v) => { st.speed = v; } });
      knobs.gamma = makeKnob(K, { label: 'Future discount (γ)', hint: 'Each step away from the cherry multiplies its value by this.', min: 0, max: 0.99, step: 0.01, value: st.gamma, onInput: (v) => { st.gamma = v; } });
      knobs.eps = makeKnob(K, { label: 'Random-move chance (ε)', hint: 'How often Pac-Man ignores the map and wanders.', min: 0, max: 1, step: 0.01, value: st.eps, onInput: (v) => { st.eps = v; } });
      knobs.alpha = makeKnob(K, { label: 'Learning rate', hint: 'For the table: how far each update moves the guess toward the target (1 = all the way).', min: 0.05, max: 1, step: 0.05, value: st.alpha, onInput: (v) => { st.alpha = v; } });
      knobs.lrNet = makeKnob(K, { label: 'Learning rate', hint: 'For the network: the size of each Adam step.', min: 0.001, max: 0.05, log: true, sig: 2, value: st.lrNet, fmt: (v) => v.toPrecision(2), onInput: (v) => { st.lrNet = v; } });
      knobs.step = makeKnob(K, { label: 'Cost per move', hint: 'A small penalty for every move makes long routes worse.', min: -1, max: 0, step: 0.05, value: 0, fmt: (v) => num(v), onInput: (v) => { world.rewards.step = v; } });
      brainNote();
      render();
    }
    function tick(ts) {
      if (!st.running) return;
      if (!st.lastT) { st.lastT = ts; return; }
      st.acc += ((ts - st.lastT) / 1000) * st.speed; st.lastT = ts;
      let n = Math.min(Math.floor(st.acc), 80);
      if (n <= 0) return;
      st.acc -= n;
      clearFlashes();
      $('#tank-board').classList.toggle('instant', st.speed > 20 || reduceMotion);
      while (n-- > 0) if (!stepOnce()) break;
      if (!$('#panel-tank').hidden) render();
    }
    return { init, tick, onShow: render, get state() { return st; }, world, sweep, stepOnce, render };
  })();

  // ============================================================== FOUR GHOSTS
  // The arcade rules (Arcade, from arcade/arcade.js) with a brain trained offline in PyTorch on the very same engine.
  const four = (() => {
    const AR = Arcade;
    const META = __ARCADE_META__;
    const CURVE = __ARCADE_CURVE__;
    const B64 = '__ARCADE_B64__';
    const GCOL = ['#ff5a5f', '#ff9fd6', '#5ad7ef', '#ffb45a'];
    const BLUE = '#3553e0', PALE = '#eef1ff';
    const STICKY = 0.25; // the brain plays with the sticky stick it trained with
    const GHOSTS = [
      { name: 'Blinky', nick: 'the chaser', rule: 'Targets Pac-Man’s own tile. Late in each level he speeds up and keeps chasing even during scatter waves (Cruise Elroy).' },
      { name: 'Pinky', nick: 'the ambusher', rule: 'Targets 4 tiles ahead of Pac-Man. When Pac-Man faces up it is 4 up and 4 left: a bug in the original code.' },
      { name: 'Inky', nick: 'the fickle one', rule: 'Takes the tile 2 ahead of Pac-Man and doubles the arrow from Blinky to it, so where he goes depends on Blinky.' },
      { name: 'Clyde', nick: 'the shy one', rule: 'Chases Pac-Man from 8 or more tiles away. Any closer and he heads back to his corner.' },
    ];
    const CORNER = ['top-right', 'top-left', 'bottom-right', 'bottom-left'];
    const FRUIT_NAMES = { cherries: 'cherries', strawberry: 'a strawberry', peach: 'a peach', apple: 'an apple', grapes: 'grapes', galaxian: 'a ship', bell: 'a bell', key: 'a key' };
    const st = {
      who: 'brain', running: true, speed: 1, layers: { targets: false, view: false }, joy: null, deciding: false,
      q: null, qAct: -1, coach: false, decisions: 0, gameDecisions: 0, games: 0, ghostsEaten: 0, overWait: 0, seed: 1001,
      acc: 0, lastTs: 0, lastDom: 0, pacPhase: 0, lastPac: [0, 0], shown: false,
    };
    let env = null, game = null, net = null, qbars = null, cv = null, ctx = null, scale = 1, dpr = 1;
    let walls = null, wallsFlash = null, pal = null;
    const x255 = new Float32Array(AR.OBS.size);
    const panel = () => $('#panel-four');
    const secs = (frames) => `${(Math.max(0, frames) / 60).toFixed(1)} s`;
    const fmt = (n) => Math.round(n).toLocaleString('en-US');

    // ------------------------------------------------------------ brain
    function loadNet() {
      const bytes = E.b64ToBytes(B64);
      let flat;
      if (META.format === 'f16') { // half floats: 16 bits per weight keeps the page small
        const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        flat = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) {
          const v = h[i], s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 31, f = v & 1023;
          flat[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? s * Infinity : s * (1 + f / 1024) * 2 ** (e - 15);
        }
      } else flat = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const m = new E.MLP(META.sizes, 1); m.loadFlat(flat);
      return m;
    }
    // Same inputs the trainer stored: every number rounded to steps of 1/255.
    function think() {
      const o = env.observe();
      for (let k = 0; k < o.length; k++) x255[k] = Math.round(o[k] * 255) / 255;
      return net.forward(x255);
    }
    function decideNow() {
      const q = think();
      st.q = q; st.qAct = E.argmax(q); st.coach = false;
      env.begin(st.qAct); st.deciding = true; st.decisions++; st.gameDecisions++;
    }

    // ------------------------------------------------------------ play
    function newGame(seed = st.seed + 1) {
      st.seed = seed; env.reset(seed);
      st.deciding = false; st.gameDecisions = 0; st.ghostsEaten = 0; st.overWait = 0; st.games++;
      st.q = null; st.qAct = -1; st.joy = null;
      st.lastPac = [game.pac.x, game.pac.y];
      renderDom(true);
    }
    // One arcade frame (1/60 s of game time).
    function advance() {
      if (game.over) { if (++st.overWait > 150 && st.who === 'brain') newGame(); return; }
      if (st.who === 'brain') {
        if (!st.deciding) decideNow();
        const r = env.tick();
        if (r) { st.deciding = false; st.ghostsEaten += r.ghosts; }
      } else {
        const p = game.pac, t0 = (p.x >> 3) + (p.y >> 3) * 28;
        const ev = game.step(st.joy == null ? null : AR.ACTION_DIRS[st.joy]);
        for (const e of ev) if (e.type === 'ghostEaten') st.ghostsEaten++;
        if (!game.over && (p.x >> 3) + (p.y >> 3) * 28 !== t0) { const q = think(); st.q = q; st.qAct = E.argmax(q); st.coach = true; }
      }
    }
    function oneMove() {
      if (st.who === 'brain') { let n = 0; do advance(); while (st.deciding && ++n < 5000); }
      else for (let k = 0; k < 8; k++) advance();
      draw(); renderDom(true);
    }
    function tick(ts) {
      if (!st.shown || panel().hidden) { st.lastTs = ts; return; }
      const dt = Math.min(100, ts - (st.lastTs || ts)); st.lastTs = ts;
      if (st.running) {
        if (st.speed === 'max') {
          const t0 = performance.now();
          for (let n = 0; n < 40000; n++) { advance(); if (n % 64 === 63 && performance.now() - t0 > 9) break; }
        } else {
          st.acc += dt * 0.06 * st.speed;
          let n = Math.min(Math.floor(st.acc), 2000); st.acc -= Math.floor(st.acc);
          while (n-- > 0) advance();
        }
      }
      draw();
      if (ts - st.lastDom > 160) { st.lastDom = ts; renderDom(); } else renderStatus();
    }

    // ------------------------------------------------------------ drawing
    function readPalette() {
      pal = { screen: cssVar('--screen'), wall: cssVar('--wall'), wallHi: cssVar('--wall-hi'), pellet: cssVar('--pellet'), pac: cssVar('--pac'), font: cssVar('--font') };
    }
    function resize() {
      if (!cv) return;
      dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 300;
      cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr * 288 / 224);
      scale = cv.width / 224;
      ctx = cv.getContext('2d');
      readPalette();
      walls = renderWalls(false); wallsFlash = renderWalls(true);
    }
    // The maze layout is the arcade's; the look is ours: merged wall blocks with a bright rim, like the Arcade station.
    function renderWalls(flash) {
      const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
      const k = c.getContext('2d');
      k.fillStyle = pal.screen; k.fillRect(0, 0, c.width, c.height);
      k.setTransform(scale, 0, 0, scale, 0, 0);
      const isWall = (x, y) => x >= 0 && x < 28 && y >= 3 && y <= 33 && AR.cellAt(x, y) === AR.WALL;
      for (const [inset, color] of [[1.4, flash ? '#f4f6ff' : pal.wallHi], [2.6, flash ? '#6b7fc8' : pal.wall]]) {
        k.fillStyle = color;
        for (let y = 3; y <= 33; y++) for (let x = 0; x < 28; x++) {
          if (!isWall(x, y)) continue;
          const X = x * 8, Y = y * 8, i = inset, s = 8 - 2 * i;
          k.fillRect(X + i, Y + i, s, s);
          if (isWall(x + 1, y)) k.fillRect(X + 8 - i, Y + i, 2 * i, s);
          if (isWall(x, y + 1)) k.fillRect(X + i, Y + 8 - i, s, 2 * i);
          if (isWall(x + 1, y) && isWall(x, y + 1) && isWall(x + 1, y + 1)) k.fillRect(X + 8 - i, Y + 8 - i, 2 * i, 2 * i);
        }
      }
      k.fillStyle = '#ffb8de'; k.fillRect(13 * 8, 15 * 8 + 3, 16, 2); // the house door
      return c;
    }
    function pacShape(k, x, y, r, dir, mouth, color) {
      const rot = [-Math.PI / 2, Math.PI, Math.PI / 2, 0][dir];
      k.fillStyle = color; k.beginPath();
      if (mouth <= 0.01) k.arc(x, y, r, 0, 2 * Math.PI);
      else { k.moveTo(x, y); k.arc(x, y, r, rot + mouth, rot + 2 * Math.PI - mouth); k.closePath(); }
      k.fill();
    }
    function ghostShape(k, x, y, body, dir, look = 'normal') {
      if (look !== 'eyes') {
        k.fillStyle = body; k.beginPath();
        k.arc(x, y - 0.8, 6.3, Math.PI, 0);
        k.lineTo(x + 6.3, y + 4.6); k.arcTo(x + 6.3, y + 6.8, x + 4.1, y + 6.8, 2.2);
        k.lineTo(x - 4.1, y + 6.8); k.arcTo(x - 6.3, y + 6.8, x - 6.3, y + 4.6, 2.2);
        k.closePath(); k.fill();
      }
      if (look === 'blue' || look === 'flash') { // frightened face: two small eyes and a wavy mouth
        const face = look === 'blue' ? '#ffd2c8' : '#ff5a5f';
        k.fillStyle = face; k.fillRect(x - 3, y - 2.5, 2, 2); k.fillRect(x + 1, y - 2.5, 2, 2);
        k.strokeStyle = face; k.lineWidth = 0.9; k.beginPath(); k.moveTo(x - 4, y + 3);
        for (let i = 1; i <= 4; i++) k.lineTo(x - 4 + i * 2, y + (i % 2 ? 1.8 : 3));
        k.stroke();
        return;
      }
      const dx = AR.DX[dir] * 1.2, dy = AR.DY[dir] * 1.3;
      for (const ex of [-2.5, 2.5]) {
        k.fillStyle = '#fff'; k.beginPath(); k.ellipse(x + ex + dx * 0.4, y - 1.4 + dy * 0.4, 1.9, 2.3, 0, 0, 2 * Math.PI); k.fill();
        k.fillStyle = '#233a99'; k.beginPath(); k.arc(x + ex + dx, y - 1.4 + dy, 1.1, 0, 2 * Math.PI); k.fill();
      }
    }
    // Generic fruit icons (our own drawings, not the arcade's sprites).
    function fruitIcon(k, name, x, y, s = 1) {
      k.save(); k.translate(x, y); k.scale(s, s);
      const circ = (cx, cy, r, c) => { k.fillStyle = c; k.beginPath(); k.arc(cx, cy, r, 0, 2 * Math.PI); k.fill(); };
      const stem = (x1, y1, x2, y2, c = '#7bd67f') => { k.strokeStyle = c; k.lineWidth = 1; k.beginPath(); k.moveTo(x1, y1); k.quadraticCurveTo((x1 + x2) / 2 + 1, y1 - 2, x2, y2); k.stroke(); };
      switch (name) {
        case 'cherries': stem(-2.4, 2, 2, -5.5); stem(2.6, 2.4, 2, -5.5); circ(-2.4, 2.8, 2.6, '#ff4f6d'); circ(2.8, 3.2, 2.6, '#ff4f6d'); break;
        case 'strawberry':
          k.fillStyle = '#ff4760'; k.beginPath(); k.moveTo(-5, -2); k.quadraticCurveTo(0, -5, 5, -2); k.quadraticCurveTo(4, 4, 0, 6); k.quadraticCurveTo(-4, 4, -5, -2); k.fill();
          k.fillStyle = '#ffe0a0'; for (const [a, b] of [[-2, 0], [1.5, -0.5], [-0.5, 2.5], [2, 2.6]]) k.fillRect(a, b, 0.9, 0.9);
          k.fillStyle = '#6fdc7a'; k.fillRect(-2.5, -4.6, 5, 1.3); break;
        case 'peach': circ(0, 1, 5, '#ffa64d'); k.fillStyle = '#6fdc7a'; k.beginPath(); k.ellipse(1.8, -4, 2.4, 1.1, -0.5, 0, 2 * Math.PI); k.fill(); break;
        case 'apple': circ(0, 1.2, 5, '#ff5a52'); stem(0, -3, 1.2, -6, '#b77b44'); k.fillStyle = '#6fdc7a'; k.beginPath(); k.ellipse(2.7, -4.6, 1.8, 0.9, -0.4, 0, 2 * Math.PI); k.fill(); break;
        case 'grapes': for (const [a, b] of [[-2.6, -1], [0, -1.4], [2.6, -1], [-1.3, 1.4], [1.3, 1.4], [0, 3.8]]) circ(a, b, 1.9, '#a86bff'); stem(0, -3, 1, -6); break;
        case 'galaxian': // a generic little ship
          k.fillStyle = '#ffd23f'; k.beginPath(); k.moveTo(0, -5.5); k.lineTo(2, 1); k.lineTo(-2, 1); k.closePath(); k.fill();
          k.fillStyle = '#5ad7ef'; k.beginPath(); k.moveTo(-5.5, 3.5); k.lineTo(-1.5, -0.5); k.lineTo(-1.5, 3); k.closePath(); k.fill();
          k.beginPath(); k.moveTo(5.5, 3.5); k.lineTo(1.5, -0.5); k.lineTo(1.5, 3); k.closePath(); k.fill();
          k.fillStyle = '#ff5a5f'; k.fillRect(-1, 1, 2, 3.5); break;
        case 'bell':
          k.fillStyle = '#ffd23f'; k.beginPath(); k.moveTo(-5, 4); k.quadraticCurveTo(-4.5, -5.5, 0, -5.5); k.quadraticCurveTo(4.5, -5.5, 5, 4); k.closePath(); k.fill();
          circ(0, 5, 1.4, '#eef1ff'); break;
        default: // key
          k.strokeStyle = '#8fd4ff'; k.lineWidth = 1.6; k.beginPath(); k.arc(0, -3, 2.4, 0, 2 * Math.PI); k.stroke();
          k.beginPath(); k.moveTo(0, -0.6); k.lineTo(0, 6); k.moveTo(0, 3.5); k.lineTo(2.2, 3.5); k.moveTo(0, 5.6); k.lineTo(1.8, 5.6); k.stroke();
      }
      k.restore();
    }
    function draw() {
      if (!ctx || !walls) return;
      const g = game, k = ctx, ph = g.phase;
      const cleared = ph === 'levelClear' && g.freeze < 180 && ((g.freeze >> 4) & 1);
      k.setTransform(1, 0, 0, 1, 0, 0); k.drawImage(cleared ? wallsFlash : walls, 0, 0);
      k.setTransform(scale, 0, 0, scale, 0, 0);
      // dots and energizers
      k.fillStyle = pal.pellet;
      const blinkOff = ph === 'play' && (g.frame >> 3) & 1;
      for (let i = 3 * 28; i < 34 * 28; i++) {
        const d = g.dots[i]; if (!d) continue;
        const x = (i % 28) * 8 + 3.5, y = ((i / 28) | 0) * 8 + 4.5;
        if (d === 1) k.fillRect(x - 1, y - 1, 2, 2);
        else if (!blinkOff) { k.beginPath(); k.arc(x, y, 3.3, 0, 2 * Math.PI); k.fill(); }
      }
      if (g.fruit) fruitIcon(k, g.spec.bonusName, 111.5, 164.5);
      if (st.layers.view) drawView();
      if (st.layers.targets && ph === 'play') drawTargets();
      // ghosts
      const dying = ph === 'dying' ? 210 - g.freeze : -1;
      const hideGhosts = (dying >= 60 && g.opts.pauses) || ph === 'levelClear' || g.over;
      if (!hideGhosts) for (const h of g.ghosts) {
        const x = h.x + 0.5, y = h.y + 0.5, eaten = h.state === AR.EYES || h.state === AR.ENTERING;
        if (ph === 'ghostEaten' && g.lastEaten && g.lastEaten.ghost === h.id) continue;
        let look = 'normal', body = GCOL[h.id];
        if (eaten) look = 'eyes';
        else if (h.frightened) { const flashOn = g.flashing && ((g.frightTimer / 14) | 0) % 2 === 0; look = flashOn ? 'flash' : 'blue'; body = flashOn ? PALE : BLUE; }
        ghostShape(k, x, y, body, h.state === AR.ACTIVE ? h.turnDir : h.dir, look);
      }
      // Pac-Man
      const p = g.pac;
      const moved = Math.abs(p.x - st.lastPac[0]) + Math.abs(p.y - st.lastPac[1]);
      if (moved < 20) st.pacPhase += moved;
      st.lastPac = [p.x, p.y];
      if (ph === 'ghostEaten' && g.lastEaten) {
        k.fillStyle = '#5ad7ef'; k.font = `800 7px ${pal.font}`; k.textAlign = 'center'; k.textBaseline = 'middle';
        k.fillText(String(g.lastEaten.points), g.lastEaten.x + 0.5, g.lastEaten.y + 0.5);
      } else if (!(g.over && g.lives <= 0 && ph !== 'dying')) {
        let mouth = ph === 'ready' ? 0 : 0.08 + 0.42 * Math.abs(Math.sin(st.pacPhase * Math.PI / 7));
        if (dying >= 60) mouth = Math.min(Math.PI, 0.35 + ((dying - 60) / 150) * Math.PI);
        if (mouth < Math.PI - 0.05) pacShape(k, p.x + 0.5, p.y + 0.5, 6.5, p.dir, mouth, pal.pac);
      }
      // text on the maze
      k.textAlign = 'center'; k.textBaseline = 'middle';
      if (ph === 'ready') { k.fillStyle = '#ffd23f'; k.font = `800 italic 8px ${pal.font}`; k.fillText('Ready!', 111.5, 20 * 8 + 4.5); }
      if (g.over) { k.fillStyle = '#ff5a5f'; k.font = `800 8px ${pal.font}`; k.fillText('Game over', 111.5, 20 * 8 + 4.5); }
      // HUD: score and level on top, lives and fruit at the bottom
      k.textBaseline = 'alphabetic'; k.textAlign = 'left'; k.fillStyle = '#aab6d3'; k.font = `700 6px ${pal.font}`;
      k.fillText('SCORE', 8, 9); k.textAlign = 'right'; k.fillText('LEVEL', 216, 9);
      k.fillStyle = '#e8ecf6'; k.font = `800 8px ${pal.font}`;
      k.textAlign = 'left'; k.fillText(fmt(g.score), 8, 19); k.textAlign = 'right'; k.fillText(String(g.level), 216, 19);
      for (let i = 0; i < Math.min(5, g.lives - (g.over ? 0 : 1)); i++) pacShape(k, 16 + i * 13, 280, 4.6, AR.LEFT, 0.55, pal.pac);
      for (let j = 0; j < 7 && g.level - j >= 1; j++) fruitIcon(k, AR.levelSpec(g.level - j).bonusName, 208 - j * 13, 280, 0.8);
    }
    function drawView() {
      const k = ctx, p = game.pac, x = ((p.x >> 3) - AR.WIN / 2 + 0.5) * 8, y = ((p.y >> 3) - AR.WIN / 2 + 0.5) * 8, s = AR.WIN * 8;
      k.save(); k.fillStyle = 'rgba(122,167,255,.08)'; k.fillRect(x, y, s, s);
      k.strokeStyle = '#7aa7ff'; k.lineWidth = 1; k.setLineDash([3, 2]); k.strokeRect(x, y, s, s); k.setLineDash([]);
      k.fillStyle = '#7aa7ff'; k.font = `700 5.5px ${pal.font}`; k.textBaseline = 'bottom';
      const label = 'the brain’s 15 × 15 view', lw = k.measureText(label).width;
      k.textAlign = 'left'; k.fillText(label, clamp(x + 2, 2, 222 - lw), y > 30 ? y - 1.5 : y + s + 7.5);
      k.restore();
    }
    function drawTargets() {
      const k = ctx, g = game, p = g.pac, px = p.x >> 3, py = p.y >> 3;
      const center = (t) => [t[0] * 8 + 3.5, t[1] * 8 + 4.5];
      k.save(); k.lineWidth = 1;
      for (const h of g.ghosts) {
        if ((h.state !== AR.ACTIVE && h.state !== AR.EYES) || h.frightened) continue;
        const t = g.targetOf(h); if (!t) continue;
        const col = GCOL[h.id], chase = !g.scatter || (h.id === AR.BLINKY && g.elroy() > 0);
        let [tx, ty] = center(t);
        const cx = clamp(tx, 3, 221), cy = clamp(ty, 3, 285), off = cx !== tx || cy !== ty;
        if (h.id === AR.INKY && chase && h.state === AR.ACTIVE) { // Blinky -> pivot, doubled
          const [ox, oy] = AR.ahead(p.dir, 2), b = g.ghosts[AR.BLINKY], pv = center([px + ox, py + oy]);
          k.strokeStyle = GCOL[AR.BLINKY]; k.globalAlpha = 0.6; k.beginPath(); k.moveTo((b.x >> 3) * 8 + 3.5, (b.y >> 3) * 8 + 4.5); k.lineTo(pv[0], pv[1]); k.stroke();
          k.fillStyle = col; k.globalAlpha = 0.9; k.beginPath(); k.arc(pv[0], pv[1], 1.6, 0, 2 * Math.PI); k.fill();
        }
        if (h.id === AR.CLYDE && chase && h.state === AR.ACTIVE) {
          k.strokeStyle = col; k.globalAlpha = 0.28; k.beginPath(); k.arc(px * 8 + 3.5, py * 8 + 4.5, 64, 0, 2 * Math.PI); k.stroke();
        }
        k.globalAlpha = 0.8; k.strokeStyle = col; k.setLineDash([2.5, 2]);
        k.beginPath(); k.moveTo(h.x + 0.5, h.y + 0.5); k.lineTo(cx, cy); k.stroke(); k.setLineDash([]);
        k.globalAlpha = 1; k.lineWidth = 1.2;
        k.strokeRect(cx - 3.5, cy - 3.5, 7, 7);
        k.beginPath(); k.moveTo(cx - 2, cy - 2); k.lineTo(cx + 2, cy + 2); k.moveTo(cx + 2, cy - 2); k.lineTo(cx - 2, cy + 2); k.stroke();
        if (off) { // target beyond the screen: an arrow at the edge
          const a = Math.atan2(ty - cy, tx - cx); k.fillStyle = col; k.beginPath();
          k.moveTo(cx + Math.cos(a) * 7, cy + Math.sin(a) * 7); k.lineTo(cx + Math.cos(a + 2.5) * 4, cy + Math.sin(a + 2.5) * 4); k.lineTo(cx + Math.cos(a - 2.5) * 4, cy + Math.sin(a - 2.5) * 4); k.fill();
        }
        k.lineWidth = 1;
      }
      k.restore();
    }

    // ------------------------------------------------------------ side panel
    function modeLine() {
      const g = game;
      if (g.over) return `Game over: ${fmt(g.score)} points`;
      if (g.phase === 'ready') return 'Get ready…';
      if (g.phase === 'levelClear') return `Level ${g.level} cleared`;
      if (g.frightTimer > 0) return `Frightened, ${secs(g.frightTimer)} left${g.flashing ? ' (flashing)' : ''}`;
      const w = g.scatter ? 'Scatter' : 'Chase';
      return g.modeTimer === Infinity ? `${w} until the level ends` : `${w}, ${secs(g.modeTimer)} left`;
    }
    function ghostNow(h) {
      const g = game, sp = g.spec;
      if (h.state === AR.HOUSE) {
        if (g.globalActive) {
          const need = [0, 7, 17, 32][h.id];
          return `In the house. Since the lost life it waits for the shared dot count to reach ${need} (now ${g.globalCounter}), or for ${secs(sp.dotTimer - g.dotTimer)} with no dot eaten.`;
        }
        if (g._preferredWaiting() !== h) return 'In the house, waiting for its turn to count dots.';
        return `In the house, counting dots: ${h.dotCounter} of ${sp.dotLimits[h.id]}. Or it leaves after ${secs(sp.dotTimer - g.dotTimer)} more with no dot eaten.`;
      }
      if (h.state === AR.LEAVING) return h.frightened ? 'Leaving the house, frightened.' : 'Leaving the house.';
      if (h.state === AR.EYES || h.state === AR.ENTERING) return 'Eaten. Its eyes are hurrying back to the house.';
      if (h.frightened) return g.flashing ? 'Frightened and flashing: about to recover.' : 'Frightened: slower, turning at random.';
      const t = g.targetOf(h), e = g.elroy();
      if (h.id === AR.BLINKY && e > 0) return `Cruise Elroy ${e}: faster, and chasing even in scatter waves. Target (${t[0]}, ${t[1]}).`;
      if (g.scatter) return `Scatter: heading for the ${CORNER[h.id]} corner.`;
      if (h.id === AR.CLYDE) {
        const d = Math.hypot((h.x >> 3) - (g.pac.x >> 3), (h.y >> 3) - (g.pac.y >> 3));
        return d >= 8 ? `Chase: ${d.toFixed(1)} tiles from Pac-Man, so going after him.` : `Chase, but only ${d.toFixed(1)} tiles from Pac-Man, so backing off to his corner.`;
      }
      const off = t[0] < 0 || t[0] > 27 || t[1] < 0 || t[1] > 35 ? ', off the screen' : '';
      return `Chase: target tile (${t[0]}, ${t[1]})${off}.`;
    }
    function renderStatus() {
      const g = game, s = `<span>${modeLine()}</span><span>${g.elroy() ? `Blinky: Cruise Elroy ${g.elroy()}` : `Level ${g.level}`}</span>`;
      const box = $('#four-status'); if (box._s !== s) { box._s = s; box.innerHTML = s; }
    }
    function renderDom(force) {
      if (!st.shown) return;
      const g = game;
      renderStatus();
      const dl = $('#four-stats');
      const rows = [
        ['Score', fmt(g.score)], ['Level', String(g.level)], ['Lives left', String(g.lives)], ['Dots left', `${g.dotsLeft} of 244`],
        [st.who === 'brain' ? 'Brain decisions this game' : 'Playing', st.who === 'brain' ? fmt(st.gameDecisions) : 'you'],
        ['Ghosts eaten this game', String(st.ghostsEaten)],
        ['Fruit', g.fruit ? `${FRUIT_NAMES[g.spec.bonusName]}, ${fmt(g.fruit)} points` : `${FRUIT_NAMES[g.spec.bonusName]} after ${g.dotsEaten < 70 ? 70 : 170} dots`],
        ['Game', `seed ${st.seed}`],
      ];
      if (!dl._dd) { dl.textContent = ''; dl._dd = rows.map(([k]) => { const d = el('div', '', dl); el('dt', '', d, k); return el('dd', '', d); }); dl._dt = $$('dt', dl); }
      rows.forEach(([k, v], i) => { if (dl._dt[i].textContent !== k) dl._dt[i].textContent = k; if (dl._dd[i].textContent !== v) dl._dd[i].textContent = v; });
      qbars.render(st.q || [0, 0, 0, 0], { hi: st.qAct, cls: 'live' });
      $('#four-q-when').textContent = st.who === 'brain' ? 'at its latest decision' : 'right here, if it were playing';
      renderWaves(force);
      const cards = $('#four-ghosts');
      if (!cards._now) {
        cards.textContent = '';
        cards._now = g.ghosts.map((h) => {
          const c = el('div', 'gcard', cards); c.style.setProperty('--c', GCOL[h.id]);
          el('div', 'sw', c);
          html(el('h4', '', c), `${GHOSTS[h.id].name}<small>${GHOSTS[h.id].nick}</small>`);
          el('p', 'rule', c, GHOSTS[h.id].rule);
          return el('p', 'now', c);
        });
      }
      g.ghosts.forEach((h, i) => { const s = ghostNow(h); if (cards._now[i].textContent !== s) cards._now[i].textContent = s; });
    }
    function renderWaves(force) {
      const root = $('#four-waves'), g = game, modes = g.spec.modes;
      const key = `${g.level > 4 ? 5 : g.level > 1 ? 2 : 1}`;
      if (root._key !== key || force) {
        root._key = key; root.textContent = '';
        root._w = modes.concat([Infinity]).map((f, i) => {
          const w = el('div', 'wave' + (i % 2 ? ' chase' : ''), root); w._fill = el('i', 'fill', w);
          const s = f === Infinity ? 'forever' : f < 60 ? '1/60 s' : `${fmt(f / 60)} s`;
          el('span', '', w, `${i % 2 ? 'Chase' : 'Scatter'} ${s}`);
          return w;
        });
      }
      root._w.forEach((w, i) => {
        const now = i === g.modeIndex, done = i < g.modeIndex;
        w.className = 'wave' + (i % 2 ? ' chase' : '') + (now ? ' now' : '') + (done ? ' done' : '');
        const f = now && modes[i] ? 1 - g.modeTimer / modes[i] : done ? 1 : 0;
        w._fill.style.width = `${Math.round(clamp(f, 0, 1) * 100)}%`;
      });
    }

    // ------------------------------------------------------------ training record
    function drawChart() {
      const c = $('#four-chart');
      if (!c || panel().hidden) return;
      const { ctx: k, w, h } = fitCanvas(c);
      const ink = cssVar('--ink'), ink3 = cssVar('--ink-3'), live = cssVar('--live');
      const ex = CURVE.exams || [], pr = CURVE.practice || [], base = CURVE.random || 0;
      k.font = '11px ' + cssVar('--font');
      if (!ex.length) { k.fillStyle = ink3; k.textAlign = 'center'; k.fillText('No training record in this build.', w / 2, h / 2); return; }
      const L = 48, R = 12, T = 12, B = 26;
      const x1 = Math.max(...ex.map((e) => e[0]), ...pr.map((p) => p[0]));
      let hi = Math.max(...ex.map((e) => e[1]), ...pr.map((p) => p[1]), base) * 1.08;
      const X = (s) => L + (s / x1) * (w - L - R), Y = (v) => T + ((hi - v) / hi) * (h - T - B);
      k.lineWidth = 1; k.fillStyle = ink3; k.strokeStyle = ink3; k.textAlign = 'right'; k.textBaseline = 'middle';
      for (const t of niceTicks(0, hi, 4)) { k.globalAlpha = 0.3; k.beginPath(); k.moveTo(L, Y(t)); k.lineTo(w - R, Y(t)); k.stroke(); k.globalAlpha = 1; k.fillText(t >= 1000 ? `${+(t / 1000).toFixed(1)}k` : String(t), L - 6, Y(t)); }
      const endLabel = `${x1 >= 1e6 ? +(x1 / 1e6).toFixed(1) + 'M' : fmt(x1)} decisions`, endW = k.measureText(endLabel).width;
      k.textAlign = 'center'; k.textBaseline = 'top';
      for (const t of niceTicks(0, x1, 4)) { // skip ticks whose label would run into the end label
        const txt = t >= 1e6 ? `${+(t / 1e6).toFixed(1)}M` : t >= 1000 ? `${+(t / 1000).toFixed(0)}k` : String(t);
        if (X(t) + k.measureText(txt).width / 2 < w - R - endW - 8) k.fillText(txt, X(t), h - B + 6);
      }
      k.textAlign = 'right'; k.fillText(endLabel, w - R, h - B + 6);
      k.setLineDash([5, 4]); k.beginPath(); k.moveTo(L, Y(base)); k.lineTo(w - R, Y(base)); k.stroke(); k.setLineDash([]);
      k.fillStyle = ink; k.globalAlpha = 0.25;
      for (const [s, v] of pr) { k.beginPath(); k.arc(X(s), Y(v), 1.6, 0, 6.3); k.fill(); }
      k.globalAlpha = 1;
      k.strokeStyle = live; k.fillStyle = live; k.lineWidth = 2.5; k.beginPath();
      ex.forEach(([s, v], i) => (i ? k.lineTo(X(s), Y(v)) : k.moveTo(X(s), Y(v)))); k.stroke();
      if (ex.length < 80) for (const [s, v] of ex) { k.beginPath(); k.arc(X(s), Y(v), 2.4, 0, 6.3); k.fill(); }
      // Labels: the exam tag, the first-cleared line, and the shipped checkpoint, which dodges the other two.
      const last = ex[ex.length - 1], placed = [];
      const label = (text, x, y, align, base, color, bold) => {
        k.font = (bold ? '600 ' : '') + '11px ' + cssVar('--font');
        const tw = k.measureText(text).width, x0 = align === 'right' ? x - tw : x, y0 = base === 'top' ? y : y - 12;
        return { text, x, y, align, base, color, bold, box: [x0 - 2, y0 - 1, x0 + tw + 2, y0 + 13] };
      };
      const overlap = (a, b) => Math.max(0, Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0])) * Math.max(0, Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]));
      const outside = (a) => a.box[0] < L || a.box[2] > w || a.box[1] < 0 || a.box[3] > h - B + 2;
      const cost = (a) => (outside(a) ? 1e9 : 0) + placed.reduce((c, b) => c + overlap(a, b), 0);
      const best = (tries) => tries.reduce((m, a) => (cost(a) < cost(m) ? a : m)); // first candidate wins ties
      const put = (a) => {
        placed.push(a); k.fillStyle = a.color; k.font = (a.bold ? '600 ' : '') + '11px ' + cssVar('--font');
        k.textAlign = a.align; k.textBaseline = a.base; haloText(k, a.text, a.x, a.y);
      };
      const first = ex.find((e) => e[2] > 1.05);
      if (first) {
        k.strokeStyle = cssVar('--frozen'); k.lineWidth = 1; k.setLineDash([3, 3]);
        k.beginPath(); k.moveTo(X(first[0]), T); k.lineTo(X(first[0]), h - B); k.stroke(); k.setLineDash([]);
        const right = X(first[0]) > w * 0.6;
        put(label('first level cleared in an exam', X(first[0]) + (right ? -4 : 4), T, right ? 'right' : 'left', 'top', cssVar('--frozen'), false));
      }
      put(label('random play', w - R - 4, Y(base) - 3, 'right', 'bottom', ink3, false));
      put(label('exam', Math.min(X(last[0]), w - R) - 4, Y(last[1]) - 4, 'right', 'bottom', live, true));
      const pick = META.steps && ex.find((e) => e[0] === META.steps);
      if (pick) { // the checkpoint this page plays
        k.strokeStyle = live; k.lineWidth = 2; k.fillStyle = cssVar('--sheet-2');
        k.beginPath(); k.arc(X(pick[0]), Y(pick[1]), 5, 0, 6.3); k.fill(); k.stroke();
        const px = X(pick[0]), py = Y(pick[1]), txt = 'the brain on this page', tries = [];
        for (const dy of [-6, 7, 20, -19]) for (const side of px > w * 0.6 ? [-1, 1] : [1, -1])
          tries.push(label(txt, px + side * 8, py + dy, side < 0 ? 'right' : 'left', dy < 0 ? 'bottom' : 'top', live, true));
        put(best(tries));
      }
    }
    function renderRecord() {
      const s = CURVE.summary || {};
      const weights = META.weights || 0;
      const hrs = (x) => `${x.toFixed(1)} hours`;
      const hours = META.hours ? hrs(META.hours) : s.hours ? hrs(s.hours) : 'several hours';
      $('#four-summary').textContent = `${fmt(weights)} weights (the Arcade station’s brain has 11,908), trained on ${fmt(META.steps || 0)} decisions over ${hours} on 4 CPU cores`;
      const dl = $('#four-train-stats'); dl.textContent = '';
      const f = s.final, millions = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : fmt(n));
      const rows = [['Decisions trained', fmt(META.steps || 0)], ['Training time', hours]];
      if (f) rows.push([`Final check (${f.games} new games)`, `${fmt(f.mean_score)} points`], ['Levels cleared per game', (f.mean_level - 1).toFixed(2)],
        [`Best game in the check (level ${f.max_level})`, `${fmt(f.max_score)} points`], ['Ghosts eaten per game', f.ghosts_per_game.toFixed(1)]);
      const partRun = s.decisions > (META.steps || 0); // shipped brain is mid-run: the run row replaces the weights row (keeps the grid even)
      if (f && !partRun) rows.push(['Network weights', fmt(weights)]);
      else rows.push(['Best exam (10 games)', META.exam ? `${fmt(META.exam.mean_score)} points` : '—'], ['Network weights', fmt(weights)]);
      if (partRun) rows.push([`Whole run${s.hours ? ` (${hrs(s.hours)})` : ''}`, `${millions(s.decisions)} decisions`]);
      rows.push(['Random play', `${fmt(CURVE.random || 0)} points`]);
      for (const [a, b] of rows) { const d = el('div', '', dl); el('dt', '', d, a); el('dd', '', d, b); }
    }
    function renderDocs() {
      const o = AR.OBS;
      html($('#four-doc-brain'), `
        <h4>What the brain sees: ${fmt(o.size)} numbers per decision</h4>
        <ul>
          <li><b>A 15 × 15 window centered on Pac-Man</b> (${fmt(o.window)} numbers): 7 layers marking walls, dots, energizers, ghosts, the square each ghost is about to enter, frightened ghosts, and fruit. Turn on <b>What the brain sees</b> above the screen to see the window.</li>
          <li><b>A compass for each move</b> (${o.paths} numbers): if Pac-Man goes this way, how many steps along the maze to the nearest dot, energizer and fruit, and to each ghost, dangerous or frightened.</li>
          <li><b>${o.global} facts about the game</b>: which way Pac-Man faces, the frightened and wave timers, each ghost’s state and where it is relative to Pac-Man, Cruise Elroy, the level, dots left, lives.</li>
          <li><b>A coarse map of the remaining dots</b> (${o.dotmap} blocks of 4 × 4 tiles), so it can find the last few.</li>
        </ul>
        <h4>How it decides and learns</h4>
        <ul>
          <li>It picks a move every time Pac-Man enters a new tile: the move whose bar is tallest. The stick is held that way until the next tile, so turns are taken as early as the rules allow.</li>
          <li>Its network is ${META.sizes.join(' → ')}, ${fmt(META.weights || 0)} weights, trained in PyTorch on the same JavaScript engine this page runs. The page runs it with the workshop’s own hand-written network code, and a test checks that both give identical Q-values and play identical games.</li>
          <li>Rewards: the square root of the arcade points, divided by 10 (a dot 0.32, an energizer 0.71, ghosts 1.4 to 4, fruit 1 to 7.1), and −2 for losing a life. A lost life also ends the target, with no future added, like the game-over case on the Bench.</li>
          <li>The same Double DQN target as the Bench, looking 3 moves ahead instead of 1: r₁ + γr₂ + γ²r₃ + γ³·v, with γ = 0.99 per move. There is a replay memory of about a million moves and a frozen copy refreshed every 2,000 updates.</li>
          <li>Step size: the weights moved with a learning rate of 0.0001 for the first 57.5 million decisions, then 0.00003 for 7.5 million more. The smaller steps let the weights settle instead of jittering, and that short fine-tune alone added about 14%.</li>
          <li>Exploration: 16 games trained at once, each with its own random-move chance from 40% down to under 0.1%.</li>
          <li>Sticky stick, in training and exams: each frame there is a 25% chance the stick stays where it was, so a new direction sometimes lands a frame or two late (the recipe from Machado et al., 2018). The ghosts are deterministic, so without this every game from the start would be nearly the same, and a memorized route could pass for skill.</li>
        </ul>`);
      html($('#four-doc-rules'), `
        <p>Brandon chose <b>documents only</b>, so there is no ROM data anywhere: no graphics, sounds or code bytes from the arcade machine. The rules come from <b>The Pac-Man Dossier</b> by Jamey Pittman, which the author checked against the ROM disassembly. A few coordinates were cross-checked against pacman.c by Andre Weissflog. The characters, fruit icons and wall style are this workshop’s own drawings. 51 rule-by-rule tests keep the engine honest.</p>
        <ul>
          <li><b>Maze</b>: 28 × 31 tiles, 240 dots and 4 energizers, the side tunnel, the house with its door, and the two red zones above the house and Pac-Man’s start where ghosts may not turn up. <span class="src">Dossier ch. 2–3</span></li>
          <li><b>Speed</b>: level 1 Pac-Man 80%, ghosts 75%, tunnel 40%, up to 100%/95% from level 5. Pac-Man stops 1 frame per dot (3 per energizer), which is why ghosts catch up while he eats. <span class="src">Table A.1</span></li>
          <li><b>Cornering</b>: Pac-Man can turn up to 4 pixels before a corner and cut it diagonally. Ghosts can’t. <span class="src">ch. 2</span></li>
          <li><b>Ghost pathfinding</b>: decide one tile ahead, never reverse by choice, take the exit closest in a straight line to the target, and break ties up, left, down, right. <span class="src">ch. 3</span></li>
          <li><b>Waves</b>: scatter 7 s, chase 20 s, scatter 7, chase 20, scatter 5, chase 20, scatter 5, then chase for good on level 1. Every switch reverses the ghosts. <span class="src">ch. 2</span></li>
          <li><b>Frightened</b>: 6 s on level 1, shrinking to none by level 19. Ghosts worth 200, 400, 800, 1600. <span class="src">Table A.1</span></li>
          <li><b>Ghost house</b>: dot counters (Inky after 30, Clyde after 60 more on level 1), a shared counter after a lost life (7, 17, 32), and a 4-second no-dot timer. <span class="src">ch. 2</span></li>
          <li><b>Cruise Elroy</b>: Blinky speeds up at 20 and 10 dots left on level 1 (more on later levels) and stops scattering. <span class="src">ch. 4</span></li>
          <li><b>Collisions</b> are by tile, checked once a frame, so Pac-Man and a ghost that swap tiles in the same frame pass right through each other, as in the arcade. <span class="src">ch. 3</span></li>
        </ul>
        <h4>Where the documents are silent</h4>
        <table><tr><th>Detail</th><th>What this version does</th></tr>
          <tr><td>Frame-by-frame speed patterns</td><td>exact average speeds, spread evenly</td></tr>
          <tr><td>Frightened ghosts’ random turns</td><td>a seeded random generator, reset every level and every life like the original (which reads its own ROM bytes instead)</td></tr>
          <tr><td>Speed of eyes and of ghosts in the house</td><td>1.5 and 0.5 pixels per frame</td></tr>
          <tr><td>Level 256, the split screen</td><td>not emulated: the game ends after level 255</td></tr>
        </table>`);
    }

    // ------------------------------------------------------------ controls
    function setWho(v) {
      st.who = v; setSeg($('#four-who'), v);
      $('#four-pad').hidden = v !== 'you';
      cv.style.touchAction = v === 'you' ? 'none' : '';
      $('#four-who-note').textContent = v === 'you'
        ? 'Arrow keys or WASD, swipe on the screen, or use the pad. The stick stays where you last pushed it, like a real joystick. Push early to cut corners.'
        : 'At every tile the brain scores the four moves and takes the tallest bar. It plays with the sticky stick it trained with: each frame there is a 25% chance a new direction waits one more frame. So no two games are the same.';
      st.deciding = false; st.joy = null;
      renderDom(true);
    }
    function setRunning(on) { st.running = on; $('#four-play').textContent = on ? 'Pause' : 'Play'; }
    function init() {
      cv = $('#four-canvas'); qbars = new QBars($('#four-q'));
      env = new AR.ArcadeEnv({ seed: st.seed, pauses: true, sticky: STICKY }); game = env.game; env.reset(st.seed);
      $('#four-play').addEventListener('click', () => setRunning(!st.running));
      $('#four-step').addEventListener('click', () => { setRunning(false); oneMove(); });
      $('#four-new').addEventListener('click', () => newGame());
      bindSeg($('#four-who'), setWho);
      bindSeg($('#four-speed'), (v) => { st.speed = v === 'max' ? 'max' : +v; });
      $$('#four-layers .chip').forEach((b) => b.addEventListener('click', () => {
        const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on));
        st.layers[b.dataset.layer] = on; draw();
      }));
      const KEYS = { ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1, ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3 };
      document.addEventListener('keydown', (e) => {
        if (panel().hidden || st.who !== 'you' || !(e.code in KEYS)) return;
        if (e.target.closest && e.target.closest('input, select, textarea')) return;
        e.preventDefault(); st.joy = KEYS[e.code];
      });
      $$('#four-pad button').forEach((b) => b.addEventListener('pointerdown', (e) => { e.preventDefault(); st.joy = +b.dataset.a; }));
      let sw = null;
      cv.addEventListener('pointerdown', (e) => { if (st.who === 'you') sw = [e.clientX, e.clientY]; });
      cv.addEventListener('pointermove', (e) => {
        if (!sw) return;
        const dx = e.clientX - sw[0], dy = e.clientY - sw[1];
        if (Math.max(Math.abs(dx), Math.abs(dy)) < 14) return;
        st.joy = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0);
        sw = [e.clientX, e.clientY];
      });
      for (const t of ['pointerup', 'pointercancel', 'pointerleave']) cv.addEventListener(t, () => { sw = null; });
      setWho('brain');
      renderDocs(); renderRecord();
    }
    function onShow() {
      if (!net) net = loadNet();
      st.shown = true;
      resize(); draw(); renderDom(true); drawChart();
    }
    function redraw() { if (!st.shown || panel().hidden) return; resize(); draw(); drawChart(); }
    // for tests: n whole brain decisions, synchronously
    function runDecisions(n) { if (!net) net = loadNet(); const was = st.who; if (was !== 'brain') setWho('brain'); for (let i = 0; i < n && !game.over; i++) { let g = 0; do advance(); while (st.deciding && ++g < 5000); } }
    return { init, tick, onShow, redraw, drawChart, newGame, setWho, runDecisions, advance, draw, get env() { return env; }, get game() { return game; }, get state() { return st; }, get net() { return net; } };
  })();

  // ===================================================================== TABS
  const TABS = { 'tab-bench': bench, 'tab-tank': tank, 'tab-arcade': arcade, 'tab-four': four };
  function selectTab(id) {
    $$('.tab').forEach((t) => {
      const on = t.id === id;
      t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      $('#' + t.getAttribute('aria-controls')).hidden = !on;
      if (on && t.parentElement.scrollWidth > t.parentElement.clientWidth) t.parentElement.scrollTo({ left: t.offsetLeft - 18, behavior: 'auto' });
    });
    TABS[id].onShow();
  }
  function initTabs() {
    const tabs = $$('.tab');
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => selectTab(t.id));
      t.addEventListener('keydown', (e) => {
        const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!d) return;
        const n = tabs[(i + d + tabs.length) % tabs.length]; n.focus(); selectTab(n.id);
      });
    });
  }

  function redrawCharts() { if (!$('#panel-bench').hidden) bench.render(); arcade.drawCurve(); four.redraw(); }

  function boot() {
    initTabs();
    arcade.init();
    bench.init();
    tank.init();
    four.init();
    requestAnimationFrame(() => requestAnimationFrame(() => $('.equation').classList.add('eq-drawn')));
    const loop = (ts) => { tank.tick(ts); arcade.tick(ts); four.tick(ts); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(redrawCharts, 120); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawCharts);
    new MutationObserver(redrawCharts).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawCharts);
    window.__workshop = { bench, tank, arcade, four, selectTab };
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
