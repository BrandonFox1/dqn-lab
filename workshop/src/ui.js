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

  // ===================================================================== TABS
  const TABS = { 'tab-bench': bench, 'tab-tank': tank, 'tab-arcade': arcade };
  function selectTab(id) {
    $$('.tab').forEach((t) => {
      const on = t.id === id;
      t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      $('#' + t.getAttribute('aria-controls')).hidden = !on;
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

  function redrawCharts() { if (!$('#panel-bench').hidden) bench.render(); arcade.drawCurve(); }

  function boot() {
    initTabs();
    arcade.init();
    bench.init();
    tank.init();
    requestAnimationFrame(() => requestAnimationFrame(() => $('.equation').classList.add('eq-drawn')));
    const loop = (ts) => { tank.tick(ts); arcade.tick(ts); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(redrawCharts, 120); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawCharts);
    new MutationObserver(redrawCharts).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawCharts);
    window.__workshop = { bench, tank, arcade, selectTab };
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
