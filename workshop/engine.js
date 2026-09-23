/* DQN Workshop engine: a tiny neural net, two environments and a DQN.
   Pure JS, no dependencies. Runs in the browser and in Node (for tests). */
const Engine = (() => {
  'use strict';

  // ------------------------------------------------------------------ RNG
  function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    const next = () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.int = (n) => Math.floor(next() * n);
    next.normal = () => {
      let u = 0;
      while (u === 0) u = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
    };
    next.pick = (arr) => arr[Math.floor(next() * arr.length)];
    return next;
  }

  const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right  (dy, dx)
  const ACTION_NAMES = ['up', 'down', 'left', 'right'];
  const OPPOSITE = [1, 0, 3, 2];

  function argmax(a) {
    let bi = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
    return bi;
  }

  // Huber (smooth L1) with delta = 1
  function huber(x) { const ax = Math.abs(x); return ax < 1 ? 0.5 * x * x : ax - 0.5; }
  function huberGrad(x) { return x > 1 ? 1 : x < -1 ? -1 : x; }

  // ------------------------------------------------------------------ MLP
  class MLP {
    constructor(sizes, seed = 1, outScale = 0.1) {
      this.sizes = sizes.slice();
      this.L = sizes.length - 1;
      this.W = []; this.b = [];
      const rng = makeRng(seed);
      for (let l = 0; l < this.L; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const W = new Float32Array(nin * nout);
        const std = Math.sqrt(2 / nin) * (l === this.L - 1 ? outScale : 1);
        for (let i = 0; i < W.length; i++) W[i] = rng.normal() * std;
        this.W.push(W);
        this.b.push(new Float32Array(nout));
      }
      this._alloc();
    }
    _alloc() {
      this.acts = this.sizes.map((n) => new Float32Array(n));
      this.deltas = this.sizes.map((n) => new Float32Array(n));
      this.gW = this.W.map((w) => new Float32Array(w.length));
      this.gb = this.b.map((b) => new Float32Array(b.length));
      this.m = null; this.v = null; this.t = 0;
    }
    get paramCount() { let n = 0; for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length; return n; }

    // Forward one input. Activations are cached for a following backward().
    forward(x) {
      const a0 = this.acts[0];
      for (let i = 0; i < a0.length; i++) a0[i] = x[i];
      for (let l = 0; l < this.L; l++) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const W = this.W[l], b = this.b[l], ain = this.acts[l], aout = this.acts[l + 1];
        const last = l === this.L - 1;
        // Collect non-zero inputs once (inputs are sparse one-hots; hidden ReLUs are ~half zero).
        let nz = 0; const idx = this._nz || (this._nz = new Int32Array(Math.max(...this.sizes)));
        for (let i = 0; i < nin; i++) if (ain[i] !== 0) idx[nz++] = i;
        for (let j = 0; j < nout; j++) {
          let s = b[j]; const off = j * nin;
          for (let k = 0; k < nz; k++) { const i = idx[k]; s += W[off + i] * ain[i]; }
          aout[j] = last ? s : (s > 0 ? s : 0);
        }
      }
      return Float32Array.from(this.acts[this.L]);
    }

    // Accumulate gradients of sum_j dOut[j]*out_j w.r.t. every weight (backprop).
    backward(dOut) {
      const dl = this.deltas[this.L];
      for (let j = 0; j < dl.length; j++) dl[j] = dOut[j];
      for (let l = this.L - 1; l >= 0; l--) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const W = this.W[l], gW = this.gW[l], gb = this.gb[l], ain = this.acts[l];
        const dout = this.deltas[l + 1], din = this.deltas[l];
        if (l > 0) din.fill(0);
        for (let j = 0; j < nout; j++) {
          const dj = dout[j];
          if (dj === 0) continue;
          gb[j] += dj;
          const off = j * nin;
          if (l > 0) {
            for (let i = 0; i < nin; i++) { gW[off + i] += dj * ain[i]; din[i] += W[off + i] * dj; }
          } else {
            for (let i = 0; i < nin; i++) { const a = ain[i]; if (a !== 0) gW[off + i] += dj * a; }
          }
        }
        if (l > 0) for (let i = 0; i < nin; i++) if (ain[i] <= 0) din[i] = 0;
      }
    }

    gradNorm() {
      let s = 0;
      for (let l = 0; l < this.L; l++) {
        for (const g of this.gW[l]) s += g * g;
        for (const g of this.gb[l]) s += g * g;
      }
      return Math.sqrt(s);
    }
    zeroGrad() { for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); } }

    // weight <- weight - lr * gradient   (optionally clipped by global norm)
    sgdStep(lr, clip = 0) {
      const norm = this.gradNorm();
      const sc = clip > 0 && norm > clip ? clip / norm : 1;
      for (let l = 0; l < this.L; l++) {
        const W = this.W[l], gW = this.gW[l], b = this.b[l], gb = this.gb[l];
        for (let i = 0; i < W.length; i++) W[i] -= lr * sc * gW[i];
        for (let i = 0; i < b.length; i++) b[i] -= lr * sc * gb[i];
      }
      this.zeroGrad();
      return norm;
    }

    // Adam: per-weight step sizes from running averages of the gradient.
    adamStep(lr, clip = 10, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      const norm = this.gradNorm();
      const sc = clip > 0 && norm > clip ? clip / norm : 1;
      if (!this.m) {
        this.m = []; this.v = [];
        for (let l = 0; l < this.L; l++) {
          this.m.push([new Float32Array(this.W[l].length), new Float32Array(this.b[l].length)]);
          this.v.push([new Float32Array(this.W[l].length), new Float32Array(this.b[l].length)]);
        }
      }
      this.t++;
      const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
      for (let l = 0; l < this.L; l++) {
        const P = [this.W[l], this.b[l]], G = [this.gW[l], this.gb[l]];
        for (let k = 0; k < 2; k++) {
          const p = P[k], g = G[k], m = this.m[l][k], v = this.v[l][k];
          for (let i = 0; i < p.length; i++) {
            const gi = g[i] * sc;
            m[i] = b1 * m[i] + (1 - b1) * gi;
            v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
          }
        }
      }
      this.zeroGrad();
      return norm;
    }

    clone() {
      const c = Object.create(MLP.prototype);
      c.sizes = this.sizes.slice(); c.L = this.L;
      c.W = this.W.map((w) => Float32Array.from(w));
      c.b = this.b.map((b) => Float32Array.from(b));
      c._alloc();
      return c;
    }
    copyFrom(o) {
      for (let l = 0; l < this.L; l++) { this.W[l].set(o.W[l]); this.b[l].set(o.b[l]); }
    }
    flat() {
      const out = new Float32Array(this.paramCount); let k = 0;
      for (let l = 0; l < this.L; l++) { out.set(this.W[l], k); k += this.W[l].length; out.set(this.b[l], k); k += this.b[l].length; }
      return out;
    }
    loadFlat(f) {
      if (f.length !== this.paramCount) throw new Error(`weight count ${f.length} != ${this.paramCount}`);
      let k = 0;
      for (let l = 0; l < this.L; l++) {
        this.W[l].set(f.subarray(k, k + this.W[l].length)); k += this.W[l].length;
        this.b[l].set(f.subarray(k, k + this.b[l].length)); k += this.b[l].length;
      }
    }
    toBase64() { return bytesToB64(new Uint8Array(this.flat().buffer)); }
    loadBase64(s) { const bytes = b64ToBytes(s); this.loadFlat(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)); }
  }

  function bytesToB64(bytes) {
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function b64ToBytes(s) {
    if (typeof Buffer !== 'undefined') { const b = Buffer.from(s, 'base64'); const out = new Uint8Array(b.length); out.set(b); return out; }
    const bin = atob(s); const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // ------------------------------------------------------ mini Pac-Man env
  const MINI_MAZE = [
    '#########',
    '#...G...#',
    '#.#.#.#.#',
    '#.......#',
    '#.#.#.#.#',
    '#...P...#',
    '#########',
  ];
  const DEFAULT_REWARDS = { pellet: 10, step: -1, death: -300, clear: 200 };

  class MiniPac {
    constructor(opts = {}) {
      const rows = opts.maze || MINI_MAZE;
      this.H = rows.length; this.W = rows[0].length;
      this.wall = new Uint8Array(this.W * this.H);
      this.cells = []; // grid index of each walkable cell
      this.cellOf = new Int32Array(this.W * this.H).fill(-1);
      const pel = [];
      for (let y = 0; y < this.H; y++) for (let x = 0; x < this.W; x++) {
        const ch = rows[y][x], g = y * this.W + x;
        if (ch === '#') { this.wall[g] = 1; continue; }
        this.cellOf[g] = this.cells.length; this.cells.push(g);
        pel.push(ch === '.' ? 1 : 0);
        if (ch === 'P') this.playerStart = this.cells.length - 1;
        if (ch === 'G') this.ghostStart = this.cells.length - 1;
      }
      this.N = this.cells.length;
      this.pelletStart = Uint8Array.from(pel);
      this.nbr = [];
      for (let c = 0; c < this.N; c++) {
        const g = this.cells[c], y = Math.floor(g / this.W), x = g % this.W;
        this.nbr.push(ACTIONS.map(([dy, dx]) => this.cellOf[(y + dy) * this.W + (x + dx)]));
      }
      this.dist = [];
      for (let s = 0; s < this.N; s++) {
        const d = new Int32Array(this.N).fill(1e6); d[s] = 0; const q = [s];
        for (let h = 0; h < q.length; h++) { const c = q[h]; for (const n of this.nbr[c]) if (n >= 0 && d[n] > d[c] + 1) { d[n] = d[c] + 1; q.push(n); } }
        this.dist.push(d);
      }
      this.rewards = Object.assign({}, DEFAULT_REWARDS, opts.rewards || {});
      this.maxSteps = opts.maxSteps ?? 150;
      this.aggression = opts.aggression ?? 0.75;
      this.obsSize = 4 * this.N;
      this.rng = makeRng(opts.seed ?? 1);
      this.reset();
    }
    xy(c) { const g = this.cells[c]; return [g % this.W, Math.floor(g / this.W)]; }
    reset(seed) {
      if (seed != null) this.rng = makeRng(seed);
      this.pellets = Uint8Array.from(this.pelletStart);
      this.player = this.playerStart; this.ghost = this.ghostStart; this.ghostPrev = this.ghostStart;
      this.ghostDir = -1; this.steps = 0; this.score = 0;
      this.done = false; this.won = false; this.dead = false; this.truncated = false;
      this.lastAction = 3;
      return this.obs();
    }
    get pelletsLeft() { let n = 0; for (const p of this.pellets) n += p; return n; }
    getState() {
      return { pellets: Uint8Array.from(this.pellets), player: this.player, ghost: this.ghost, ghostPrev: this.ghostPrev,
        ghostDir: this.ghostDir, steps: this.steps, score: this.score, done: this.done, won: this.won, dead: this.dead, lastAction: this.lastAction };
    }
    setState(s) {
      this.pellets = Uint8Array.from(s.pellets); this.player = s.player; this.ghost = s.ghost;
      this.ghostPrev = s.ghostPrev ?? s.ghost; this.ghostDir = s.ghostDir ?? -1; this.steps = s.steps ?? 0;
      this.score = s.score ?? 0; this.done = !!s.done; this.won = !!s.won; this.dead = !!s.dead; this.truncated = false;
      this.lastAction = s.lastAction ?? 3;
    }
    // Observation: 4 blocks of N (one entry per walkable square): pellet, Pac-Man, ghost, ghost's previous square.
    encode(s = this) {
      const N = this.N, o = new Float32Array(4 * N);
      for (let c = 0; c < N; c++) o[c] = s.pellets[c];
      o[N + s.player] = 1; o[2 * N + s.ghost] = 1; o[3 * N + (s.ghostPrev ?? s.ghost)] = 1;
      return o;
    }
    obs() { return this.encode(this); }

    _ghostMove(deterministic) {
      let legal = [];
      for (let a = 0; a < 4; a++) if (this.nbr[this.ghost][a] >= 0) legal.push(a);
      if (this.ghostDir >= 0 && legal.length > 1) {
        const nr = legal.filter((a) => a !== OPPOSITE[this.ghostDir]);
        if (nr.length) legal = nr;
      }
      let a;
      if (deterministic || this.rng() < this.aggression) {
        const d = this.dist[this.player];
        let best = Infinity, choices = [];
        for (const m of legal) { const v = d[this.nbr[this.ghost][m]]; if (v < best) { best = v; choices = [m]; } else if (v === best) choices.push(m); }
        a = deterministic ? choices[0] : this.rng.pick(choices);
      } else a = this.rng.pick(legal);
      this.ghostPrev = this.ghost; this.ghost = this.nbr[this.ghost][a]; this.ghostDir = a;
    }

    step(a, opts = {}) {
      if (this.done) throw new Error('episode is over; call reset()');
      const R = this.rewards, parts = [];
      let r = R.step; parts.push(['step', R.step]);
      this.steps++; this.lastAction = a;
      const n = this.nbr[this.player][a];
      const bumped = n < 0;
      if (!bumped) this.player = n;
      if (this.pellets[this.player]) { this.pellets[this.player] = 0; r += R.pellet; parts.push(['pellet', R.pellet]); }
      let caught = this.player === this.ghost;
      if (!caught) { this._ghostMove(!!opts.deterministicGhost); caught = this.player === this.ghost; }
      else this.ghostPrev = this.ghost;
      if (caught) { r += R.death; parts.push(['caught', R.death]); this.dead = true; this.done = true; }
      else if (this.pelletsLeft === 0) { r += R.clear; parts.push(['cleared', R.clear]); this.won = true; this.done = true; }
      else if (this.steps >= this.maxSteps) { this.truncated = true; this.done = true; }
      this.score += r;
      return { obs: this.obs(), reward: r, terminal: this.dead || this.won, truncated: this.truncated, parts, bumped, caught };
    }
    legal(c = this.player) { const out = []; for (let a = 0; a < 4; a++) if (this.nbr[c][a] >= 0) out.push(a); return out; }
  }

  // ------------------------------------------------------------ gridworld
  // cell types: 0 empty, 1 wall, 2 cherry (+), 3 ghost (-)
  const EMPTY = 0, WALL = 1, CHERRY = 2, GHOST = 3;
  class GridWorld {
    constructor(W = 7, H = 6, opts = {}) {
      this.W = W; this.H = H;
      this.type = new Uint8Array(W * H);
      this.rewards = Object.assign({ cherry: 10, ghost: -10, step: 0 }, opts.rewards || {});
      this.maxSteps = opts.maxSteps ?? 40;
      this.rng = makeRng(opts.seed ?? 3);
      if (opts.layout) this.load(opts.layout); else this.load(GridWorld.defaultLayout(W, H));
    }
    static defaultLayout(W, H) {
      const t = new Uint8Array(W * H);
      const set = (x, y, v) => { if (x < W && y < H) t[y * W + x] = v; };
      set(W - 1, 0, CHERRY); set(W - 2, 2, GHOST);
      set(2, 1, WALL); set(2, 2, WALL); set(2, 3, WALL); set(4, 3, WALL); set(4, 4, WALL);
      return t;
    }
    load(t) { this.type.set(t); }
    isState(s) { return this.type[s] === EMPTY; }
    // Environment dynamics: bump into walls/edges = stay put; entering cherry/ghost ends the episode.
    transition(s, a) {
      const x = s % this.W, y = Math.floor(s / this.W);
      const nx = x + ACTIONS[a][1], ny = y + ACTIONS[a][0];
      let n = s;
      if (nx >= 0 && nx < this.W && ny >= 0 && ny < this.H && this.type[ny * this.W + nx] !== WALL) n = ny * this.W + nx;
      const t = this.type[n];
      if (t === CHERRY) return { next: n, reward: this.rewards.cherry + this.rewards.step, terminal: true };
      if (t === GHOST) return { next: n, reward: this.rewards.ghost + this.rewards.step, terminal: true };
      return { next: n, reward: this.rewards.step, terminal: false };
    }
    states() { const out = []; for (let s = 0; s < this.type.length; s++) if (this.isState(s)) out.push(s); return out; }
    randomStart() { const st = this.states(); return st.length ? this.rng.pick(st) : -1; }
    encode(s) { const o = new Float32Array(this.W + this.H); o[s % this.W] = 1; o[this.W + Math.floor(s / this.W)] = 1; return o; }
    // BFS distance (in moves) from every square to the nearest cherry, ignoring ghosts as passable? no: ghosts are terminal.
    distanceToCherry() {
      const d = new Int32Array(this.type.length).fill(-1), q = [];
      for (let s = 0; s < d.length; s++) if (this.type[s] === CHERRY) { d[s] = 0; q.push(s); }
      for (let h = 0; h < q.length; h++) {
        const s = q[h], x = s % this.W, y = Math.floor(s / this.W);
        for (const [dy, dx] of ACTIONS) {
          const nx = x + dx, ny = y + dy; if (nx < 0 || nx >= this.W || ny < 0 || ny >= this.H) continue;
          const n = ny * this.W + nx; if (d[n] >= 0 || this.type[n] !== EMPTY) continue;
          d[n] = d[s] + 1; q.push(n);
        }
      }
      return d;
    }
  }

  // Q "brains" for the gridworld: a table (one number per square per move) or a small network.
  class TableQ {
    constructor(nStates) { this.q = new Float32Array(nStates * 4); this.kind = 'table'; }
    values(s) { return this.q.subarray(4 * s, 4 * s + 4); }
    snapshot() { const t = new TableQ(this.q.length / 4); t.q.set(this.q); return t; }
    update(s, a, target, alpha) { const i = 4 * s + a; const old = this.q[i]; this.q[i] = old + alpha * (target - old); return old; }
  }
  class NetQ {
    constructor(world, hidden = 32, seed = 5) { this.world = world; this.net = new MLP([world.W + world.H, hidden, 4], seed, 0.05); this.kind = 'net'; }
    values(s) { return this.net.forward(this.world.encode(s)); }
    snapshot() { const n = Object.create(NetQ.prototype); n.world = this.world; n.kind = 'net'; n.net = this.net.clone(); return n; }
    // batch of {s, a, target}; one Adam step on mean Huber loss. Returns mean loss.
    train(batch, lr) {
      let loss = 0; const d = new Float32Array(4);
      for (const { s, a, target } of batch) {
        const q = this.net.forward(this.world.encode(s));
        const gap = q[a] - target; loss += huber(gap);
        d.fill(0); d[a] = huberGrad(gap) / batch.length; this.net.backward(d);
      }
      this.net.adamStep(lr, 10);
      return loss / batch.length;
    }
  }

  // -------------------------------------------------------------------- DQN
  const DQN_DEFAULTS = {
    hidden: [64, 64], lr: 1e-3, gamma: 0.97, batch: 32, bufferSize: 20000, learningStarts: 1000,
    trainEvery: 1, targetSync: 500, epsStart: 1, epsEnd: 0.05, epsDecay: 15000, rewardScale: 0.01,
    clip: 10, double: true, useTarget: true, useReplay: true, seed: 1,
  };

  class DQN {
    constructor(obsSize, nActions, cfg = {}) {
      this.cfg = Object.assign({}, DQN_DEFAULTS, cfg);
      this.obsSize = obsSize; this.nActions = nActions;
      this.rng = makeRng(this.cfg.seed * 7919 + 13);
      this.live = new MLP([obsSize, ...this.cfg.hidden, nActions], this.cfg.seed);
      this.frozen = this.live.clone();
      this._allocBuffer(this.cfg.bufferSize);
      this.steps = 0; this.updates = 0; this.syncs = 0; this.lastLoss = NaN;
    }
    _allocBuffer(n) {
      this.cap = n; this.bObs = new Uint8Array(n * this.obsSize); this.bNext = new Uint8Array(n * this.obsSize);
      this.bA = new Uint8Array(n); this.bR = new Float32Array(n); this.bT = new Uint8Array(n);
      this.pos = 0; this.size = 0;
    }
    resizeBuffer(n) { this._allocBuffer(n); this.cfg.bufferSize = n; }
    epsilon(step = this.steps) {
      const f = Math.min(1, step / Math.max(1, this.cfg.epsDecay));
      return this.cfg.epsStart + f * (this.cfg.epsEnd - this.cfg.epsStart);
    }
    q(obs) { return this.live.forward(obs); }
    act(obs, eps = this.epsilon()) { return this.rng() < eps ? this.rng.int(this.nActions) : argmax(this.q(obs)); }

    // The heart of it: target = r + gamma * v, v taken from the next state (0 if the game ended).
    target(r, nextObs, terminal) {
      if (terminal) return { target: r, v: 0, aStar: -1, qLiveNext: null, qValNext: null };
      const qLiveNext = this.live.forward(nextObs);
      const valueNet = this.cfg.useTarget ? this.frozen : this.live;
      const qValNext = this.cfg.useTarget ? valueNet.forward(nextObs) : qLiveNext;
      const aStar = this.cfg.double ? argmax(qLiveNext) : argmax(qValNext);
      const v = qValNext[aStar];
      return { target: r + this.cfg.gamma * v, v, aStar, qLiveNext, qValNext };
    }

    // One gradient step on a list of transitions {obs, a, r, next, terminal} (r already scaled).
    trainOn(batch, optimizer = 'adam', lr = this.cfg.lr) {
      let loss = 0; const d = new Float32Array(this.nActions);
      for (const tr of batch) {
        const { target } = this.target(tr.r, tr.next, tr.terminal);
        const q = this.live.forward(tr.obs); // forward last so backward sees these activations
        const gap = q[tr.a] - target; loss += huber(gap);
        d.fill(0); d[tr.a] = huberGrad(gap) / batch.length; this.live.backward(d);
      }
      if (optimizer === 'sgd') this.live.sgdStep(lr, this.cfg.clip); else this.live.adamStep(lr, this.cfg.clip);
      this.updates++;
      this.lastLoss = loss / batch.length;
      return this.lastLoss;
    }

    _get(i) {
      const o = this.obsSize;
      return { obs: this.bObs.subarray(i * o, i * o + o), a: this.bA[i], r: this.bR[i], next: this.bNext.subarray(i * o, i * o + o), terminal: this.bT[i] === 1 };
    }
    store(obs, a, rScaled, next, terminal) {
      const i = this.pos, o = this.obsSize;
      for (let k = 0; k < o; k++) { this.bObs[i * o + k] = obs[k] > 0.5 ? 1 : 0; this.bNext[i * o + k] = next[k] > 0.5 ? 1 : 0; }
      this.bA[i] = a; this.bR[i] = rScaled; this.bT[i] = terminal ? 1 : 0;
      this.pos = (this.pos + 1) % this.cap; this.size = Math.min(this.size + 1, this.cap);
    }
    sample(n) { const out = []; for (let k = 0; k < n; k++) out.push(this._get(this.rng.int(this.size))); return out; }

    // Called once per environment step. Returns the loss if a gradient step happened.
    observe(obs, a, reward, next, terminal) {
      const r = reward * this.cfg.rewardScale;
      this.steps++;
      let loss = null;
      if (this.cfg.useReplay) {
        this.store(obs, a, r, next, terminal);
        if (this.size >= Math.min(this.cfg.learningStarts, this.cap) && this.steps % this.cfg.trainEvery === 0) loss = this.trainOn(this.sample(this.cfg.batch));
      } else if (this.steps >= this.cfg.learningStarts && this.steps % this.cfg.trainEvery === 0) {
        loss = this.trainOn([{ obs: Float32Array.from(obs), a, r, next: Float32Array.from(next), terminal }]);
      }
      if (this.cfg.useTarget && this.steps % this.cfg.targetSync === 0) this.sync();
      return loss;
    }
    sync() { this.frozen.copyFrom(this.live); this.syncs++; }
  }

  return { makeRng, argmax, huber, huberGrad, ACTIONS, ACTION_NAMES, OPPOSITE, MLP, MiniPac, MINI_MAZE, DEFAULT_REWARDS,
    GridWorld, TableQ, NetQ, EMPTY, WALL, CHERRY, GHOST, DQN, DQN_DEFAULTS, bytesToB64, b64ToBytes };
})();
if (typeof module !== 'undefined') module.exports = Engine;
