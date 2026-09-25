/* Arcade: the 1980 Pac-Man arcade rules as a frame-by-frame simulation, written from public documentation.
   Every rule is listed in SPEC.md with its source; the numbers in comments (e.g. "6.3") point there.
   No ROM data of any kind is used. Pure JS, no dependencies: runs in the browser (global Arcade) and in
   Node (module.exports), the same way workshop/engine.js does. */
const Arcade = (() => {
  'use strict';

  // ------------------------------------------------------------------ RNG
  // xorshift32: tiny and easy to reseed (6.4 needs the same reseed at every level start and lost life).
  function makeXorshift(seed) {
    let s = (seed >>> 0) || 0x9e3779b9;
    const next = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; };
    next.int = (n) => next() % n;
    next.getState = () => s;
    next.setState = (v) => { s = v >>> 0 || 0x9e3779b9; };
    return next;
  }
  const FRIGHT_SEED = 0x12345678;

  // ----------------------------------------------------------- directions
  // Internal order is the ghosts' tie-break preference (6.3): up, left, down, right.
  const UP = 0, LEFT = 1, DOWN = 2, RIGHT = 3;
  const DX = [0, -1, 0, 1], DY = [-1, 0, 1, 0];
  const REV = [DOWN, RIGHT, UP, LEFT];
  const CLOCKWISE = [RIGHT, UP, LEFT, DOWN]; // the next direction clockwise from each (6.4)
  const DIR_NAMES = ['up', 'left', 'down', 'right'];

  // ----------------------------------------------------------------- maze
  // Screen rows 3..33 (2.1). '#' wall, '.' dot, 'o' energizer, '-' ghost house door, ' ' open or dead space.
  const MAZE = [
    '############################',
    '#............##............#',
    '#.####.#####.##.#####.####.#',
    '#o#  #.#   #.##.#   #.#  #o#',
    '#.####.#####.##.#####.####.#',
    '#..........................#',
    '#.####.##.########.##.####.#',
    '#.####.##.########.##.####.#',
    '#......##....##....##......#',
    '######.##### ## #####.######',
    '     #.##### ## #####.#     ',
    '     #.##          ##.#     ',
    '     #.## ###--### ##.#     ',
    '######.## #      # ##.######',
    '      .   #      #   .      ',
    '######.## #      # ##.######',
    '     #.## ######## ##.#     ',
    '     #.##          ##.#     ',
    '     #.## ######## ##.#     ',
    '######.## ######## ##.######',
    '#............##............#',
    '#.####.#####.##.#####.####.#',
    '#.####.#####.##.#####.####.#',
    '#o..##.......  .......##..o#',
    '###.##.##.########.##.##.###',
    '###.##.##.########.##.##.###',
    '#......##....##....##......#',
    '#.##########.##.##########.#',
    '#.##########.##.##########.#',
    '#..........................#',
    '############################',
  ];
  const COLS = 28, ROWS = 36, TOP = 3, TUNNEL_ROW = 17, NDOTS = 244;
  const OPEN = 0, WALL = 1, DOOR = 2;
  const cell = new Uint8Array(COLS * ROWS).fill(WALL);
  const DOTS0 = new Uint8Array(COLS * ROWS); // 1 = dot, 2 = energizer
  MAZE.forEach((row, r) => {
    for (let x = 0; x < COLS; x++) {
      const ch = row[x], i = (r + TOP) * COLS + x;
      cell[i] = ch === '#' ? WALL : ch === '-' ? DOOR : OPEN;
      if (ch === '.') DOTS0[i] = 1;
      if (ch === 'o') DOTS0[i] = 2;
    }
  });
  function cellAt(x, y) {
    if (y < 0 || y >= ROWS) return WALL;
    if (x < 0 || x >= COLS) { if (y !== TUNNEL_ROW) return WALL; x = (x + COLS) % COLS; }
    return cell[y * COLS + x];
  }
  const wrapX = (x) => (x + COLS) % COLS;
  // Legal space: every tile Pac-Man can reach from his start (the house interior is not part of it).
  const legal = new Uint8Array(COLS * ROWS);
  (() => {
    const q = [26 * COLS + 13]; legal[q[0]] = 1;
    for (let h = 0; h < q.length; h++) {
      const i = q[h], x = i % COLS, y = (i / COLS) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d], ny = y + DY[d];
        if (cellAt(nx, ny) !== OPEN) continue;
        const j = ny * COLS + wrapX(nx);
        if (!legal[j]) { legal[j] = 1; q.push(j); }
      }
    }
  })();
  const walkable = (x, y) => cellAt(x, y) === OPEN && legal[y * COLS + wrapX(x)] === 1;
  const isRedZone = (x, y) => (y === 14 || y === 26) && x >= 11 && x <= 16; // 2.5
  const isTunnel = (x, y) => y === TUNNEL_ROW && (x <= 4 || x >= 23); // 2.4

  // ------------------------------------------------------------ level data
  const BONUS = [ // 15.2
    ['cherries', 100], ['strawberry', 300], ['peach', 500], ['peach', 500], ['apple', 700], ['apple', 700],
    ['grapes', 1000], ['grapes', 1000], ['galaxian', 2000], ['galaxian', 2000], ['bell', 3000], ['bell', 3000], ['key', 5000],
  ];
  const FRIGHT_SEC = [6, 5, 4, 3, 2, 5, 2, 2, 1, 5, 2, 1, 1, 3, 1, 1, 0, 1, 0]; // 9.1, level 19+ is 0
  const FLASHES = [5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3, 3, 5, 3, 3, 0, 3, 0]; // 9.3
  const ELROY1_DOTS = [20, 30, 40, 40, 40, 50, 50, 50, 60, 60, 60, 80, 80, 80, 100, 100, 100, 100, 120]; // 12.1
  const FLASH_FRAMES = 28;

  function levelSpec(level) {
    const L = Math.max(1, level | 0), i = Math.min(L, 19) - 1;
    const g = L === 1 ? 0 : L <= 4 ? 1 : L <= 20 ? 2 : 3; // speed groups of 4.4
    const b = BONUS[Math.min(L, 13) - 1];
    const s = (secs) => secs.map((t) => Math.round(t * 60));
    return {
      level: L,
      pac: [80, 90, 100, 90][g], pacFright: [90, 95, 100, 100][g], // 4.4
      ghost: [75, 85, 95, 95][g], ghostFright: [50, 55, 60, 60][g], ghostTunnel: [40, 45, 50, 50][g],
      elroy1: [80, 90, 100, 100][g], elroy2: [85, 95, 105, 105][g], // 12.2
      elroy1Dots: ELROY1_DOTS[i], elroy2Dots: ELROY1_DOTS[i] / 2, // 12.1
      frightFrames: FRIGHT_SEC[i] * 60, flashes: FLASHES[i], // 9.1, 9.3
      modes: L === 1 ? s([7, 20, 7, 20, 5, 20, 5]) : L <= 4 ? s([7, 20, 7, 20, 5, 1033, 1 / 60]) : s([5, 20, 5, 20, 5, 1037, 1 / 60]), // 8.1
      dotLimits: [0, 0, L === 1 ? 30 : 0, L === 1 ? 60 : L === 2 ? 50 : 0], // 10.3 (Blinky never waits)
      dotTimer: L <= 4 ? 240 : 180, // 10.5
      bonusName: b[0], bonus: b[1],
    };
  }

  // --------------------------------------------------------------- actors
  const BLINKY = 0, PINKY = 1, INKY = 2, CLYDE = 3;
  const GHOST_NAMES = ['Blinky', 'Pinky', 'Inky', 'Clyde'];
  const SCATTER = [[25, 0], [2, 0], [27, 35], [0, 35]]; // 7.1
  const EYES_TARGET = [13, 14]; // 7.6
  const EXIT_X = 111, EXIT_Y = 116, HOUSE_Y = 140; // 3.2: the spot above the door, and the house's middle row
  const HOME_X = [111, 111, 95, 127];
  const PAC_START = [111, 212]; // 3.1
  const EYES_SPEED = 120, HOUSE_SPEED = 40; // 4.7
  // ghost states
  const HOUSE = 0, LEAVING = 1, ACTIVE = 2, EYES = 3, ENTERING = 4;
  const STATE_NAMES = ['house', 'leaving', 'active', 'eyes', 'entering'];
  // pauses (14.1, 14.2), in frames
  const READY_FRAMES = 130, GHOST_EATEN_FRAMES = 60, DYING_FRAMES = 210, LEVEL_CLEAR_FRAMES = 240;
  const EXTRA_LIFE_AT = 10000, MAX_LEVEL = 255;

  class Game {
    constructor(opts = {}) {
      this.opts = Object.assign({ seed: 1, level: 1, lives: 3, readyFrames: READY_FRAMES, pauses: true }, opts);
      this.dots = new Uint8Array(COLS * ROWS);
      this.pac = { x: 0, y: 0, dir: LEFT, acc: 0, stall: 0, moving: false };
      this.ghosts = [BLINKY, PINKY, INKY, CLYDE].map((id) => ({
        id, name: GHOST_NAMES[id], x: 0, y: 0, tx: 0, ty: 0, dir: LEFT, turnDir: LEFT, nextTurnDir: LEFT,
        state: HOUSE, frightened: false, pendingReverse: false, exitReversed: false, acc: 0, dotCounter: 0,
      }));
      this.events = [];
      this.newGame(this.opts.seed);
    }

    // ------------------------------------------------------ game setup
    newGame(seed = this.opts.seed) {
      this.seed = seed >>> 0;
      this.rng = makeXorshift(Math.imul(this.seed + 1, 0x9e3779b1));
      this.score = 0; this.lives = this.opts.lives; this.extraLifeGiven = false;
      this.frame = 0; this.over = false; this.lastEaten = null;
      this.startLevel(this.opts.level);
    }
    startLevel(level) {
      this.level = level; this.spec = levelSpec(level);
      this.dots.set(DOTS0); this.dotsLeft = NDOTS; this.dotsEaten = 0;
      for (const g of this.ghosts) g.dotCounter = 0; // 10.3
      this.globalActive = false; this.globalCounter = 0; // 10.4
      this.elroySuspended = false;
      this._resetRound();
    }
    // Positions, modes and timers for a new level or a new life.
    _resetRound() {
      const sp = this.spec;
      this.modeIndex = 0; this.modeTimer = sp.modes[0]; // 8.2
      this.frightTimer = 0; this.chain = 0; this.dotTimer = 0;
      this.fruit = 0; this.fruitTimer = 0;
      this.frightRng = makeXorshift(FRIGHT_SEED); // 6.4
      const p = this.pac;
      p.x = PAC_START[0]; p.y = PAC_START[1]; p.dir = LEFT; p.acc = 0; p.stall = 0; p.moving = false;
      for (const g of this.ghosts) {
        g.acc = 0; g.frightened = false; g.pendingReverse = false; g.exitReversed = false;
        g.y = g.id === BLINKY ? EXIT_Y : HOUSE_Y; g.x = HOME_X[g.id];
        g.tx = g.x >> 3; g.ty = g.y >> 3;
        g.state = g.id === BLINKY ? ACTIVE : HOUSE;
        g.dir = g.id === BLINKY ? LEFT : g.id === PINKY ? DOWN : UP;
      }
      const b = this.ghosts[BLINKY];
      b.turnDir = LEFT; b.nextTurnDir = this._decide(b, b.tx - 1, b.ty, LEFT);
      this.phase = 'ready'; this.freeze = this.opts.pauses ? this.opts.readyFrames : 0;
      if (this.freeze === 0) this.phase = 'play';
    }

    // ------------------------------------------------------ one frame
    // joy: UP, LEFT, DOWN, RIGHT or null (stick centered). Returns this frame's events.
    step(joy = null) {
      this.events.length = 0;
      if (this.over) return this.events;
      this.frame++;
      if (this.freeze > 0) {
        if (--this.freeze === 0) this._endPause();
        return this.events;
      }
      this._play(joy);
      return this.events;
    }
    // Without pauses (training) a pause still takes one frame, so nothing else happens in the frame that caused it.
    _pause(phase, frames) { this.phase = phase; this.freeze = this.opts.pauses ? frames : 1; }
    _endPause() {
      const phase = this.phase;
      this.phase = 'play';
      if (phase === 'dying') this._afterDeath();
      else if (phase === 'levelClear') {
        if (this.level >= MAX_LEVEL) { this.over = true; this.phase = 'over'; this.events.push({ type: 'gameOver', reason: 'level 256' }); } // 13.4
        else { this.startLevel(this.level + 1); this.events.push({ type: 'level', level: this.level }); }
      }
    }

    _play(joy) {
      const sp = this.spec;
      if (this.frightTimer > 0) { if (--this.frightTimer === 0) this._endFright(); } // 8.3: mode timer paused
      else if (--this.modeTimer === 0) this._nextMode();
      if (++this.dotTimer >= sp.dotTimer) { this.dotTimer = 0; const g = this._preferredWaiting(); if (g) this._release(g, 'timer'); } // 10.5
      if (this.fruit && --this.fruitTimer === 0) { this.fruit = 0; this.events.push({ type: 'fruitGone' }); }
      this._checkHouse();
      this._movePac(joy);
      if (this.phase !== 'play') return; // the last dot ends the level at once
      for (const g of this.ghosts) this._moveGhost(g);
      this._collide(); // 11.1: once, after everyone has moved
    }

    // ------------------------------------------------------ Pac-Man (5.x)
    _movePac(joy) {
      const p = this.pac, sp = this.spec;
      p.moving = false;
      if (p.stall > 0) { p.stall--; return; } // 4.3: the whole movement waits a frame
      p.acc += this.frightTimer > 0 ? sp.pacFright : sp.pac; // 4.2, 4.5
      const n = (p.acc / 80) | 0; p.acc -= n * 80;
      for (let k = 0; k < n && this.phase === 'play'; k++) this._pacStep(joy);
    }
    _pacStep(joy) {
      const p = this.pac, tx = p.x >> 3, ty = p.y >> 3;
      if (joy != null && joy !== p.dir && (joy === REV[p.dir] || walkable(tx + DX[joy], ty + DY[joy]))) p.dir = joy; // 5.2, 5.3
      const d = p.dir, cx = tx * 8 + 3, cy = ty * 8 + 4;
      if (!walkable(tx + DX[d], ty + DY[d]) && // 5.5: wall ahead, so stop at the tile's center
        ((d === RIGHT && p.x >= cx) || (d === LEFT && p.x <= cx) || (d === DOWN && p.y >= cy) || (d === UP && p.y <= cy))) return;
      p.x += DX[d]; p.y += DY[d];
      if (DX[d] !== 0) p.y += Math.sign(cy - p.y); else p.x += Math.sign(cx - p.x); // 5.4: cut the corner
      if (p.x < 0) p.x += 224; else if (p.x >= 224) p.x -= 224; // 2.3
      p.moving = true;
      const nx = p.x >> 3, ny = p.y >> 3;
      if (nx !== tx || ny !== ty) this._pacEnter(nx, ny);
    }
    _pacEnter(x, y) {
      const i = y * COLS + x, t = this.dots[i];
      if (!t) return;
      this.dots[i] = 0; this.dotsLeft--; this.dotsEaten++;
      if (t === 1) { this._addScore(10); this.pac.stall = 1; this.events.push({ type: 'dot', points: 10 }); } // 4.3, 13.1
      else { this._addScore(50); this.pac.stall = 3; this.events.push({ type: 'energizer', points: 50 }); this._energize(); }
      this.dotTimer = 0; // 10.5, 10.7
      if (this.globalActive) this.globalCounter++; // 10.4
      else { const g = this._preferredWaiting(); if (g) g.dotCounter++; } // 10.2
      if (this.dotsEaten === 70 || this.dotsEaten === 170) { // 15.1
        this.fruit = this.spec.bonus; this.fruitTimer = 540 + this.rng.int(61);
        this.events.push({ type: 'fruit', name: this.spec.bonusName, points: this.spec.bonus });
      }
      if (this.dotsLeft === 0) { this.fruit = 0; this._pause('levelClear', LEVEL_CLEAR_FRAMES); this.events.push({ type: 'levelClear', level: this.level }); }
    }
    _addScore(n) {
      const before = this.score; this.score += n;
      if (!this.extraLifeGiven && before < EXTRA_LIFE_AT && this.score >= EXTRA_LIFE_AT) { // 13.2
        this.extraLifeGiven = true; this.lives++; this.events.push({ type: 'extraLife' });
      }
    }

    // ------------------------------------------------------ modes (8.x, 9.x)
    get scatter() { return this.modeIndex % 2 === 0 && this.modeIndex < this.spec.modes.length; }
    get flashing() { return this.frightTimer > 0 && this.frightTimer <= this.spec.flashes * FLASH_FRAMES; }
    _nextMode() {
      this.modeIndex++;
      this.modeTimer = this.modeIndex < this.spec.modes.length ? this.spec.modes[this.modeIndex] : Infinity;
      this._signalReverse();
      this.events.push({ type: 'mode', mode: this.scatter ? 'scatter' : 'chase' });
    }
    // 8.4, 10.6: ghosts in the maze reverse at their next tile; ghosts in or entering the house remember it.
    _signalReverse() {
      for (const g of this.ghosts) {
        if (g.state === ACTIVE) g.pendingReverse = true;
        else if (g.state === HOUSE || g.state === LEAVING || g.state === ENTERING) g.exitReversed = true;
      }
    }
    _energize() {
      const f = this.spec.frightFrames;
      this.chain = 0; // 9.4
      if (f > 0) this.frightTimer = f;
      for (const g of this.ghosts) if (f > 0 && g.state !== EYES && g.state !== ENTERING) g.frightened = true; // 9.6
      this._signalReverse(); // 9.2: on levels without fright they still reverse
      this.events.push({ type: 'fright', frames: f });
    }
    _endFright() { for (const g of this.ghosts) g.frightened = false; this.events.push({ type: 'frightEnd' }); } // 8.4: no reversal

    // ------------------------------------------------------ Cruise Elroy (12.x)
    elroy() {
      if (this.elroySuspended) return 0;
      return this.dotsLeft <= this.spec.elroy2Dots ? 2 : this.dotsLeft <= this.spec.elroy1Dots ? 1 : 0;
    }

    // ------------------------------------------------------ ghost house (10.x)
    _preferredWaiting() {
      for (const id of [PINKY, INKY, CLYDE]) if (this.ghosts[id].state === HOUSE) return this.ghosts[id];
      return null;
    }
    _release(g, why) {
      g.state = LEAVING;
      if (g.id === CLYDE) this.elroySuspended = false; // 12.4
      this.events.push({ type: 'release', ghost: g.id, why });
    }
    _checkHouse() {
      if (this.globalActive) { // 10.4
        const c = this.globalCounter, P = this.ghosts[PINKY], I = this.ghosts[INKY], C = this.ghosts[CLYDE];
        if (c === 7 && P.state === HOUSE) this._release(P, 'global');
        if (c === 17 && I.state === HOUSE) this._release(I, 'global');
        if (c === 32 && C.state === HOUSE) { this.globalActive = false; this.globalCounter = 0; }
      } else {
        const g = this._preferredWaiting();
        if (g && g.dotCounter >= this.spec.dotLimits[g.id]) this._release(g, 'dots'); // 10.3
      }
    }

    // ------------------------------------------------------ ghosts (6.x, 7.x)
    _ghostSpeed(g) { // 4.6
      const sp = this.spec;
      if (g.state === EYES || g.state === ENTERING) return EYES_SPEED;
      if (g.state === HOUSE || g.state === LEAVING) return HOUSE_SPEED;
      if (isTunnel(g.x >> 3, g.y >> 3)) return sp.ghostTunnel;
      if (g.frightened) return sp.ghostFright;
      if (g.id === BLINKY) { const e = this.elroy(); if (e === 2) return sp.elroy2; if (e === 1) return sp.elroy1; }
      return sp.ghost;
    }
    _moveGhost(g) {
      g.acc += this._ghostSpeed(g);
      const n = (g.acc / 80) | 0; g.acc -= n * 80;
      for (let k = 0; k < n; k++) this._ghostStep(g);
    }
    _ghostStep(g) {
      if (g.state === HOUSE) { // bob up and down
        if (g.y <= HOUSE_Y - 4) g.dir = DOWN; else if (g.y >= HOUSE_Y + 4) g.dir = UP;
        g.y += DY[g.dir];
      } else if (g.state === LEAVING) { // to the middle row, to the center, then up through the door
        if (g.x === EXIT_X) { g.dir = UP; g.y--; if (g.y === EXIT_Y) this._activate(g); }
        else if (g.y !== HOUSE_Y) { g.dir = g.y < HOUSE_Y ? DOWN : UP; g.y += DY[g.dir]; }
        else { g.dir = g.x < EXIT_X ? RIGHT : LEFT; g.x += DX[g.dir]; }
      } else if (g.state === ENTERING) { // down through the door, then to this ghost's slot
        if (g.y < HOUSE_Y) { g.dir = DOWN; g.y++; }
        else if (g.x !== HOME_X[g.id]) { g.dir = g.x < HOME_X[g.id] ? RIGHT : LEFT; g.x += DX[g.dir]; }
        else this._revive(g);
      } else { // ACTIVE or EYES: tile-by-tile pathfinding
        g.x += DX[g.dir]; g.y += DY[g.dir];
        if (g.x < 0) g.x += 224; else if (g.x >= 224) g.x -= 224;
        const tx = g.x >> 3, ty = g.y >> 3;
        if (tx !== g.tx || ty !== g.ty) { g.tx = tx; g.ty = ty; this._ghostEnter(g); }
        if (g.state === EYES && g.x === EXIT_X && g.y === EXIT_Y) { g.state = ENTERING; g.exitReversed = false; return; }
        if ((g.x & 7) === 3 && (g.y & 7) === 4) g.dir = g.turnDir; // 6.5: turns happen at tile centers
      }
      g.tx = g.x >> 3; g.ty = g.y >> 3;
    }
    // 6.1: on entering a tile, the turn decided a tile ago becomes current, and the next tile is decided now.
    _ghostEnter(g) {
      if (g.pendingReverse && g.state === ACTIVE) { // 8.5
        g.pendingReverse = false;
        g.dir = REV[g.dir]; g.turnDir = g.dir;
        g.nextTurnDir = this._decide(g, g.tx + DX[g.dir], g.ty + DY[g.dir], g.dir);
        return;
      }
      g.turnDir = g.nextTurnDir;
      g.nextTurnDir = this._decide(g, g.tx + DX[g.turnDir], g.ty + DY[g.turnDir], g.turnDir);
    }
    // The exit a ghost will take from tile (bx, by), arriving there moving in direction e.
    _decide(g, bx, by, e) {
      bx = wrapX(bx);
      if (g.frightened && g.state === ACTIVE) { // 6.4
        let d = this.frightRng() & 3;
        for (let k = 0; k < 4; k++, d = CLOCKWISE[d]) if (d !== REV[e] && walkable(bx + DX[d], by + DY[d])) return d;
        return REV[e];
      }
      const [tx, ty] = this.targetOf(g);
      let best = -1, bestDist = Infinity;
      for (let d = 0; d < 4; d++) { // 6.2, 6.3: up, left, down, right breaks ties
        if (d === REV[e] || (d === UP && g.state !== EYES && isRedZone(bx, by))) continue;
        const nx = bx + DX[d], ny = by + DY[d];
        if (!walkable(nx, ny)) continue;
        const dist = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty);
        if (dist < bestDist) { bestDist = dist; best = d; }
      }
      return best >= 0 ? best : REV[e];
    }
    // 7.x: the tile a ghost is heading for right now.
    targetOf(g) {
      if (g.state === EYES || g.state === ENTERING) return EYES_TARGET;
      const p = this.pac, px = p.x >> 3, py = p.y >> 3;
      if (g.frightened) return null;
      if (g.id === BLINKY && this.elroy() > 0) return [px, py]; // 12.3
      if (this.scatter) return SCATTER[g.id];
      if (g.id === BLINKY) return [px, py]; // 7.2
      if (g.id === PINKY) { const [ox, oy] = ahead(p.dir, 4); return [px + ox, py + oy]; } // 7.3
      if (g.id === INKY) { // 7.4
        const [ox, oy] = ahead(p.dir, 2), b = this.ghosts[BLINKY];
        return [2 * (px + ox) - (b.x >> 3), 2 * (py + oy) - (b.y >> 3)];
      }
      const dx = (g.x >> 3) - px, dy = (g.y >> 3) - py; // 7.5
      return dx * dx + dy * dy >= 64 ? [px, py] : SCATTER[CLYDE];
    }
    _activate(g) { // left the house (10.6)
      g.state = ACTIVE; g.x = EXIT_X; g.y = EXIT_Y; g.tx = g.x >> 3; g.ty = g.y >> 3;
      g.dir = g.exitReversed ? RIGHT : LEFT; g.exitReversed = false; g.pendingReverse = false;
      g.turnDir = g.dir;
      g.nextTurnDir = this._decide(g, g.tx + DX[g.dir], g.ty, g.dir);
    }
    _revive(g) { // 9.5, 10.1
      g.frightened = false; g.pendingReverse = false;
      if (g.id === BLINKY) g.state = LEAVING; else { g.state = HOUSE; g.dir = UP; }
      this.events.push({ type: 'revived', ghost: g.id });
    }

    // ------------------------------------------------------ collisions (11.x)
    _collide() {
      const p = this.pac, px = p.x >> 3, py = p.y >> 3;
      for (const g of this.ghosts) {
        if ((g.state !== ACTIVE && g.state !== LEAVING) || (g.x >> 3) !== px || (g.y >> 3) !== py) continue;
        if (g.frightened) { this._eatGhost(g); return; }
        this._pacDies(g); return;
      }
      if (this.fruit && px === 13 && py === 20) { // 15.1
        const pts = this.fruit; this.fruit = 0; this._addScore(pts);
        this.events.push({ type: 'fruitEaten', points: pts, name: this.spec.bonusName });
      }
    }
    _eatGhost(g) {
      this.chain++;
      const pts = 200 << (this.chain - 1); // 9.4
      this._addScore(pts);
      g.frightened = false; g.pendingReverse = false;
      if (g.state === LEAVING) { g.state = ENTERING; g.exitReversed = false; } // caught in the doorway: straight back in
      else { g.state = EYES; g.nextTurnDir = this._decide(g, g.tx + DX[g.turnDir], g.ty + DY[g.turnDir], g.turnDir); }
      this.lastEaten = { ghost: g.id, points: pts, x: g.x, y: g.y, frame: this.frame };
      this.events.push({ type: 'ghostEaten', ghost: g.id, points: pts });
      this._pause('ghostEaten', GHOST_EATEN_FRAMES); // 14.1
    }
    _pacDies(g) {
      this.lives--; this.fruit = 0;
      this.events.push({ type: 'death', ghost: g.id, lives: this.lives });
      this._pause('dying', DYING_FRAMES);
    }
    _afterDeath() {
      if (this.lives <= 0) { this.over = true; this.phase = 'over'; this.events.push({ type: 'gameOver', reason: 'no lives' }); return; }
      this.globalActive = true; this.globalCounter = 0; // 10.4
      this.elroySuspended = true; // 12.4
      this._resetRound();
    }

    // ------------------------------------------------------ snapshots
    snapshot() {
      return JSON.parse(JSON.stringify({
        seed: this.seed, rng: this.rng.getState(), frightRng: this.frightRng.getState(), dots: Array.from(this.dots),
        pac: this.pac, ghosts: this.ghosts, level: this.level, score: this.score, lives: this.lives, extraLifeGiven: this.extraLifeGiven,
        frame: this.frame, over: this.over, phase: this.phase, freeze: this.freeze, dotsLeft: this.dotsLeft, dotsEaten: this.dotsEaten,
        globalActive: this.globalActive, globalCounter: this.globalCounter, elroySuspended: this.elroySuspended,
        modeIndex: this.modeIndex, modeTimer: this.modeTimer === Infinity ? -1 : this.modeTimer, frightTimer: this.frightTimer, chain: this.chain,
        dotTimer: this.dotTimer, fruit: this.fruit, fruitTimer: this.fruitTimer,
      }));
    }
    restore(s) {
      this.seed = s.seed; this.rng.setState(s.rng); this.frightRng.setState(s.frightRng); this.dots.set(s.dots);
      Object.assign(this.pac, s.pac); s.ghosts.forEach((g, i) => Object.assign(this.ghosts[i], g));
      this.level = s.level; this.spec = levelSpec(s.level);
      for (const k of ['score', 'lives', 'extraLifeGiven', 'frame', 'over', 'phase', 'freeze', 'dotsLeft', 'dotsEaten', 'globalActive',
        'globalCounter', 'elroySuspended', 'modeIndex', 'frightTimer', 'chain', 'dotTimer', 'fruit', 'fruitTimer']) this[k] = s[k];
      this.modeTimer = s.modeTimer === -1 ? Infinity : s.modeTimer;
    }
    // A short fingerprint of the whole state (determinism tests).
    hash() {
      let h = 2166136261;
      const mix = (v) => { h ^= v & 0xffff; h = Math.imul(h, 16777619); h ^= (v >>> 16) & 0xffff; h = Math.imul(h, 16777619); };
      const p = this.pac; [p.x, p.y, p.dir, p.acc, p.stall, this.score, this.lives, this.level, this.dotsLeft, this.frightTimer,
        this.modeIndex, this.dotTimer, this.fruit, this.fruitTimer, this.frame, this.globalCounter].forEach(mix);
      for (const g of this.ghosts) [g.x, g.y, g.dir, g.turnDir, g.nextTurnDir, g.state, g.frightened ? 1 : 0, g.acc, g.dotCounter].forEach(mix);
      return (h >>> 0).toString(16);
    }
  }
  function ahead(dir, n) { // 7.3, 7.4: facing up also shifts left (the overflow bug)
    return dir === UP ? [-n, -n] : dir === LEFT ? [-n, 0] : dir === DOWN ? [0, n] : [n, 0];
  }

  // =================================================================== RL
  // A decision is made every time Pac-Man enters a new tile (or every 8 frames while he stands at a wall).
  // Actions use the Workshop's order: 0 up, 1 down, 2 left, 3 right.
  const ACTION_DIRS = [UP, DOWN, LEFT, RIGHT];
  const ACTION_OF_DIR = [0, 2, 1, 3]; // internal dir -> action index
  const ACTION_NAMES = ['up', 'down', 'left', 'right'];

  // Precomputed maze graph over legal tiles for the observation's path distances.
  const legalTiles = [];
  const tileIndex = new Int16Array(COLS * ROWS).fill(-1);
  for (let i = 0; i < COLS * ROWS; i++) if (legal[i]) { tileIndex[i] = legalTiles.length; legalTiles.push(i); }
  const NLEGAL = legalTiles.length;
  const nbr = new Int16Array(NLEGAL * 4).fill(-1);
  legalTiles.forEach((i, k) => {
    const x = i % COLS, y = (i / COLS) | 0;
    for (let a = 0; a < 4; a++) {
      const d = ACTION_DIRS[a], nx = x + DX[d], ny = y + DY[d];
      if (walkable(nx, ny)) nbr[k * 4 + a] = tileIndex[ny * COLS + wrapX(nx)];
    }
  });

  // Observation layout (all values in [0, 1]):
  //   WINDOW: 15x15 tiles centered on Pac-Man x 7 channels
  //   PATHS: for each move, maze distances from the square it leads to (not back through Pac-Man)
  //   GLOBAL: Pac-Man, timers, each ghost's state and offset, level, lives
  //   DOTMAP: remaining dots per 4x4-tile block
  const WIN = 15, HALF = 7, WCH = ['wall', 'dot', 'energizer', 'ghost', 'ghost heading', 'blue ghost', 'fruit'];
  const PATH_F = 23, GLOBAL_F = 45, DOTMAP_W = 7, DOTMAP_H = 8;
  const OBS = { window: WIN * WIN * WCH.length, paths: 4 * PATH_F, global: GLOBAL_F, dotmap: DOTMAP_W * DOTMAP_H };
  OBS.size = OBS.window + OBS.paths + OBS.global + OBS.dotmap;
  const PATH_MAX = 64;

  class ArcadeEnv {
    constructor(opts = {}) {
      // sticky: each frame, the chance that the stick stays where it was last frame (Machado et al. 2018), so a new
      // direction arrives a frame or two late now and then. Games stop repeating exactly and memorized routes break.
      this.opts = Object.assign({ seed: 1, level: 1, lives: 3, maxWaitFrames: 8, pauses: false, sticky: 0 }, opts);
      this.game = new Game({ seed: this.opts.seed, level: this.opts.level, lives: this.opts.lives, pauses: this.opts.pauses });
      this.obsSize = OBS.size; this.nActions = 4;
      this._dist = new Int16Array(NLEGAL); this._queue = new Int16Array(NLEGAL);
      this._obs = new Float32Array(OBS.size);
      this.decisions = 0;
      this._reseed(this.opts.seed);
    }
    _reseed(seed) { this._stickRng = makeXorshift(Math.imul((seed >>> 0) + 17, 0x2c1b3c6d)); this.joy = null; }
    reset(seed = this.opts.seed) { this.game.newGame(seed); this.decisions = 0; this._reseed(seed); return this.observe(); }
    // Hold the stick in the chosen direction until the next decision point. Returns the points scored meanwhile.
    step(action) {
      this.begin(action);
      let r = null;
      while (r === null) r = this.tick();
      return r;
    }
    // The same decision one frame at a time, for the page: begin(action), then tick() once per frame until it
    // returns the result. With the game's pauses on, frozen frames don't count toward the decision, except the last
    // frame of a ghost-eaten pause, which the training game (no pauses) also spends. So both make identical decisions.
    begin(action) {
      const g = this.game;
      this._dir = ACTION_DIRS[action];
      this._s = { score0: g.score, t0: (g.pac.x >> 3) + (g.pac.y >> 3) * COLS, frames: 0, ending: false,
        lifeLost: false, cleared: false, ghosts: 0, energizers: 0, fruit: 0, dots: 0 };
    }
    tick() {
      const g = this.game, s = this._s;
      if (s.ending) { // after a lost life or a cleared level, play on to the fresh start so the next decision sees it
        if (!g.over && g.phase !== 'play') { g.step(null); s.frames++; }
        return !g.over && g.phase !== 'play' ? null : this._finish();
      }
      if (g.over) return this._finish();
      const fz = g.freeze, counts = fz === 0 || (fz === 1 && g.phase === 'ghostEaten');
      let dir = this._dir;
      if (this.opts.sticky > 0 && counts) { // only on frames the training game also has, so paused and unpaused stay in step
        if (this.joy !== null && this._stickRng() / 4294967296 < this.opts.sticky) dir = this.joy;
        this.joy = dir;
      }
      const ev = g.step(dir);
      if (counts) s.frames++;
      for (const e of ev) {
        if (e.type === 'death') s.lifeLost = true;
        else if (e.type === 'levelClear') s.cleared = true;
        else if (e.type === 'ghostEaten') s.ghosts++;
        else if (e.type === 'energizer') s.energizers++;
        else if (e.type === 'fruitEaten') s.fruit++;
        else if (e.type === 'dot') s.dots++;
      }
      if (s.lifeLost || s.cleared) { s.ending = true; return !g.over && g.phase !== 'play' ? null : this._finish(); }
      if (g.over) return this._finish();
      if (!counts) return null;
      const t = (g.pac.x >> 3) + (g.pac.y >> 3) * COLS;
      if (t !== s.t0) return this._finish();
      if ((!g.pac.moving && g.pac.stall === 0 && s.frames >= this.opts.maxWaitFrames) || s.frames >= 64) return this._finish();
      return null;
    }
    _finish() {
      const g = this.game, s = this._s;
      this.decisions++;
      return { obs: this.observe(), points: g.score - s.score0, lifeLost: s.lifeLost, cleared: s.cleared, done: g.over, frames: s.frames,
        ghosts: s.ghosts, energizers: s.energizers, fruit: s.fruit, dots: s.dots, level: g.level, score: g.score, lives: g.lives };
    }
    // Maze distances from legal tile k to everything, not passing through tile `block`.
    _bfs(k, block) {
      const dist = this._dist, q = this._queue;
      dist.fill(-1); dist[k] = 0; let head = 0, tail = 0; q[tail++] = k;
      if (block >= 0) dist[block] = -2;
      while (head < tail) {
        const c = q[head++], dc = dist[c];
        if (dc >= PATH_MAX) continue;
        for (let a = 0; a < 4; a++) { const n = nbr[c * 4 + a]; if (n >= 0 && dist[n] === -1) { dist[n] = dc + 1; q[tail++] = n; } }
      }
      return dist;
    }
    observe(out = this._obs) {
      const g = this.game, p = g.pac, o = out;
      o.fill(0);
      const px = p.x >> 3, py = p.y >> 3;
      const danger = (h) => (h.state === ACTIVE || h.state === LEAVING) && !h.frightened;
      const blue = (h) => h.state === ACTIVE && h.frightened;
      // window
      const put = (c, x, y, v = 1) => {
        let wx = x - px + HALF;
        const wy = y - py + HALF;
        if (wy < 0 || wy >= WIN) return;
        if ((wx < 0 || wx >= WIN) && y === TUNNEL_ROW) wx += wx < 0 ? COLS : -COLS; // seen through the tunnel
        if (wx < 0 || wx >= WIN) return;
        o[(c * WIN + wy) * WIN + wx] = v;
      };
      for (let wy = 0; wy < WIN; wy++) for (let wx = 0; wx < WIN; wx++) {
        const x = px + wx - HALF, y = py + wy - HALF;
        if (!walkable(x, y)) { o[wy * WIN + wx] = 1; continue; }
        const t = g.dots[y * COLS + wrapX(x)];
        if (t === 1) o[(WIN + wy) * WIN + wx] = 1; else if (t === 2) o[(2 * WIN + wy) * WIN + wx] = 1;
      }
      for (const h of g.ghosts) {
        const hx = h.x >> 3, hy = h.y >> 3;
        if (danger(h)) { put(3, hx, hy); put(4, hx + DX[h.dir], hy + DY[h.dir]); }
        else if (blue(h)) put(5, hx, hy);
      }
      if (g.fruit) put(6, 13, 20);
      // paths
      const here = tileIndex[py * COLS + px];
      let off = OBS.window;
      const near = (d) => (d < 0 ? 0 : Math.max(0, 1 - d / 10)), far = (d) => (d < 0 ? 0 : Math.max(0, 1 - d / PATH_MAX));
      for (let a = 0; a < 4; a++, off += PATH_F) {
        const n = here >= 0 ? nbr[here * 4 + a] : -1;
        if (n < 0) continue;
        o[off] = 1;
        const dist = this._bfs(n, here);
        let dDot = -1, dEn = -1;
        for (let k = 0; k < NLEGAL; k++) {
          const dk = dist[k]; if (dk < 0) continue;
          const t = g.dots[legalTiles[k]];
          if (t === 1 && (dDot < 0 || dk < dDot)) dDot = dk;
          if (t === 2 && (dEn < 0 || dk < dEn)) dEn = dk;
        }
        const at = (x, y) => { const k = tileIndex[y * COLS + x]; return k >= 0 && dist[k] >= 0 ? dist[k] + 1 : -1; };
        const put2 = (i, d) => { o[off + i] = near(d); o[off + i + 1] = far(d); };
        put2(1, dDot < 0 ? -1 : dDot + 1); put2(3, dEn < 0 ? -1 : dEn + 1);
        if (g.fruit) put2(5, at(13, 20));
        for (const h of g.ghosts) {
          const d = at(h.x >> 3, h.y >> 3);
          if (danger(h)) put2(7 + 2 * h.id, d); else if (blue(h)) put2(15 + 2 * h.id, d);
        }
      }
      // global
      let k = OBS.window + OBS.paths;
      o[k + ACTION_OF_DIR[p.dir]] = 1; k += 4;
      o[k++] = p.x / 223; o[k++] = p.y / 287; o[k++] = p.moving ? 1 : 0;
      o[k++] = Math.min(1, g.frightTimer / 360); o[k++] = g.flashing ? 1 : 0;
      o[k++] = g.scatter ? 1 : 0; o[k++] = g.modeTimer === Infinity ? 1 : Math.min(1, g.modeTimer / 1200);
      for (const h of g.ghosts) {
        const s = h.state === HOUSE || h.state === LEAVING ? (h.frightened ? 2 : 0) : h.state === ACTIVE ? (h.frightened ? 2 : 1) : 3;
        o[k + s] = 1; k += 4;
      }
      for (const h of g.ghosts) { o[k++] = ((h.x >> 3) - px + 28) / 56; o[k++] = ((h.y >> 3) - py + 36) / 72; }
      const e = g.elroy(); o[k++] = e >= 1 ? 1 : 0; o[k++] = e === 2 ? 1 : 0;
      o[k + (g.level === 1 ? 0 : g.level <= 4 ? 1 : g.level <= 20 ? 2 : 3)] = 1; k += 4;
      o[k++] = g.dotsLeft / NDOTS; o[k++] = Math.min(1, g.lives / 5); o[k++] = g.fruit ? 1 : 0; o[k++] = Math.min(1, g.chain / 4);
      // dot map
      k = OBS.window + OBS.paths + OBS.global;
      for (let by = 0; by < DOTMAP_H; by++) for (let bx = 0; bx < DOTMAP_W; bx++) {
        let c = 0;
        for (let y = TOP + by * 4; y < TOP + by * 4 + 4 && y < ROWS; y++) for (let x = bx * 4; x < bx * 4 + 4; x++) if (g.dots[y * COLS + x]) c++;
        o[k + by * DOTMAP_W + bx] = Math.min(1, c / 8);
      }
      return o;
    }
    // Moves that change anything here: open directions (the stick into a wall just keeps going).
    openActions() {
      const p = this.game.pac, px = p.x >> 3, py = p.y >> 3, out = [];
      for (let a = 0; a < 4; a++) { const d = ACTION_DIRS[a]; if (walkable(px + DX[d], py + DY[d]) || d === REV[p.dir]) out.push(a); }
      return out;
    }
  }

  return {
    Game, ArcadeEnv, levelSpec, makeXorshift, MAZE, COLS, ROWS, TOP, TUNNEL_ROW, NDOTS, cellAt, walkable, isRedZone, isTunnel, legal,
    UP, LEFT, DOWN, RIGHT, DX, DY, REV, CLOCKWISE, DIR_NAMES, BLINKY, PINKY, INKY, CLYDE, GHOST_NAMES, SCATTER, EYES_TARGET,
    HOUSE, LEAVING, ACTIVE, EYES, ENTERING, STATE_NAMES, EXIT_X, EXIT_Y, HOUSE_Y, HOME_X, PAC_START, EYES_SPEED, HOUSE_SPEED,
    READY_FRAMES, GHOST_EATEN_FRAMES, DYING_FRAMES, LEVEL_CLEAR_FRAMES, FLASH_FRAMES, EXTRA_LIFE_AT, DOTS0, OPEN, WALL, DOOR,
    ACTION_DIRS, ACTION_OF_DIR, ACTION_NAMES, OBS, WCH, WIN, NLEGAL, legalTiles, tileIndex, ahead,
  };
})();
if (typeof module !== 'undefined') module.exports = Arcade;
