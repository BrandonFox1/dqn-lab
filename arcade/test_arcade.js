// Rule-by-rule tests for arcade.js. Each test names the SPEC.md rule(s) it checks.
const A = require('./arcade.js');
const assert = require('assert');
let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ok  ', name); }
  catch (e) { failed++; console.log('  FAIL', name, '\n       ', e.message); }
}
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b} (tol ${tol})`);
const { UP, LEFT, DOWN, RIGHT, BLINKY, PINKY, INKY, CLYDE, HOUSE, LEAVING, ACTIVE, EYES, ENTERING, DX, DY, REV } = A;
const cx = (tx) => tx * 8 + 3, cy = (ty) => ty * 8 + 4; // 1.3

// ------------------------------------------------------------- helpers
const game = (opts = {}) => new A.Game(Object.assign({ pauses: false }, opts));
const noCollide = (g) => { g._collide = () => {}; return g; };
const noRelease = (g) => { g._checkHouse = () => {}; g.spec = Object.assign({}, g.spec, { dotTimer: Infinity }); return g; };
function placePac(g, x, y, dir) { Object.assign(g.pac, { x, y, dir, acc: 0, stall: 0 }); }
function placeGhost(g, id, x, y, dir, state = ACTIVE, frightened = false) {
  const h = g.ghosts[id];
  Object.assign(h, { x, y, tx: x >> 3, ty: y >> 3, dir, turnDir: dir, state, acc: 0, pendingReverse: false, frightened });
  h.nextTurnDir = g._decide(h, h.tx + DX[dir], h.ty + DY[dir], dir);
  return h;
}
function clearDots(g, pred) {
  for (let i = 0; i < g.dots.length; i++) if (g.dots[i] && pred(i % 28, (i / 28) | 0, g.dots[i])) { g.dots[i] = 0; g.dotsLeft--; }
}
const dotTiles = () => { const out = []; A.DOTS0.forEach((d, i) => { if (d === 1) out.push([i % 28, (i / 28) | 0]); }); return out; };
function ghostPixels(g, id, frames) { // path length a ghost covers in `frames` frames
  const h = g.ghosts[id]; let moved = 0;
  for (let f = 0; f < frames; f++) {
    const x = h.x, y = h.y; g.step(null);
    let dx = Math.abs(h.x - x); if (dx > 100) dx = 224 - dx;
    moved += dx + Math.abs(h.y - y);
  }
  return moved;
}

// ------------------------------------------------------ 1-3 geometry
test('2.1 maze is 28x31 with 240 dots + 4 energizers, mirror-symmetric, 300 legal tiles', () => {
  assert.strictEqual(A.MAZE.length, 31);
  for (const r of A.MAZE) { assert.strictEqual(r.length, 28); assert.strictEqual(r, [...r].reverse().join('')); }
  let dots = 0, en = 0; for (const d of A.DOTS0) { if (d === 1) dots++; if (d === 2) en++; }
  assert.deepStrictEqual([dots, en], [240, 4]);
  assert.strictEqual(A.NLEGAL, 300);
});

test('2.2 energizers sit at (1,6), (26,6), (1,26), (26,26)', () => {
  const at = []; A.DOTS0.forEach((d, i) => { if (d === 2) at.push([i % 28, (i / 28) | 0]); });
  assert.deepStrictEqual(at, [[1, 6], [26, 6], [1, 26], [26, 26]]);
});

test('2.3 only the tunnel row wraps around', () => {
  assert.ok(A.walkable(-1, 17) && A.walkable(28, 17));
  assert.ok(!A.walkable(-1, 8) && !A.walkable(28, 26));
  const g = noRelease(noCollide(game()));
  placePac(g, 2, cy(17), LEFT);
  for (let f = 0; f < 4; f++) g.step(LEFT);
  assert.strictEqual(g.pac.x, 222, 'left edge wraps to the right edge');
});

test('2.6 the house door is a wall for Pac-Man', () => {
  const g = noRelease(noCollide(game()));
  placePac(g, cx(13), cy(14) - 3, DOWN);
  for (let f = 0; f < 20; f++) g.step(DOWN);
  assert.deepStrictEqual([g.pac.x, g.pac.y, g.pac.moving], [cx(13), cy(14), false]);
});

test('3.1-3.2 start positions, directions and states', () => {
  const g = game();
  assert.deepStrictEqual([g.pac.x, g.pac.y, g.pac.dir], [111, 212, LEFT]);
  const want = [[111, 116, LEFT, ACTIVE], [111, 140, DOWN, HOUSE], [95, 140, UP, HOUSE], [127, 140, UP, HOUSE]];
  g.ghosts.forEach((h, i) => assert.deepStrictEqual([h.x, h.y, h.dir, h.state], want[i], h.name));
  assert.ok(g.scatter, 'play starts in scatter mode (8.2)');
  assert.strictEqual(g.lives, 3);
});

// ------------------------------------------------------------ 4 speed
function pacSpeed(level, fright, dots) {
  const g = noRelease(noCollide(game({ level })));
  if (!dots) clearDots(g, (x, y) => y === 32);
  g.dotsLeft = 244;
  if (fright) g.frightTimer = 1e9;
  placePac(g, cx(26), cy(32), LEFT);
  let frames = 0;
  while (g.pac.x > cx(6)) { g.step(LEFT); frames++; }
  return (cx(26) - g.pac.x) / frames / 1.25 * 100;
}

test('4.1 Pac-Man speed per level matches Table A.1, including its "with dots" column', () => {
  for (const [L, fr, norm, withDots] of [[1, 0, 80, 71], [1, 1, 90, 79], [2, 0, 90, 79], [2, 1, 95, 83], [5, 0, 100, 87], [5, 1, 100, 87], [21, 0, 90, 79]]) {
    close(pacSpeed(L, fr, false), norm, 0.7, `L${L}${fr ? ' fright' : ''} empty corridor:`);
    close(pacSpeed(L, fr, true), withDots, 1.0, `L${L}${fr ? ' fright' : ''} eating dots:`);
  }
});

test('4.3 a dot stops Pac-Man for exactly 1 frame, an energizer for 3', () => {
  for (const [kind, still] of [[1, 1], [2, 3]]) {
    const g = noRelease(noCollide(game()));
    clearDots(g, (x, y) => y === 32); g.dotsLeft = 244;
    g.dots[32 * 28 + 20] = kind;
    placePac(g, cx(22), cy(32), LEFT);
    let stopped = 0, x = g.pac.x;
    for (let f = 0; f < 40; f++) { g.step(LEFT); if (g.pac.x === x) stopped++; x = g.pac.x; }
    assert.strictEqual(stopped, still, kind === 1 ? 'dot' : 'energizer');
  }
});

test('4.4 ghost normal, frightened and tunnel speeds per level', () => {
  for (const [L, norm, fr, tun] of [[1, 75, 50, 40], [2, 85, 55, 45], [5, 95, 60, 50], [21, 95, 60, 50]]) {
    let g = noRelease(noCollide(game({ level: L })));
    placeGhost(g, BLINKY, cx(26), cy(8), LEFT);
    assert.strictEqual(ghostPixels(g, BLINKY, 160), 160 * norm / 80, `L${L} normal`);
    if (L < 17) {
      g = noRelease(noCollide(game({ level: L }))); g.frightTimer = 1e6;
      placeGhost(g, BLINKY, cx(26), cy(8), LEFT, ACTIVE, true);
      assert.strictEqual(ghostPixels(g, BLINKY, 160), 160 * fr / 80, `L${L} frightened`);
    }
    g = noRelease(noCollide(game({ level: L })));
    placeGhost(g, BLINKY, cx(4), cy(17), LEFT);
    assert.strictEqual(ghostPixels(g, BLINKY, 32), 32 * tun / 80, `L${L} tunnel`);
  }
});

test('4.6 the tunnel slows frightened ghosts too; 2.4 the slow zone is x<=4 and x>=23', () => {
  const g = noRelease(noCollide(game())); g.frightTimer = 1e6;
  const h = placeGhost(g, BLINKY, cx(4), cy(17), LEFT, ACTIVE, true);
  assert.strictEqual(g._ghostSpeed(h), 40);
  for (const [x, slow] of [[4, true], [5, false], [22, false], [23, true]]) assert.strictEqual(A.isTunnel(x, 17), slow, `x=${x}`);
});

test('4.7 eyes move 1.5 px/frame; ghosts inside the house 0.5 px/frame', () => {
  const g = noRelease(noCollide(game()));
  const h = placeGhost(g, BLINKY, cx(26), cy(8), LEFT); h.state = EYES;
  h.nextTurnDir = g._decide(h, h.tx - 1, h.ty, LEFT);
  assert.strictEqual(ghostPixels(g, BLINKY, 16), 24);
  assert.strictEqual(ghostPixels(g, INKY, 16), 8, 'Inky bobbing in the house');
});

// ------------------------------------------------------- 5 Pac-Man
test('5.2 Pac-Man reverses on the very next frame', () => {
  const g = noRelease(noCollide(game())); clearDots(g, (x, y) => y === 32);
  placePac(g, cx(10), cy(32), LEFT);
  g.step(LEFT); assert.strictEqual(g.pac.x, cx(10) - 1);
  g.step(RIGHT); assert.strictEqual(g.pac.x, cx(10));
});

// The Dossier's cornering figure: pre-turn pixels are 3 approaching from the left, 4 from the right,
// 4 from above and 3 from below. We hold the stick early and see where the turn starts at tile (6,8).
function preTurnPixels(startX, startY, dir, hold) {
  const g = noRelease(noCollide(game())); clearDots(g, () => true); g.dotsLeft = 244;
  placePac(g, startX, startY, dir);
  for (let f = 0; f < 60; f++) {
    const x = g.pac.x, y = g.pac.y;
    g.step(hold);
    if (g.pac.dir === hold) return Math.abs(DX[dir] !== 0 ? cx(6) - x : cy(8) - y);
  }
  throw new Error('never turned');
}
test('1.3 + 5.3 earliest pre-turn is 3 px from the left, 4 from the right, 4 from above, 3 from below', () => {
  assert.strictEqual(preTurnPixels(cx(3), cy(8), RIGHT, UP), 3);
  assert.strictEqual(preTurnPixels(cx(9), cy(8), LEFT, UP), 4);
  assert.strictEqual(preTurnPixels(cx(6), cy(5), DOWN, RIGHT), 4);
  assert.strictEqual(preTurnPixels(cx(6), cy(11), UP, RIGHT), 3);
});

test('5.4 cornering gains one frame per pixel of pre-turn (3 from the left, 4 from the right)', () => {
  const run = (startX, dir, early) => {
    const g = noRelease(noCollide(game())); clearDots(g, () => true); g.dotsLeft = 244;
    placePac(g, startX, cy(8), dir);
    let frames = 0, turned = false;
    while (g.pac.y > cy(5)) {
      if (g.pac.x === cx(6)) turned = true;
      g.step(early || turned ? UP : dir); frames++;
    }
    return frames;
  };
  assert.strictEqual(run(cx(3), RIGHT, false) - run(cx(3), RIGHT, true), 3);
  assert.strictEqual(run(cx(9), LEFT, false) - run(cx(9), LEFT, true), 4);
});

test('5.5 Pac-Man stops at the center of the tile before a wall; the stick into a wall changes nothing', () => {
  const g = noRelease(noCollide(game())); clearDots(g, (x, y) => y === 32);
  placePac(g, cx(4), cy(32), LEFT);
  for (let f = 0; f < 40; f++) g.step(LEFT);
  assert.deepStrictEqual([g.pac.x, g.pac.moving], [cx(1), false]);
  placePac(g, cx(20), cy(32), LEFT);
  for (let f = 0; f < 5; f++) g.step(DOWN);
  assert.deepStrictEqual([g.pac.dir, g.pac.x], [LEFT, cx(20) - 5]);
});

// ------------------------------------------------ 6 ghost pathfinding
test('6.1 a ghost decides one tile ahead and turns exactly at the center of the next tile', () => {
  const g = noRelease(noCollide(game()));
  const h = g.ghosts[BLINKY];
  let target = [6, 0];
  h.targetOf = null; g.targetOf = (gh) => (gh.id === BLINKY ? target : A.Game.prototype.targetOf.call(g, gh));
  placeGhost(g, BLINKY, cx(9), cy(8), LEFT);
  g.frightTimer = 1e6; // freeze the mode timer so no reversal interferes
  while (h.tx !== 7) g.step(null);
  assert.strictEqual(h.nextTurnDir, UP, 'decided for (6,8) on entering (7,8)');
  target = [6, 30]; // too late: the decision for (6,8) is already made
  let turnedAt = null;
  for (let f = 0; f < 40 && turnedAt === null; f++) { const x = h.x; g.step(null); if (h.y < cy(8)) turnedAt = x; }
  assert.strictEqual(turnedAt, cx(6), 'turned at the center pixel');
  assert.strictEqual(h.dir, UP);
});

test('6.2 a ghost never chooses to reverse, never picks a wall (whole games)', () => {
  let checks = 0;
  for (let seed = 1; seed <= 20; seed++) {
    const g = game({ seed }); const orig = g._decide.bind(g);
    g._decide = (h, bx, by, e) => {
      const d = orig(h, bx, by, e); checks++;
      assert.notStrictEqual(d, REV[e], `${h.name} reversed by choice`);
      assert.ok(A.walkable(bx + DX[d], by + DY[d]), `${h.name} chose a wall`);
      return d;
    };
    const rng = A.makeXorshift(seed); let joy = LEFT;
    for (let f = 0; f < 30000 && !g.over; f++) { if (rng() % 12 === 0) joy = rng() % 4; g.step(joy); }
  }
  assert.ok(checks > 5000, `only ${checks} decisions checked`);
});

test('6.3 ties go up, then left, then down, then right', () => {
  const g = game(), h = g.ghosts[BLINKY];
  const choose = (target, e) => { g.targetOf = () => target; return g._decide(h, 6, 8, e); };
  assert.strictEqual(choose([5, 7], LEFT), UP, 'up beats left');
  assert.strictEqual(choose([5, 9], LEFT), LEFT, 'left beats down');
  assert.strictEqual(choose([7, 9], DOWN), DOWN, 'down beats right');
  assert.strictEqual(choose([6, 30], LEFT), DOWN, 'plain nearest');
});

test('6.4 frightened: random first choice, then clockwise; reseeded at every level start and life', () => {
  const g = game(), h = g.ghosts[BLINKY]; h.state = ACTIVE; h.frightened = true;
  const pick = (r, bx, by, e) => { g.frightRng = () => r; return g._decide(h, bx, by, e); };
  assert.strictEqual(pick(UP, 6, 8, LEFT), UP);
  assert.strictEqual(pick(RIGHT, 6, 8, LEFT), DOWN, 'right is the reverse, so clockwise to down');
  assert.strictEqual(pick(UP, 4, 8, LEFT), LEFT, 'up and down are walls, right is the reverse');
  const g2 = game(); const s0 = g2.frightRng.getState();
  g2.frightRng(); g2.frightRng(); assert.notStrictEqual(g2.frightRng.getState(), s0);
  g2._pacDies(g2.ghosts[0]); g2.step(); assert.strictEqual(g2.frightRng.getState(), s0, 'after a lost life');
  g2.frightRng(); g2.startLevel(2); assert.strictEqual(g2.frightRng.getState(), s0, 'at a new level');
});

test('6.5 roaming ghosts stay on tile center lines (they never cut corners)', () => {
  const g = game({ seed: 9 }); const rng = A.makeXorshift(4); let joy = LEFT, frames = 0;
  for (let f = 0; f < 40000 && !g.over; f++) {
    if (rng() % 10 === 0) joy = rng() % 4;
    g.step(joy); frames++;
    for (const h of g.ghosts) {
      if (h.state !== ACTIVE && h.state !== EYES) continue;
      if (DX[h.dir] !== 0) assert.strictEqual(h.y & 7, 4, `${h.name} off its row at frame ${g.frame}`);
      else assert.strictEqual(h.x & 7, 3, `${h.name} off its column at frame ${g.frame}`);
    }
  }
  assert.ok(frames > 1000);
});

test('2.5 red zones: no turning up in scatter or chase; frightened ghosts and eyes may', () => {
  const g = game(), h = g.ghosts[BLINKY];
  g.targetOf = () => [12, 0];
  h.state = ACTIVE; h.frightened = false;
  assert.strictEqual(g._decide(h, 12, 14, LEFT), LEFT, 'chase/scatter: up is forbidden at (12,14)');
  assert.strictEqual(g._decide(h, 12, 26, LEFT), LEFT, 'and at (12,26)');
  assert.strictEqual(g._decide(h, 12, 8, LEFT), UP, 'outside the zone the same target sends it up');
  h.state = EYES; assert.strictEqual(g._decide(h, 12, 14, LEFT), UP, 'eyes ignore red zones');
  h.state = ACTIVE; h.frightened = true; g.frightRng = () => UP;
  assert.strictEqual(g._decide(h, 12, 14, LEFT), UP, 'frightened ghosts ignore red zones');
});

// ------------------------------------------------------------ 7 targets
test('7.1 scatter targets are the four fixed corners outside the maze', () => {
  const g = game(); g.ghosts.forEach((h) => { h.state = ACTIVE; });
  assert.deepStrictEqual(g.ghosts.map((h) => g.targetOf(h)), [[25, 0], [2, 0], [27, 35], [0, 35]]);
});

test('7.2-7.4 chase targets: Blinky direct, Pinky 4 ahead, Inky doubled from Blinky, both with the up bug', () => {
  const g = game(); g.modeIndex = 1; // chase
  g.ghosts.forEach((h) => { h.state = ACTIVE; });
  placeGhost(g, BLINKY, cx(20), cy(20), LEFT);
  const tgt = (id, dir) => { placePac(g, cx(13), cy(26), dir); return g.targetOf(g.ghosts[id]); };
  assert.deepStrictEqual(tgt(BLINKY, LEFT), [13, 26]);
  assert.deepStrictEqual(tgt(PINKY, LEFT), [9, 26]);
  assert.deepStrictEqual(tgt(PINKY, RIGHT), [17, 26]);
  assert.deepStrictEqual(tgt(PINKY, DOWN), [13, 30]);
  assert.deepStrictEqual(tgt(PINKY, UP), [9, 22], 'facing up: 4 up and 4 left');
  assert.deepStrictEqual(tgt(INKY, LEFT), [2 * 11 - 20, 2 * 26 - 20]);
  assert.deepStrictEqual(tgt(INKY, UP), [2 * 11 - 20, 2 * 24 - 20], 'facing up: 2 up and 2 left before doubling');
});

test('7.5 Clyde chases from 8 tiles or more, else heads for his corner', () => {
  const g = game(); g.modeIndex = 1;
  const c = g.ghosts[CLYDE]; c.state = ACTIVE;
  placePac(g, cx(13), cy(26), LEFT);
  placeGhost(g, CLYDE, cx(13), cy(18), LEFT); assert.deepStrictEqual(g.targetOf(c), [13, 26], 'exactly 8 away');
  placeGhost(g, CLYDE, cx(13), cy(19), LEFT); assert.deepStrictEqual(g.targetOf(c), [0, 35], '7 away');
});

test('7.6 + 9.5 eaten ghosts: eyes aim at (13,14), enter at the door, revive un-frightened, Blinky leaves at once', () => {
  const g = noCollide(game()); g.frightTimer = 1e6; g._energize(); g.frightTimer = 1e6;
  const h = placeGhost(g, BLINKY, cx(17), cy(14), LEFT, ACTIVE, true);
  placePac(g, cx(17), cy(14), LEFT);
  A.Game.prototype._collide.call(g);
  assert.strictEqual(h.state, EYES);
  assert.deepStrictEqual(g.targetOf(h), [13, 14]);
  const seen = new Set(); let enteredAt = null;
  for (let f = 0; f < 400 && h.state !== ACTIVE; f++) {
    const ev = g.step(null); seen.add(h.state);
    if (h.state === ENTERING && enteredAt === null) enteredAt = [h.x, h.y];
    if (ev.some((e) => e.type === 'revived' && e.ghost === BLINKY)) assert.ok(!h.frightened && g.frightTimer > 0);
  }
  assert.deepStrictEqual(enteredAt, [111, 116]);
  assert.ok(seen.has(ENTERING) && seen.has(LEAVING) && h.state === ACTIVE, [...seen].join(','));
  assert.ok(!h.frightened, 'revived ghosts stay dangerous while the fright timer runs');
});

// ------------------------------------------------------ 8 scatter/chase
function modeFrames(level, until) {
  const g = noCollide(game({ level }));
  clearDots(g, (x, y, t) => t === 2); g.dotsLeft = 244;
  const frames = [];
  while (g.frame < until) for (const e of g.step(null)) if (e.type === 'mode') frames.push(g.frame);
  return frames;
}
test('8.1 scatter/chase schedule per level, in frames (7/20/7/20/5/20/5 s on level 1, etc.)', () => {
  assert.deepStrictEqual(modeFrames(1, 6000), [420, 1620, 2040, 3240, 3540, 4740, 5040]);
  assert.deepStrictEqual(modeFrames(2, 66000), [420, 1620, 2040, 3240, 3540, 65520, 65521]);
  assert.deepStrictEqual(modeFrames(5, 66000), [300, 1500, 1800, 3000, 3300, 65520, 65521]);
});

test('8.2 the schedule restarts after a lost life and at a new level', () => {
  const g = noCollide(game());
  for (let f = 0; f < 1000; f++) g.step(null);
  assert.ok(!g.scatter);
  g._pacDies(g.ghosts[0]); g.step();
  assert.ok(g.scatter && g.modeTimer === 420 && g.modeIndex === 0);
  for (let f = 0; f < 1000; f++) g.step(null);
  g.startLevel(5); assert.ok(g.scatter && g.modeTimer === 300);
});

test('8.3 frightened time pauses the mode timer', () => {
  const g = noCollide(game());
  let first = null;
  for (let f = 1; f <= 2000 && first === null; f++) {
    if (f === 100) g._energize();
    for (const e of g.step(null)) if (e.type === 'mode') first = g.frame;
  }
  assert.strictEqual(first, 420 + 360);
});

test('8.4-8.5 mode changes reverse ghosts at their next tile; the end of fright does not', () => {
  const g = noRelease(noCollide(game()));
  const h = placeGhost(g, BLINKY, cx(20), cy(8), LEFT);
  g.step(null); g.step(null);
  g._nextMode();
  assert.strictEqual(h.dir, LEFT, 'not immediately');
  let frames = 0;
  while (h.dir === LEFT) { g.step(null); frames++; }
  assert.strictEqual(h.dir, RIGHT);
  assert.ok(frames <= 9, `reversed after ${frames} frames`);
  assert.ok(h.x % 8 === 0 || h.x % 8 === 7, 'reversal happens at a tile boundary');
  // fright ending: no reversal
  g.frightTimer = 2; g.ghosts.forEach((q) => { q.frightened = true; q.pendingReverse = false; });
  g.step(null); g.step(null);
  assert.ok(g.ghosts.every((q) => !q.frightened && !q.pendingReverse));
});

// ------------------------------------------------------------- 9 fright
test('9.1-9.2 fright time by level; levels without fright only reverse', () => {
  const secs = [6, 5, 4, 3, 2, 5, 2, 2, 1, 5, 2, 1, 1, 3, 1, 1, 0, 1, 0, 0, 0, 0];
  secs.forEach((s, i) => {
    const g = game({ level: i + 1 });
    g._energize();
    assert.strictEqual(g.frightTimer, s * 60, `level ${i + 1}`);
    assert.strictEqual(g.ghosts[BLINKY].frightened, s > 0);
    assert.ok(g.ghosts[BLINKY].pendingReverse, 'always reverses');
  });
  assert.deepStrictEqual([9, 12, 13, 15, 16, 18].map((L) => A.levelSpec(L).flashes), [3, 3, 3, 3, 3, 3]);
  assert.deepStrictEqual([1, 2, 10, 14].map((L) => A.levelSpec(L).flashes), [5, 5, 5, 5]);
});

test('9.4 ghosts from one energizer are worth 200, 400, 800, 1600; a new energizer resets the chain', () => {
  const g = noRelease(game()); g._energize();
  const pts = [];
  for (const id of [BLINKY, PINKY, INKY, CLYDE]) {
    placeGhost(g, id, cx(6), cy(8), LEFT, ACTIVE, true); placePac(g, cx(6), cy(8), LEFT);
    const s = g.score; g._collide(); pts.push(g.score - s);
  }
  assert.deepStrictEqual(pts, [200, 400, 800, 1600]);
  g._energize(); placeGhost(g, BLINKY, cx(6), cy(8), LEFT, ACTIVE, true);
  const s = g.score; g._collide(); assert.strictEqual(g.score - s, 200);
});

test('9.6 ghosts waiting in the house turn blue and come out blue', () => {
  const g = noCollide(game());
  g._energize();
  assert.ok(g.ghosts[INKY].frightened && g.ghosts[INKY].state === HOUSE);
  g._release(g.ghosts[INKY], 'test');
  for (let f = 0; f < 200 && g.ghosts[INKY].state !== ACTIVE; f++) g.step(null);
  assert.ok(g.ghosts[INKY].state === ACTIVE && g.ghosts[INKY].frightened);
});

// ------------------------------------------------------- 10 ghost house
function releaseDots(level) {
  const g = game({ level }); g.spec = Object.assign({}, g.spec, { dotTimer: Infinity });
  const at = {}; let eaten = 0;
  const check = () => { g._checkHouse(); for (const id of [PINKY, INKY, CLYDE]) if (at[id] === undefined && g.ghosts[id].state !== HOUSE) at[id] = eaten; };
  check(); check(); check();
  for (const [x, y] of dotTiles()) { if (Object.keys(at).length === 3) break; g._pacEnter(x, y); eaten++; check(); }
  return [at[PINKY], at[INKY], at[CLYDE]];
}
test('10.2-10.3 dot limits: level 1 Inky 30 then Clyde 60 more; level 2 Clyde 50; level 3+ nobody waits', () => {
  assert.deepStrictEqual(releaseDots(1), [0, 30, 90]);
  assert.deepStrictEqual(releaseDots(2), [0, 0, 50]);
  assert.deepStrictEqual(releaseDots(3), [0, 0, 0]);
  assert.deepStrictEqual(releaseDots(21), [0, 0, 0]);
});

test('10.4 after a lost life the global counter frees Pinky at 7, Inky at 17, and switches off at 32', () => {
  const g = game({ level: 3 }); g.spec = Object.assign({}, g.spec, { dotTimer: Infinity });
  g._pacDies(g.ghosts[0]); g.step();
  assert.ok(g.globalActive);
  const at = {}; let eaten = 0;
  for (const [x, y] of dotTiles()) {
    if (eaten >= 40) break;
    g._pacEnter(x, y); eaten++; g._checkHouse();
    for (const id of [PINKY, INKY, CLYDE]) if (at[id] === undefined && g.ghosts[id].state !== HOUSE) at[id] = eaten;
    if (eaten === 31) assert.ok(g.globalActive);
    if (eaten === 32) assert.ok(!g.globalActive, 'Clyde was inside at 32');
    g._checkHouse(); for (const id of [PINKY, INKY, CLYDE]) if (at[id] === undefined && g.ghosts[id].state !== HOUSE) at[id] = eaten;
  }
  assert.deepStrictEqual([at[PINKY], at[INKY], at[CLYDE]], [7, 17, 32]);
  // The Dossier's trap: if Clyde left before 32, the counter never switches off and cannot free anyone again.
  const t = game({ level: 3 }); t.spec = Object.assign({}, t.spec, { dotTimer: Infinity });
  t._pacDies(t.ghosts[0]); t.step();
  t._release(t.ghosts[CLYDE], 'timer');
  let n = 0; for (const [x, y] of dotTiles()) { if (n++ >= 40) break; t._pacEnter(x, y); t._checkHouse(); }
  assert.ok(t.globalActive && t.globalCounter === 40);
  t.ghosts[PINKY].state = HOUSE; t._checkHouse();
  assert.strictEqual(t.ghosts[PINKY].state, HOUSE, 'a returning Pinky stays stuck');
});

test('10.5 the no-dot timer frees the next ghost after 4 s (3 s from level 5)', () => {
  for (const [level, limit] of [[1, 240], [5, 180]]) {
    const g = noCollide(game({ level }));
    clearDots(g, () => true); g.dotsLeft = 244;
    g.spec = Object.assign({}, g.spec, { dotLimits: [0, 0, 100, 100] });
    const frames = [];
    for (let f = 0; f < 3 * limit && frames.length < 2; f++) for (const e of g.step(null)) if (e.type === 'release' && e.why === 'timer') frames.push(g.frame);
    assert.deepStrictEqual(frames, [limit, 2 * limit], `level ${level}`);
  }
});

test('10.6 ghosts leave heading left, or right if the mode changed while they were inside', () => {
  const exitDir = (changeMode) => {
    const g = noCollide(game());
    if (changeMode) g._nextMode();
    g._release(g.ghosts[INKY], 'test');
    for (let f = 0; f < 300 && g.ghosts[INKY].state !== ACTIVE; f++) g.step(null);
    return g.ghosts[INKY].dir;
  };
  assert.strictEqual(exitDir(false), LEFT);
  assert.strictEqual(exitDir(true), RIGHT);
});

// ------------------------------------------------------ 11 collisions
test('11.1 sharing a tile kills Pac-Man; swapping tiles in the same frame passes through', () => {
  const run = (ghostX) => {
    const g = noRelease(game()); clearDots(g, (x, y) => y === 8); g.frightTimer = 0;
    placePac(g, 87, cy(8), RIGHT); // last pixel of tile 10
    const h = placeGhost(g, BLINKY, ghostX, cy(8), LEFT); h.acc = 79; // moves on the next frame
    const ev = g.step(RIGHT).slice();
    return ev.some((e) => e.type === 'death');
  };
  assert.strictEqual(run(89), true, 'both land in tile 11');
  assert.strictEqual(run(88), false, 'they swap tiles 10 and 11: pass-through');
});

test('11.2 frightened ghosts are eaten; eyes are harmless', () => {
  const g = noRelease(game()); g._energize();
  placePac(g, cx(6), cy(8), LEFT);
  const h = placeGhost(g, BLINKY, cx(6), cy(8), LEFT, ACTIVE, true);
  g._collide(); assert.strictEqual(h.state, EYES);
  g.step(); // the 1-frame pause
  const lives = g.lives; g._collide(); assert.strictEqual(g.lives, lives, 'eyes do nothing');
});

// ----------------------------------------------------------- 12 Elroy
test('12.1-12.2 Cruise Elroy thresholds and speeds by level', () => {
  for (const [L, d1, s0, s1, s2] of [[1, 20, 75, 80, 85], [2, 30, 85, 90, 95], [5, 40, 95, 100, 105], [6, 50, 95, 100, 105], [12, 80, 95, 100, 105], [19, 120, 95, 100, 105]]) {
    const g = game({ level: L }), b = g.ghosts[BLINKY];
    placeGhost(g, BLINKY, cx(26), cy(8), LEFT);
    g.dotsLeft = d1 + 1; assert.deepStrictEqual([g.elroy(), g._ghostSpeed(b)], [0, s0], `L${L} above`);
    g.dotsLeft = d1; assert.deepStrictEqual([g.elroy(), g._ghostSpeed(b)], [1, s1], `L${L} Elroy 1`);
    g.dotsLeft = d1 / 2; assert.deepStrictEqual([g.elroy(), g._ghostSpeed(b)], [2, s2], `L${L} Elroy 2`);
    assert.strictEqual(g._ghostSpeed(g.ghosts[PINKY].state === HOUSE ? Object.assign({}, b, { id: PINKY }) : b), s0, 'only Blinky');
  }
});

test('12.3 as Elroy, Blinky targets Pac-Man even in scatter mode', () => {
  const g = game(); placePac(g, cx(6), cy(26), LEFT);
  assert.deepStrictEqual(g.targetOf(g.ghosts[BLINKY]), [25, 0]);
  g.dotsLeft = 20; assert.deepStrictEqual(g.targetOf(g.ghosts[BLINKY]), [6, 26]);
});

test('12.4 after a lost life Elroy waits until Clyde starts to leave', () => {
  const g = game(); g.dotsLeft = 5;
  assert.strictEqual(g.elroy(), 2);
  g._pacDies(g.ghosts[0]); g.step();
  assert.strictEqual(g.elroy(), 0, 'suspended');
  g._release(g.ghosts[CLYDE], 'test');
  assert.strictEqual(g.elroy(), 2, 'back once Clyde leaves');
});

// ------------------------------------------------------- 13 scoring
test('13.1-13.2 dots 10, energizers 50, 3 lives, one extra life at 10,000', () => {
  const g = noRelease(noCollide(game()));
  let s = g.score; g._pacEnter(1, 4); assert.strictEqual(g.score - s, 10);
  s = g.score; g._pacEnter(1, 6); assert.strictEqual(g.score - s, 50);
  g.score = 9990; g._pacEnter(2, 4);
  assert.deepStrictEqual([g.score, g.lives], [10000, 4]);
  g.score = 19995; g._pacEnter(3, 4); assert.strictEqual(g.lives, 4, 'only one extra life');
  const d = game(); for (let k = 0; k < 3; k++) { d._pacDies(d.ghosts[0]); d.step(); }
  assert.ok(d.over, 'game over after the third life');
});

test('13.3 clearing the maze starts the next level with fresh dots and faster speeds', () => {
  const g = noCollide(game());
  clearDots(g, (x, y) => !(x === 12 && y === 26));
  placePac(g, cx(13), cy(26), LEFT);
  let ev = [];
  for (let f = 0; f < 20 && !ev.some((e) => e.type === 'levelClear'); f++) ev = g.step(LEFT).slice();
  assert.ok(ev.some((e) => e.type === 'levelClear'));
  g.step();
  assert.deepStrictEqual([g.level, g.dotsLeft, g.spec.pac, g.pac.x, g.pac.y], [2, 244, 90, 111, 212]);
});

test('13.4 the game ends after level 255 (no split screen)', () => {
  const g = noCollide(game({ level: 255 }));
  clearDots(g, (x, y) => !(x === 12 && y === 26));
  placePac(g, cx(13), cy(26), LEFT);
  for (let f = 0; f < 30 && !g.over; f++) g.step(LEFT);
  assert.ok(g.over);
});

// ----------------------------------------------------------- 15 fruit
test('15.1-15.2 fruit shows after 70 and 170 dots, stays 9-10 s, and is worth the level value', () => {
  for (const [L, pts] of [[1, 100], [2, 300], [3, 500], [5, 700], [7, 1000], [9, 2000], [11, 3000], [13, 5000], [40, 5000]]) {
    const g = noRelease(noCollide(game({ level: L, seed: L })));
    const shows = []; let eaten = 0;
    for (const [x, y] of dotTiles()) { if (eaten === 171) break; const n = g.events.length; g._pacEnter(x, y); eaten++; if (g.fruit && g.events.slice(n).some((e) => e.type === 'fruit')) shows.push(eaten); }
    assert.deepStrictEqual(shows, [70, 170], `L${L}`);
    assert.ok(g.fruitTimer >= 540 && g.fruitTimer <= 600, `timer ${g.fruitTimer}`);
    placePac(g, cx(13), cy(20), LEFT);
    const s = g.score; A.Game.prototype._collide.call(g);
    assert.strictEqual(g.score - s, pts, `L${L} fruit value`);
  }
  const g = noRelease(noCollide(game())); g.fruit = 100; g.fruitTimer = 3;
  g.step(); g.step(); assert.ok(g.fruit); g.step(); assert.ok(!g.fruit, 'gone when the timer runs out');
});

// ------------------------------------------------------------ 14 pauses
test('14.1 eating a ghost freezes everything for 60 frames', () => {
  const g = new A.Game({ readyFrames: 1 }); g.step();
  g._energize();
  const h = placeGhost(g, PINKY, cx(6), cy(8), LEFT, ACTIVE, true); placePac(g, cx(6), cy(8), LEFT);
  g._collide();
  const before = JSON.stringify([g.pac, g.ghosts.map((q) => [q.x, q.y]), g.frightTimer]);
  for (let f = 0; f < 59; f++) g.step(LEFT);
  assert.strictEqual(JSON.stringify([g.pac, g.ghosts.map((q) => [q.x, q.y]), g.frightTimer]), before);
  g.step(LEFT); g.step(LEFT);
  assert.notStrictEqual(JSON.stringify([g.pac, g.ghosts.map((q) => [q.x, q.y])]), before, 'moving again');
  assert.strictEqual(h.state, EYES);
});

// ------------------------------------------------------- determinism
test('same seed + same inputs = same game; snapshot/restore replays exactly', () => {
  const play = (g, n, seed) => { const r = A.makeXorshift(seed), out = []; let joy = LEFT; for (let f = 0; f < n; f++) { if (r() % 9 === 0) joy = r() % 4; g.step(joy); out.push(g.hash()); } return out; };
  assert.deepStrictEqual(play(game({ seed: 5 }), 8000, 1), play(game({ seed: 5 }), 8000, 1));
  const g = game({ seed: 7 }); play(g, 3000, 2);
  const snap = g.snapshot(); const a = play(g, 3000, 3);
  g.restore(snap); const b = play(g, 3000, 3);
  assert.deepStrictEqual(a, b);
});

// ------------------------------------------------------------------ RL
test('env: one decision per new tile, points = score change, lost life resets before the next decision', () => {
  const env = new A.ArcadeEnv({ seed: 3 }); env.reset(3);
  const g = env.game; let lost = null;
  for (let k = 0; k < 3000 && !lost; k++) {
    const t0 = [g.pac.x >> 3, g.pac.y >> 3], s0 = g.score;
    const a = env.openActions()[k % env.openActions().length];
    const r = env.step(a);
    assert.strictEqual(r.points, g.score - s0);
    if (r.lifeLost) { lost = r; break; }
    if (r.frames < 64 && g.pac.moving) {
      const t1 = [g.pac.x >> 3, g.pac.y >> 3], dist = Math.abs(t1[0] - t0[0]) + Math.abs(t1[1] - t0[1]);
      assert.ok(dist === 1 || dist === 27, `moved ${dist} tiles in one decision`);
    }
  }
  assert.ok(lost, 'random play loses a life');
  assert.deepStrictEqual([g.pac.x, g.pac.y, g.phase], [111, 212, 'play'], 'next decision starts from the fresh start');
});

test('env: observation is 1768 values in [0,1] that see walls, dots, ghosts and open paths', () => {
  const env = new A.ArcadeEnv({ seed: 1 }); const o = env.reset(1);
  assert.strictEqual(o.length, 1768); assert.strictEqual(A.OBS.size, 1768);
  assert.ok(o.every((v) => v >= 0 && v <= 1 && Number.isFinite(v)));
  const W = A.WIN, at = (c, dx, dy) => o[(c * W + 7 + dy) * W + 7 + dx];
  assert.strictEqual(at(0, 0, 0), 0, 'Pac-Man stands on floor');
  assert.strictEqual(at(0, 0, 1), 1, 'wall below the start');
  assert.strictEqual(at(1, -1, 0), 1, 'dot to the left');
  assert.strictEqual(at(3, 0, -12), 0, 'Blinky is 12 rows up: outside the 15x15 window');
  const p = A.OBS.window; // paths: open flags for up, down, left, right
  assert.deepStrictEqual([0, 1, 2, 3].map((a) => o[p + a * 23]), [0, 0, 1, 1]);
  assert.ok(o[p + 2 * 23 + 1] > 0.8, 'a dot 1 step to the left');
  const env2 = new A.ArcadeEnv({ seed: 1 }); env2.reset(1);
  assert.deepStrictEqual(Array.from(env2.observe()), Array.from(o), 'deterministic');
});

// A simple greedy player that reads the path features: dots, energizers and blue ghosts good, ghosts bad.
function heuristic(o, rng) {
  let best = -1e9, arg = 0;
  for (let a = 0; a < 4; a++) {
    const b = A.OBS.window + a * 23;
    const danger = Math.max(o[b + 7], o[b + 9], o[b + 11], o[b + 13]), blue = Math.max(o[b + 15], o[b + 17], o[b + 19], o[b + 21]);
    const v = o[b] ? o[b + 1] + 0.6 * o[b + 3] + 2 * blue - 3 * danger + 0.01 * (rng() % 100) : -9;
    if (v > best) { best = v; arg = a; }
  }
  return arg;
}
test('env: the page (real pauses, frame by frame) and training (no pauses) make identical decisions', () => {
  const a = new A.ArcadeEnv({ seed: 8 }), b = new A.ArcadeEnv({ seed: 8, pauses: true });
  a.reset(8); b.reset(8);
  const rng = A.makeXorshift(99);
  let ghosts = 0, lost = 0, cleared = 0, k = 0;
  for (; k < 20000 && !a.game.over; k++) {
    const oa = a.observe(), ob = b.observe();
    for (let i = 0; i < oa.length; i++) if (oa[i] !== ob[i]) throw new Error(`observations differ at decision ${k}, index ${i}`);
    const act = heuristic(oa, rng);
    const ra = a.step(act);
    b.begin(act); let rb = null; while (rb === null) rb = b.tick();
    assert.deepStrictEqual([rb.points, rb.lifeLost, rb.cleared, rb.done, rb.ghosts], [ra.points, ra.lifeLost, ra.cleared, ra.done, ra.ghosts], `decision ${k}`);
    ghosts += ra.ghosts; lost += ra.lifeLost ? 1 : 0; cleared += ra.cleared ? 1 : 0;
  }
  assert.ok(ghosts > 0 && lost > 0, `the check should cover ghost and death pauses (ghosts ${ghosts}, lives ${lost}, levels ${cleared})`);
  assert.ok(b.game.frame > a.game.frame + 300, 'the page version really paused');
});

test('env: sticky moves are per frame: a new direction is late one frame 25% of the time, two frames 6.25%', () => {
  const env = new A.ArcadeEnv({ seed: 5, sticky: 0.25 }); env.reset(5);
  let changes = 0, late1 = 0, second = 0, late2 = 0;
  for (let k = 0; k < 12000; k++) {
    if (env.game.over) env.reset(5 + k);
    const want = k % 2 ? 2 : 3, wantDir = A.ACTION_DIRS[want];
    env.begin(want);
    let f = 0, r = null, stale = false;
    while (r === null) {
      const before = env.joy, frozen = env.game.freeze > 0;
      r = env.tick();
      if (frozen) continue;
      if (f === 0 && before !== null && before !== wantDir) { changes++; stale = env.joy === before; if (stale) late1++; }
      else if (f === 1 && stale && r === null) { second++; if (env.joy !== wantDir) late2++; }
      f++;
    }
  }
  close(late1 / changes, 0.25, 0.03, `late on the first frame (${changes} changes)`);
  close(late2 / second, 0.25, 0.06, `still late on the second frame (${second} cases)`);
  const plain = new A.ArcadeEnv({ seed: 5 }); plain.reset(5); plain.step(2);
  assert.strictEqual(plain.joy, null, 'sticky 0 never holds the stick back');
});

// Everything that matters about a game's state, except the frame counter (pauses add frames).
const stateKey = (g) => JSON.stringify([g.pac.x, g.pac.y, g.pac.dir, g.score, g.lives, g.level, g.dotsLeft, g.frightTimer, g.modeIndex, g.modeTimer,
  g.ghosts.map((h) => [h.x, h.y, h.dir, h.state, h.frightened])]);
test('env: with sticky moves on too, the page (paused) and training (unpaused) stay identical', () => {
  const a = new A.ArcadeEnv({ seed: 12, sticky: 0.25 }), b = new A.ArcadeEnv({ seed: 12, sticky: 0.25, pauses: true });
  a.reset(12); b.reset(12);
  const rng = A.makeXorshift(7);
  let k = 0, ghosts = 0;
  for (; k < 20000 && !a.game.over; k++) {
    const act = heuristic(a.observe(), rng);
    const ra = a.step(act);
    b.begin(act); let rb = null; while (rb === null) rb = b.tick();
    assert.deepStrictEqual([rb.points, rb.lifeLost, rb.done, stateKey(b.game)], [ra.points, ra.lifeLost, ra.done, stateKey(a.game)], `decision ${k}`);
    ghosts += ra.ghosts;
  }
  assert.ok(k > 500 && ghosts > 0, `${k} decisions, ${ghosts} ghosts`);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
