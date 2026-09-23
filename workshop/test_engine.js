const E = require('./engine.js');
const assert = require('assert');
let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ok  ', name); }
  catch (e) { failed++; console.log('  FAIL', name, '\n       ', e.message); }
}
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b} (tol ${tol})`);

// ---------------------------------------------------------------- MLP
test('MLP forward shape and determinism', () => {
  const a = new E.MLP([5, 7, 3], 42), b = new E.MLP([5, 7, 3], 42);
  const x = [0.1, 0, 1, -0.5, 0.3];
  assert.deepStrictEqual(Array.from(a.forward(x)), Array.from(b.forward(x)));
  assert.strictEqual(a.forward(x).length, 3);
  assert.strictEqual(a.paramCount, 5 * 7 + 7 + 7 * 3 + 3);
});

test('backprop matches finite differences (every weight)', () => {
  const net = new E.MLP([6, 5, 4, 3], 7, 1.0);
  const rng = E.makeRng(1);
  const x = Array.from({ length: 6 }, () => rng.normal());
  const a = 1, target = 0.3;
  const loss = () => E.huber(net.forward(x)[a] - target);
  net.zeroGrad();
  const q = net.forward(x); const d = new Float32Array(3); d[a] = E.huberGrad(q[a] - target); net.backward(d);
  let checked = 0;
  for (let l = 0; l < net.L; l++) for (const [P, G] of [[net.W[l], net.gW[l]], [net.b[l], net.gb[l]]]) {
    for (let i = 0; i < P.length; i++) {
      const old = P[i], h = 1e-3;
      P[i] = old + h; const lp = loss(); P[i] = old - h; const lm = loss(); P[i] = old;
      const num = (lp - lm) / (2 * h);
      close(G[i], num, 2e-3 + 2e-2 * Math.abs(num), `layer ${l} idx ${i}`);
      checked++;
    }
  }
  assert.strictEqual(checked, net.paramCount);
});

test('SGD step moves the output toward the target', () => {
  const net = new E.MLP([4, 8, 2], 3, 1.0); const x = [1, 0, 0.5, 0];
  const before = net.forward(x)[0], target = before + 1;
  for (let k = 0; k < 20; k++) { const q = net.forward(x); const d = new Float32Array(2); d[0] = E.huberGrad(q[0] - target); net.backward(d); net.sgdStep(0.05); }
  const after = net.forward(x)[0];
  assert.ok(Math.abs(after - target) < Math.abs(before - target) * 0.5, `${before} -> ${after} target ${target}`);
});

test('Adam reduces loss on a fixed batch', () => {
  const net = new E.MLP([4, 8, 2], 3); const rng = E.makeRng(2);
  const data = Array.from({ length: 16 }, () => ({ x: Array.from({ length: 4 }, () => rng.normal()), y: rng.normal() }));
  const L = () => data.reduce((s, { x, y }) => s + E.huber(net.forward(x)[1] - y), 0);
  const l0 = L();
  for (let k = 0; k < 200; k++) { for (const { x, y } of data) { const q = net.forward(x); const d = new Float32Array(2); d[1] = E.huberGrad(q[1] - y) / 16; net.backward(d); } net.adamStep(0.01); }
  assert.ok(L() < l0 * 0.3, `${l0} -> ${L()}`);
});

test('clone / copyFrom / base64 round trip', () => {
  const a = new E.MLP([5, 6, 3], 9); const x = [1, 0, 0, 1, 0];
  const c = a.clone(); assert.deepStrictEqual(Array.from(c.forward(x)), Array.from(a.forward(x)));
  const b = new E.MLP([5, 6, 3], 10); b.loadBase64(a.toBase64());
  assert.deepStrictEqual(Array.from(b.forward(x)), Array.from(a.forward(x)));
  const z = new E.MLP([5, 6, 3], 11); z.copyFrom(a); assert.deepStrictEqual(Array.from(z.forward(x)), Array.from(a.forward(x)));
  assert.throws(() => new E.MLP([5, 7, 3], 1).loadBase64(a.toBase64()));
});

// ------------------------------------------------------------ MiniPac
test('maze parses: 29 squares, 27 pellets, connected', () => {
  const env = new E.MiniPac();
  assert.strictEqual(env.N, 29); assert.strictEqual(env.pelletsLeft, 27); assert.strictEqual(env.obsSize, 116);
  for (let c = 0; c < env.N; c++) assert.ok(env.dist[env.playerStart][c] < 1e6);
});

test('walls block, pellets pay, step costs', () => {
  const env = new E.MiniPac({ aggression: 0 }); env.reset(1);
  const start = env.player;
  const r0 = env.step(1); // down into the wall
  assert.ok(r0.bumped); assert.strictEqual(env.player, start); assert.strictEqual(r0.reward, -1);
  const r1 = env.step(2); // left onto a pellet
  assert.strictEqual(r1.reward, 9); assert.strictEqual(env.pelletsLeft, 26);
  assert.deepStrictEqual(r1.parts, [['step', -1], ['pellet', 10]]);
});

test('observation encodes pellets, Pac-Man, ghost, ghost trail', () => {
  const env = new E.MiniPac(); const o = env.reset(1);
  const N = env.N;
  assert.strictEqual(o.slice(0, N).reduce((a, b) => a + b), 27);
  assert.strictEqual(o[N + env.playerStart], 1); assert.strictEqual(o[2 * N + env.ghostStart], 1); assert.strictEqual(o[3 * N + env.ghostStart], 1);
  assert.strictEqual(o.reduce((a, b) => a + b), 30);
});

test('walking into the ghost ends the game with the death penalty', () => {
  const env = new E.MiniPac(); env.reset(1);
  const s = env.getState(); s.ghost = env.nbr[env.player][2]; s.ghostPrev = s.ghost; env.setState(s);
  const r = env.step(2);
  assert.ok(r.terminal && r.caught && env.dead);
  assert.ok(r.parts.some(([k, v]) => k === 'caught' && v === -300));
  assert.throws(() => env.step(0));
});

test('deterministic chase ghost closes the distance', () => {
  const env = new E.MiniPac(); env.reset(1);
  const d0 = env.dist[env.player][env.ghost];
  env.step(1, { deterministicGhost: true }); // bump wall, ghost steps closer
  assert.strictEqual(env.dist[env.player][env.ghost], d0 - 1);
});

test('eating the last pellet wins; time limit truncates', () => {
  const env = new E.MiniPac({ aggression: 0 }); env.reset(1);
  const s = env.getState(); s.pellets.fill(0); s.pellets[env.nbr[env.player][2]] = 1; env.setState(s);
  const r = env.step(2); assert.ok(r.terminal && env.won); assert.strictEqual(r.reward, -1 + 10 + 200);
  const e2 = new E.MiniPac({ maxSteps: 3, aggression: 0 }); e2.reset(1);
  e2.step(1); e2.step(1); const t = e2.step(1); assert.ok(t.truncated && !t.terminal);
});

test('same seed, same game', () => {
  const run = (seed) => { const env = new E.MiniPac(); env.reset(seed); const rng = E.makeRng(seed); const tr = [];
    while (!env.done) { const r = env.step(rng.int(4)); tr.push(env.player * 100 + env.ghost, r.reward); } return tr.join(','); };
  assert.strictEqual(run(5), run(5)); assert.notStrictEqual(run(5), run(6));
});

// ---------------------------------------------------------------- DQN
test('Double DQN target: live net picks, frozen net values, terminal ignores future', () => {
  const env = new E.MiniPac(); const agent = new E.DQN(env.obsSize, 4, { gamma: 0.9, seed: 3 });
  for (const w of agent.frozen.W) for (let i = 0; i < w.length; i++) w[i] += 0.3 * Math.sin(i);
  const o = env.reset(2);
  const t = agent.target(0.5, o, false);
  const ql = agent.live.forward(o), qf = agent.frozen.forward(o);
  const aStar = E.argmax(ql);
  close(t.target, 0.5 + 0.9 * qf[aStar], 1e-6); assert.strictEqual(t.aStar, aStar);
  agent.cfg.double = false; const t2 = agent.target(0.5, o, false);
  close(t2.target, 0.5 + 0.9 * Math.max(...qf), 1e-6);
  agent.cfg.useTarget = false; agent.cfg.double = true; const t3 = agent.target(0.5, o, false);
  close(t3.target, 0.5 + 0.9 * Math.max(...ql), 1e-6);
  const t4 = agent.target(0.5, o, true); assert.strictEqual(t4.target, 0.5); assert.strictEqual(t4.v, 0);
});

test('trainOn moves q(S,a) toward the target', () => {
  const env = new E.MiniPac(); const agent = new E.DQN(env.obsSize, 4, { seed: 4 });
  const o = env.reset(3); const r = env.step(2); const tr = { obs: o, a: 2, r: r.reward * 0.01, next: r.obs, terminal: r.terminal };
  const tgt = agent.target(tr.r, tr.next, tr.terminal).target;
  const before = agent.live.forward(o)[2];
  for (let k = 0; k < 30; k++) agent.trainOn([tr], 'sgd', 0.01);
  const after = agent.live.forward(o)[2];
  assert.ok(Math.abs(after - tgt) < Math.abs(before - tgt), `${before} -> ${after}, target ${tgt}`);
});

test('replay buffer wraps and stores binary obs', () => {
  const env = new E.MiniPac(); const agent = new E.DQN(env.obsSize, 4, { bufferSize: 10, learningStarts: 1e9 });
  let o = env.reset(1);
  for (let k = 0; k < 25; k++) { const r = env.step(k % 4); agent.observe(o, k % 4, r.reward, r.obs, r.terminal); o = r.obs; if (env.done) o = env.reset(); }
  assert.strictEqual(agent.size, 10); assert.strictEqual(agent.pos, 5);
  const b = agent.sample(4); assert.strictEqual(b.length, 4); assert.strictEqual(b[0].obs.length, env.obsSize);
});

test('target network syncs on schedule', () => {
  const env = new E.MiniPac(); const agent = new E.DQN(env.obsSize, 4, { targetSync: 50, learningStarts: 10, batch: 4 });
  let o = env.reset(1);
  for (let k = 0; k < 100; k++) { const a = agent.act(o); const r = env.step(a); agent.observe(o, a, r.reward, r.obs, r.terminal); o = env.done ? env.reset() : r.obs; }
  assert.strictEqual(agent.syncs, 2);
  assert.deepStrictEqual(Array.from(agent.frozen.W[0].slice(0, 20)), Array.from(agent.live.W[0].slice(0, 20)));
});

// ---------------------------------------------------------- GridWorld
test('gridworld dynamics', () => {
  const g = new E.GridWorld(7, 6);
  const cherry = 6; // top-right
  const r = g.transition(5, 3); assert.ok(r.terminal); assert.strictEqual(r.reward, 10); assert.strictEqual(r.next, cherry);
  const w = g.transition(1, 3); assert.strictEqual(w.next, 2); // open square to the right
  assert.strictEqual(g.transition(0, 0).next, 0); // edge bump
  assert.strictEqual(g.transition(8, 3).next, 8); // (1,1) -> (2,1) is a wall
});

test('table sweeps spread value exactly one square per sweep', () => {
  const g = new E.GridWorld(7, 6); const Q = new E.TableQ(42); const gamma = 0.9;
  const d = g.distanceToCherry();
  for (let k = 1; k <= 4; k++) {
    const snap = Q.snapshot();
    for (const s of g.states()) for (let a = 0; a < 4; a++) {
      const tr = g.transition(s, a); const v = tr.terminal ? 0 : Math.max(...snap.values(tr.next));
      Q.update(s, a, tr.reward + gamma * v, 1);
    }
    for (const s of g.states()) {
      const V = Math.max(...Q.values(s));
      if (d[s] >= 1 && d[s] <= k) close(V, 10 * Math.pow(gamma, d[s] - 1), 1e-4, `sweep ${k} square ${s} d=${d[s]}`);
      if (d[s] > k) assert.ok(V <= 1e-6, `square ${s} at distance ${d[s]} knows too early after ${k} sweeps: ${V}`);
    }
  }
});

test('network brain learns the gridworld from experience', () => {
  const g = new E.GridWorld(7, 6); const Q = new E.NetQ(g); const rng = E.makeRng(9); const gamma = 0.9;
  const buf = [];
  for (let ep = 0; ep < 400; ep++) {
    let s = g.randomStart();
    for (let t = 0; t < 40; t++) {
      const a = rng() < 0.3 ? rng.int(4) : E.argmax(Q.values(s));
      const tr = g.transition(s, a); buf.push({ s, a, ...tr }); if (buf.length > 3000) buf.shift();
      const frozen = Q.snapshot();
      const batch = []; for (let k = 0; k < 16; k++) { const b = buf[rng.int(buf.length)]; batch.push({ s: b.s, a: b.a, target: b.reward + (b.terminal ? 0 : gamma * Math.max(...frozen.values(b.next))) }); }
      Q.train(batch, 0.01);
      if (tr.terminal) break; s = tr.next;
    }
  }
  // greedy policy from every square should reach the cherry
  let reached = 0, total = 0;
  for (const s0 of g.states()) { total++; let s = s0; for (let t = 0; t < 30; t++) { const tr = g.transition(s, E.argmax(Q.values(s))); if (tr.terminal) { if (tr.reward > 0) reached++; break; } s = tr.next; } }
  assert.ok(reached / total >= 0.9, `reached ${reached}/${total}`);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
