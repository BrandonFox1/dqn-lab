/* Plays an exported arcade brain in pure JavaScript, with the Workshop's own hand-written MLP (workshop/engine.js).
     node eval_brain.js BRAIN [--games 10] [--seed-base 1000] [--sticky 0.25] [--max 50000]
   BRAIN is the path without extension (BRAIN.b64 + BRAIN.json). Observations are rounded to 1/255 steps, exactly as
   the trainer stores them, so these games match the Python exam (pacdqn arcade_train exam) move for move.
     node eval_brain.js BRAIN --q < obs.bin    prints Q-values for raw u8 observations (the parity test uses this) */
const fs = require('fs');
const A = require('./arcade.js');
const E = require('../workshop/engine.js');
const { VecEnv } = require('./vec.js');

// Half floats (meta.format 'f16') are widened to float32 exactly as the page does it.
function halfToFloat(v) {
  const s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 31, f = v & 1023;
  return e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? s * Infinity : s * (1 + f / 1024) * 2 ** (e - 15);
}
function loadBrain(base) {
  const meta = JSON.parse(fs.readFileSync(base + '.json', 'utf8'));
  const net = new E.MLP(meta.sizes, 1);
  const b64 = fs.readFileSync(base + '.b64', 'utf8').trim();
  if (meta.format === 'f16') {
    const bytes = E.b64ToBytes(b64), h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    net.loadFlat(Float32Array.from(h, halfToFloat));
  } else net.loadBase64(b64);
  return { net, meta };
}
function quantize(obs, out) { for (let k = 0; k < obs.length; k++) out[k] = Math.round(obs[k] * 255) / 255; return out; }

function play(net, { games = 10, seedBase = 1000, sticky = 0.25, max = 50000 } = {}) {
  const vec = new VecEnv(games, seedBase, sticky, false);
  const x = new Float32Array(A.OBS.size), active = new Array(games).fill(true);
  const score = new Array(games).fill(0), level = new Array(games).fill(1), ghosts = new Array(games).fill(0);
  for (let t = 0; t < max && active.some(Boolean); t++) {
    const acts = vec.envs.map((e) => E.argmax(net.forward(quantize(e.observe(), x))));
    const res = vec.step(acts);
    res.forEach((r, i) => { if (!active[i]) return; score[i] = r.score; level[i] = r.level; ghosts[i] += r.ghosts; if (r.done) active[i] = false; });
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const sorted = score.slice().sort((x, y) => x - y), mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2; // same as numpy's median
  return { mean_score: mean(score), median_score: median, max_score: Math.max(...score), mean_level: mean(level), max_level: Math.max(...level),
    levels_cleared: level.reduce((s, v) => s + v - 1, 0), ghosts_per_game: mean(ghosts), unfinished: active.filter(Boolean).length, scores: score, levels: level };
}

if (require.main === module) {
  const args = process.argv.slice(2), base = args[0];
  const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? Number(args[i + 1]) : d; };
  const { net, meta } = loadBrain(base);
  if (args.includes('--q')) {
    const raw = fs.readFileSync(0), D = meta.sizes[0], x = new Float32Array(D), rows = [];
    for (let off = 0; off + D <= raw.length; off += D) { for (let k = 0; k < D; k++) x[k] = raw[off + k] / 255; rows.push(Array.from(net.forward(x))); }
    process.stdout.write(JSON.stringify(rows));
  } else {
    const t0 = Date.now();
    const r = play(net, { games: opt('games', 10), seedBase: opt('seed-base', 1000), sticky: opt('sticky', 0.25), max: opt('max', 50000) });
    console.log(JSON.stringify(Object.assign(r, { seconds: (Date.now() - t0) / 1000 })));
  }
}
module.exports = { loadBrain, play, quantize };
