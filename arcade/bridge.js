/* Serves a VecEnv (vec.js) to the Python trainer (pacdqn/pacdqn/arcade_env.py) over stdin/stdout.
   Binary protocol, little-endian:
     'I' u32 count, u32 seedBase, f32 sticky, u8 autoReset  -> create the games, reply with observations
     'S' u8[count] actions                                -> one decision in every game, reply with results
     'X'                                                   -> exit
   Reply: obs u8[count*D] (values*255), points f32[count], flags u8[count] (1 life lost, 2 game over, 4 level
   cleared), level u16[count], score u32[count], frames u16[count], ghosts u8[count]. Observations describe the
   next decision (after any restart); the other fields describe the decision just made. */
const A = require('./arcade.js');
const { VecEnv } = require('./vec.js');
const D = A.OBS.size;
let vec = null;
const obsF = new Float32Array(D);

function layout(n) {
  const obs = n * D, pts = obs + ((4 - (obs % 4)) % 4); // keep the f32 block aligned
  const flags = pts + 4 * n, level = flags + n + (n % 2), score = level + 2 * n, frames = score + 4 * n, ghosts = frames + 2 * n;
  return { pts, flags, level, score, frames, ghosts, total: ghosts + n };
}
function reply(results) {
  const n = vec.count, L = layout(n), out = Buffer.alloc(L.total);
  const pts = new Float32Array(n), level = new Uint16Array(n), score = new Uint32Array(n), frames = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const o = vec.envs[i].observe(obsF), base = i * D;
    for (let k = 0; k < D; k++) out[base + k] = Math.round(o[k] * 255);
    const r = results && results[i];
    if (!r) continue;
    pts[i] = r.points; level[i] = r.level; score[i] = r.score; frames[i] = Math.min(65535, r.frames);
    out[L.flags + i] = (r.lifeLost ? 1 : 0) | (r.done ? 2 : 0) | (r.cleared ? 4 : 0);
    out[L.ghosts + i] = r.ghosts;
  }
  Buffer.from(pts.buffer).copy(out, L.pts); Buffer.from(level.buffer).copy(out, L.level);
  Buffer.from(score.buffer).copy(out, L.score); Buffer.from(frames.buffer).copy(out, L.frames);
  process.stdout.write(out);
}

let pending = Buffer.alloc(0);
process.stdin.on('data', (chunk) => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  for (;;) {
    if (!pending.length) return;
    const op = String.fromCharCode(pending[0]);
    if (op === 'X') process.exit(0);
    if (op === 'I') {
      if (pending.length < 14) return;
      vec = new VecEnv(pending.readUInt32LE(1), pending.readUInt32LE(5), pending.readFloatLE(9), pending[13] === 1);
      pending = pending.subarray(14);
      reply(null);
    } else if (op === 'S') {
      if (!vec || pending.length < 1 + vec.count) return;
      const results = vec.step(pending.subarray(1, 1 + vec.count));
      pending = pending.subarray(1 + vec.count);
      reply(results);
    } else { process.stderr.write(`bridge: unknown opcode ${pending[0]}\n`); process.exit(2); }
  }
});
process.stdin.on('end', () => process.exit(0));
process.stdout.write(JSON.stringify({ obsSize: D, nActions: 4, version: 1 }) + '\n');
