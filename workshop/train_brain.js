const E = require('./engine.js');
const fs = require('fs');
const steps = +process.argv[2] || 60000, tag = process.argv[3] || 'run', extra = JSON.parse(process.argv[4] || '{}');
const env = new E.MiniPac({ seed: 1 });
const agent = new E.DQN(env.obsSize, 4, Object.assign({ seed: 1 }, extra));
function evaluate(n, eps, seed0) { const e = new E.MiniPac({ seed: seed0 }); let sc = 0, wins = 0, len = 0;
  for (let i = 0; i < n; i++) { let o = e.reset(seed0 + i); while (!e.done) o = e.step(agent.act(o, eps)).obs; sc += e.score; wins += e.won; len += e.steps; }
  return { score: sc / n, win: wins / n, len: len / n }; }
function randomBaseline(n) { const e = new E.MiniPac(); const r = E.makeRng(99); let sc = 0; for (let i = 0; i < n; i++) { e.reset(1000 + i); while (!e.done) e.step(r.int(4)); sc += e.score; } return sc / n; }
console.log('random baseline', randomBaseline(500).toFixed(1), 'params', agent.live.paramCount);
let o = env.reset(1), best = -1e9, t0 = Date.now();
for (let s = 1; s <= steps; s++) {
  const a = agent.act(o); const r = env.step(a); agent.observe(o, a, r.reward, r.obs, r.terminal);
  o = env.done ? env.reset() : r.obs;
  if (s % 5000 === 0) { const ev = evaluate(50, 0.0, 50000 + s);
    const line = `${tag} step ${s} eval ${ev.score.toFixed(1)} win ${(ev.win*100).toFixed(0)}% len ${ev.len.toFixed(0)} eps ${agent.epsilon().toFixed(2)} loss ${agent.lastLoss.toFixed(4)} ${((s)/((Date.now()-t0)/1000)).toFixed(0)} sps`;
    console.log(line);
    if (ev.score > best) { best = ev.score; fs.writeFileSync(`brains/brain_${tag}.b64`, agent.live.toBase64()); fs.writeFileSync(`brains/brain_${tag}.json`, JSON.stringify({ step: s, ...ev })); } }
}
