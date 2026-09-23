const E = require('./engine.js'); const fs = require('fs');
const steps = +process.argv[2] || 80000;
const env = new E.MiniPac({ seed: 3 });
const agent = new E.DQN(env.obsSize, 4, { seed: 5, lr: 3e-4, epsDecay: 15000, learningStarts: 2000 });
agent.live.loadBase64(fs.readFileSync('brains/brain_A.b64', 'utf8')); agent.frozen.copyFrom(agent.live); agent.steps = 15000;
function evaluate(net, n, base) { const e = new E.MiniPac({ seed: base }); let sc = 0, w = 0, len = 0;
  for (let i = 0; i < n; i++) { let o = e.reset(base + i); while (!e.done) o = e.step(E.argmax(net.forward(o))).obs; sc += e.score; w += e.won; len += e.steps; } return { score: +(sc / n).toFixed(1), win: w / n, len: +(len / n).toFixed(1) }; }
const A = new E.MLP([116, 64, 64, 4], 1); A.loadBase64(fs.readFileSync('brains/brain_A.b64', 'utf8'));
let best = evaluate(A, 100, 70000), bestB64 = A.toBase64(), bestStep = 0;
console.log('brain_A on selection set', JSON.stringify(best));
let o = env.reset(3);
for (let s = 1; s <= steps; s++) { const a = agent.act(o); const r = env.step(a); agent.observe(o, a, r.reward, r.obs, r.terminal); o = env.done ? env.reset() : r.obs;
  if (s % 5000 === 0) { const ev = evaluate(agent.live, 100, 70000); console.log('ft step', s, JSON.stringify(ev)); if (ev.win > best.win || (ev.win === best.win && ev.score > best.score)) { best = ev; bestB64 = agent.live.toBase64(); bestStep = s; } } }
const B = new E.MLP([116, 64, 64, 4], 1); B.loadBase64(bestB64);
const hA = evaluate(A, 500, 900000), hB = evaluate(B, 500, 900000);
console.log('HELD-OUT brain_A', JSON.stringify(hA)); console.log('HELD-OUT finetuned (ft step ' + bestStep + ')', JSON.stringify(hB));
const pick = hB.win >= hA.win ? { b64: bestB64, ev: hB, step: 85000 + bestStep } : { b64: A.toBase64(), ev: hA, step: 85000 };
fs.writeFileSync('brains/brain_final.b64', pick.b64); fs.writeFileSync('brains/brain_final.json', JSON.stringify({ step: pick.step, heldOutGames: 500, win: pick.ev.win, score: pick.ev.score, len: pick.ev.len }));
console.log('picked', JSON.stringify(pick.ev), 'step', pick.step);
