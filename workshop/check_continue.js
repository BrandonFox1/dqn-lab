const E = require('./engine.js'); const fs = require('fs');
const b64 = fs.readFileSync('brains/brain_A.b64', 'utf8');
function run(lr, learningStarts, steps = 12000) {
  const env = new E.MiniPac({ seed: 9 }); const ag = new E.DQN(116, 4, { seed: 3, lr, learningStarts });
  ag.live.loadBase64(b64); ag.frozen.copyFrom(ag.live); ag.steps = 15000;
  let o = env.reset(9); const sc = []; const out = [];
  for (let s = 1; s <= steps; s++) { const a = ag.act(o); const r = env.step(a); ag.observe(o, a, r.reward, r.obs, r.terminal); o = r.obs; if (env.done) { sc.push(env.score); o = env.reset(); }
    if (s % 2000 === 0) { const l = sc.slice(-20); out.push((l.reduce((x, y) => x + y, 0) / l.length).toFixed(0)); } }
  return out.join(' ');
}
console.log('lr 1e-3, starts 1000:', run(1e-3, 1000));
console.log('lr 3e-4, starts 1000:', run(3e-4, 1000));
console.log('lr 1e-3, starts 5000:', run(1e-3, 5000));
