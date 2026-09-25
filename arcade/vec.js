/* Many ArcadeEnv games stepped in lockstep. Shared by bridge.js (training and Python exams) and eval_brain.js,
   so an exam played in Python and the same exam played in pure JavaScript are the same games move for move. */
const A = require('./arcade.js');

class VecEnv {
  // sticky: per-frame chance that the stick stays put (ArcadeEnv's sticky option); autoReset: restart finished games
  constructor(count, seedBase = 1, sticky = 0, autoReset = true) {
    this.count = count; this.autoReset = autoReset; this.nextSeed = seedBase + count;
    this.envs = [];
    for (let i = 0; i < count; i++) { const e = new A.ArcadeEnv({ seed: seedBase + i, sticky }); e.reset(seedBase + i); this.envs.push(e); }
  }
  // Returns one result per game. A finished game (autoReset off) keeps reporting game over and its final score.
  step(actions) {
    const out = [];
    for (let i = 0; i < this.count; i++) {
      const e = this.envs[i];
      if (e.game.over) { out.push({ points: 0, lifeLost: false, done: true, cleared: false, level: e.game.level, score: e.game.score, frames: 0, ghosts: 0 }); continue; }
      const r = e.step(actions[i]);
      out.push(r);
      if (r.done && this.autoReset) e.reset(this.nextSeed++);
    }
    return out;
  }
}
module.exports = { VecEnv };
