# Roadmap

## Workshop
- **Real-phone performance check.** Target Turbo ≥ 150 steps/s with rendering. If it's slow, move training into a Web Worker (verify the page's content-security policy allows blob workers) or render less often.
- **Multi-seed experiments.** Re-run `run_ablations.sh` with 3–5 seeds each and show the spread; update `src/experiment-notes.json`.
- **Brain scan on the Bench.** Show hidden-layer activations for the current board and which weights moved most on a nudge.
- **Accessibility.** Screen-reader descriptions for boards and charts, plus an explicit theme toggle.
- **Teaching extras.** An n-step target toggle on the Bench; show the actual gradient for a few weights.

## pacdqn
- A longer GPU run (300k–1M steps) on `small`, then a curriculum to `medium` with 2 ghosts (`--eps-decay-steps 300000 --steps 1000000 --buffer-size 200000`).
- Prioritized replay and n-step returns (the Rainbow pieces in the pacdqn README).
- Exams with sticky actions for honest evaluation.

## Arcade-faithful version (4 ghosts and all)
Two routes:
1. **Fastest: ALE Ms. Pac-Man via Gymnasium.** Pixels with Atari preprocessing, 4 stacked frames, and a Nature-style CNN plugged into pacdqn's agent. Needs a rented GPU for days. Results are directly comparable to published work (2015 DQN ≈ 2,300 vs human ≈ 15,700; the 2017 Hybrid Reward Architecture hit 999,990).
2. **Most educational: extend pacdqn's own environment to arcade rules.**
   - 28×31 maze with wraparound tunnel, 240 dots and 4 power pellets, fruit, 3 lives, levels that speed up.
   - Ghost personalities: Blinky targets Pac-Man, Pinky 4 tiles ahead, Inky from a vector through Blinky, Clyde retreats within 8 tiles. Scatter/chase timers, frightened mode, eyes returning home, house release rules.
   - Decisions only at intersections; γ ≈ 0.99.
   - Observation layers per ghost plus frightened and timer features (or frame stacking / a recurrent net).
   - Reward design: don't clip to ±1; compress large rewards; treat a lost life as episode end for targets.
   - Replay ~1M with prioritized replay and n-step returns.
   - Exams with sticky actions, because deterministic ghosts invite memorized routes.
   - Safer exploration: a lower random-move floor, exploring only at intersections, or noisy networks.
   - Workshop impact: the Bench and Ripple Tank carry over; the Arcade becomes "watch a pretrained brain" plus a recorded training curve.
   - Keep original stand-in characters and maze.
