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
- Exams with sticky actions for honest evaluation (the arcade trainer already does this; port its per-frame version to `train.py`).

## Four ghosts (the arcade-rules version, built 2026-09-25)
Done: the rules engine (`arcade/`), the trainer (`pacdqn/pacdqn/arcade_train.py`), the brain (29,654 points and 4.86 levels cleared per game over 50 new games) and the Workshop station. Next, roughly in order of value:
- **Beyond 60M decisions.** Doubling the training from 30M to 60M bought about 7% (30,688 vs 28,648 over 100 test games). More of the same is slow going; a bigger network, prioritized replay or starting some games at later levels (below) is likely worth more per hour.
- **Several seeds.** Train 3 seeds of the same recipe and show the spread on the chart. The shipped curve is a single run.
- **Checkpoint picker on the page.** Let the reader switch between the brain at 250k, 2M and the final checkpoint, to see what "more training" buys. At 1.4 MB per half-float brain, this needs 8-bit weights (about 0.5 MB each), plus a measured check that they change no moves.
- **Faster training.** On 4 CPU cores the learner is the bottleneck: about 1,650 decisions/s, while the JavaScript bridge alone serves about 15,800. With a GPU learner, the single bridge becomes the ceiling, so run several bridge processes, each with its own games, and aim for 100M+ decisions.
- **Later levels.** Exam games rarely reach the late levels, where frightened time drops to zero and the ghosts get faster. Start some training games at higher levels: `ArcadeEnv` already takes a `level` option, and the bridge would need to pass it through.
- **Prioritized replay** for rare, valuable moments (ghost chains, fruit).
- **A Bench for the arcade.** Show one real update on an arcade position: the 3-step target, the gap, and which observation numbers mattered.
- **Real-phone check** of the Four ghosts station at flat-out speed. Headless QA requires more than 150 decisions in 2 seconds.
- **Not planned, by choice.** Anything that needs ROM data stays out: the exact per-frame speed patterns, the frightened generator's ROM bytes, level 256, the original graphics and sounds. The ALE route (Atari Ms. Pac-Man through Gymnasium) runs the actual Atari ROM, so it also falls outside the documents-only decision.
