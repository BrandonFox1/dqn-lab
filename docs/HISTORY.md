# Project history

Everything below happened in one claude.ai conversation on 2026-09-23. The verbatim conversation is in `chat-log.md`; this file is the organized record: what was built, what was measured, and why things are the way they are.

## Timeline
1. **"Will Jev be able to learn how to play Pac-Man?"** Jev (TypeSafe AI's "System One" decision model) is pretrained and zero-shot. It doesn't learn online. It could play through a harness that serializes the game state each tick (its 70–500 ms latency fits a game loop), but it would never get better by playing.
2. **"I'd rather build a DQN agent."** Built `pacdqn` (below), trained it, delivered a zip, GIF and learning curve.
3. **Beginner explanation** (Brandon on his phone): Q is a scorecard for moves; the network approximates it; it learns from reward plus its own next guess; ε-greedy exploration; replay memory.
4. **Mechanism explanation**: loss from q vs target, backprop (chain rule → gradients), gradient descent/Adam, target network, Double DQN, reward scaling, gradient clipping.
5. **Explainer GIF** (`explainers/`) built from one real transition of pacdqn's `best.pt` (numbers below).
6. **The DQN Workshop**: an interactive page, built, tested and published at https://claude.ai/artifact/83NAYFuuY2pjCfeYbm4UG4
7. **"What if it were the original arcade game, 4 ghosts and all?"** Answered in detail (summary below; plan in `ROADMAP.md`).
8. **Move to Claude Code**: this repo. First session: all checks green (pytest 33 pass + 1 gymnasium skip, 19 engine tests, build byte-identical to the shipped page, 23/23 QA).
9. **GitHub Pages**: `.github/workflows/workshop.yml` checks every workshop PR (tests, build, dist freshness, browser QA) and publishes the page to https://brandonfox1.github.io/dqn-lab/ on every merge to `main`, so updates no longer need a claude.ai chat.

## pacdqn (Python)
- **Environment**: original Pac-Man-style maze (no Namco assets). Mazes `small` 11×13 and `medium` 18×19. 8-channel grid observation: walls, pellets, power pellets, player, dangerous ghosts, frightened ghosts, ghosts' previous cells, power-timer plane. Ghosts take the BFS-shortest step with probability "aggression" (0.9, 0.7, 0.5, 0.3), can't reverse mid-corridor, flee at half speed after a power pellet. Rewards in points: pellet +10, power +50, ghost +200, clear +500, death −500, step −1; trained on points × 0.01. Death stops bootstrapping, the step limit does not.
- **Agent**: dueling Double DQN, uint8 replay buffer, Huber loss, gradient clipping 10, hard target sync every 1,000 steps, ε 1 → 0.05 over 30k steps, γ 0.99, Adam lr 2.5e-4.
- **Adaptations for a 1-core sandbox**: model slimmed to 1.23M parameters with a 1×1 "squeeze" conv; train every 4 steps; batch 32.
- **Demo run** (`runs/small`, 60k steps, ~12 min): greedy eval −460 (10k), −338 (20k), −276 (30k), −239 (40k), −205 (50k), −39 (60k). `best.pt` averages +54 over 20 fresh seeds vs −478 random. It eats 20–30 pellets and dodges the ghost but has 0% clears. The curve was still rising steeply.
- **Repo slimming**: `best.pt` shipped weights-only (optimizer state removed, 14.8 → 4.9 MB). `DQNAgent.load` skips the optimizer when absent (covered by `tests/test_demo_checkpoint.py`). `latest.pt` not included.

## The explainer: one real update, with real numbers
From `best.pt`, seed 5, move 12, Pac-Man moves LEFT onto a pellet:
- reward +9 points (pellet +10, step −1) → r = +0.09
- live network q(S) = [1.81, 1.78, 1.86, 1.29] for up, down, left, right, so the guess q(S, left) = 1.86
- on the next board S′ the live network picks DOWN; the frozen copy values that move at v = 2.03
- target = 0.09 + 0.99 × 2.03 = 2.10; gap = 1.86 − 2.10 = −0.24; Huber loss ≈ 0.03

## The DQN Workshop
### Design
- Brief: a stylized workshop page to visualize the process and experiment with situations. Brandon is on a phone, so it is mobile-first.
- Plan (then reviewed against generic defaults): light = engineering notebook on graph paper, dark = blueprint; arcade screens always dark navy. Color = meaning (guess yellow, target green, gap red, live blue, frozen purple, reward orange) with highlighter-stroke styling. One typeface (Recursive) used expressively: casual axis for headings, mono axis for numbers. The hero is the update rule itself, with highlighter strokes that draw in once on load (the page's single orchestrated motion). Binder-style tabs. The Bench's worked solution is a true sequence, so it gets numbered margin steps on a notebook margin line instead of cards.
### Engine (`workshop/engine.js`)
- Seeded RNG (mulberry32), MLP with He init and sparse-aware forward, backprop, SGD, Adam with global-norm clipping, clone/copy/base64.
- MiniPac: 9×7 maze, 29 walkable squares, 27 pellets, 116-number observation, rewards pellet +10 / step −1 / caught −300 / cleared +200, 150-step limit, ghost aggression 0.75, optional deterministic ghost.
- GridWorld (7×6, cherry +10, ghost −10, walls), TableQ, NetQ (input = one-hot column + one-hot row, hidden 32).
- DQN: hidden [64, 64] = 11,908 weights; lr 1e-3, γ 0.97, batch 32, replay 20k, learning starts 1k, target sync 500, ε decay 15k, reward scale 0.01. Supports double / frozen copy / replay on-off (no replay = learn from the latest transition only).
- 19 tests: finite-difference gradient check on every weight, SGD/Adam, base64 round trip, env rules, determinism, Double DQN target math, replay wrap, sync schedule, gridworld dynamics, table sweeps spreading exactly one square per sweep, network gridworld reaching the cherry from ≥90% of squares.
### Training the brain (Node, ~450–500 steps/s on one core)
Run A, 100k steps, greedy eval of 50 games every 5k. Random play averages −279.7.

| steps | 5k | 10k | 15k | 20k | 25k | 45k | 60k | 65k | 85k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| eval score | −213 | −190 | −208 | −39.5 | +23.5 | +250 (60% clears) | +284 (66%) | +378 (96%) | **+411.6 (100%)** | +356 (88%) |

Fine-tune (`finetune.js`, lr 3e-4, +50k steps, selected on 100 games, compared on 500 held-out games): brain_A 95.6% clears / avg 390.8 vs fine-tuned **97.8% / 407.9** → shipped as `brain_final` (135k steps total). In-page QA: 29–30 of 30 boards cleared.
### Bench calibration
- The trained brain's Q-values sit around 1.3–1.8 almost everywhere (it expects to win from most places), so the value map uses a relative color scale with the actual numbers printed.
- "Walk into the ghost" is the instructive case: the trained brain rates the move at +1.1 to +1.6 while the target is −2.91 (the brain rarely saw a ghost placed adjacent with no heading). The gap is about 4 to 4.5.
- |∇q|² is about 30–50 for the trained brain and about 9.5 for a fresh one. With plain SGD, lr 0.003 closes gaps smoothly, while 0.01 and above overshoot and oscillate. That became the learning-rate lesson and the overshoot message.
### Experiment measurements (from scratch, single seed, greedy eval; logs in `workshop/results/`)
| run | 20k | 30k | 40k | 50k | note |
|---|---|---|---|---|---|
| default | −39.5 | +3.2 | +80.4 | +156.5 | 60% clears at 45k, 96% at 65k (run A) |
| no frozen copy | −144.3 | −15.2 | +37.9 | −15.6 | hovered near zero, never above 14% clears |
| γ 0.5 | +8.1 | **+346.1 (90%)** | | | learned faster in this tiny maze (run stopped at 35k: +314) |
| no replay | −214.8 | −226.4 | −187.2 | | never took off |
| lr 0.01 | −213.6 | −121.6 | −146.1 | | lurched, never took off |
| never explore (ε = 0) | +59.9 | +265.1 (74%) | −154.1 (10%) | | learned, then fell apart |
### The practice-vs-exam finding (why the Arcade chart shows an exam line)
After loading the trained brain and continuing training, the chart of training-game scores dropped from about +300 to +66. Before shipping, this was investigated. With no learning at all:

| random-move chance ε | 0 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| avg score | 414 | 313 | 220 | 52 |
| boards cleared | 99% | 81% | 65% | 36% |

Continuing training at default settings, greedy exams every 4k steps stayed at 98%, 93%, 93%, 100%, 100% clears while practice averages were −65 to +101. The brain wasn't collapsing: one random step next to the ghost costs 300 points. So the Arcade now runs an exam (10 fixed greedy games every 2,500 steps) as the headline line, with practice games as dots.
### QA
- `qa.js`: 23 checks at phone (390×844) and desktop (1280×900), light and dark. They cover no horizontal scroll, Bench math (ghost target −2.91, pellet r = 0.09), nudges shrinking the gap, the ripple explanation, the target moving with no frozen copy, the overshoot message, board editing, Tank sweeps and exploring, Arcade training, the exam schedule, experiment markers, restoring defaults, the loaded brain's in-page performance, test drive, send-to-Bench, and zero console errors.
- `qa_extra.js`: exams land exactly at steps 0, 2,500, 5,000, 7,500. Raw in-page training runs about 500 steps/s; Turbo with rendering ran 130–190 steps/s in headless phone emulation.
- Fixed along the way: the target label was green-on-green inside its bar (now a chip); Turbo Q-bar labels desynced by CSS transitions (transitions off at speed); the exam line drew a diagonal across a trained-brain load (line now breaks); QA clicked a button inside a closed `<details>`; a sticky tab bar intercepted a coordinate click.

## The 4-ghost arcade question (summary; full answer in chat-log)
- **Same**: the whole learning loop (target r + γ·v, Huber, backprop, replay, frozen copy, Double DQN, exploration, exams). The Bench math is identical, and the Ripple Tank lesson holds over longer chains.
- **Different**: the game (28×31 maze with tunnel, 240 dots and 4 power pellets, fruit, lives, levels; four ghost personalities with scatter/chase timers, frightened mode and eyes); richer observation (per-ghost layers, frightened layer, timers) or pixels; a roughly million-weight conv net and GPU-days of training; reward design (arcade rewards run from 10 to 1,600, and clipping to ±1 undervalues ghosts; no death penalty in the arcade, so treat life loss as episode end); decisions at intersections with γ ≈ 0.99; replay ~1M with prioritized replay and n-step returns.
- **Harder, not just bigger**: hidden timers (partial observability), deterministic ghosts (a DQN can memorize a route, so exams need sticky actions), and deadlier exploration.
- **Real-world reference**: in the 2015 DQN paper, Ms. Pac-Man (Atari 2600) scored roughly 2,300 vs about 15,700 for the human tester. Microsoft's 2017 Hybrid Reward Architecture hit the maximum 999,990 by splitting the reward into many small learners.

## Decisions and why
- Build a learning agent (DQN) rather than drive a pretrained model: Brandon wanted something that improves by playing.
- Original maze and characters, no Namco assets.
- Rewards in game points × 0.01 so targets stay in a sane range; Huber loss so big surprises can't yank the weights.
- The workshop uses a small MLP on a walkable-square encoding (116 inputs) instead of a CNN, so it trains in a browser in minutes.
- The Bench uses plain SGD for nudges (step size proportional to the gradient, easy to explain), a deterministic chase ghost (predictable S′), and a live Double DQN / frozen copy toggle.
- The Arcade headline metric is an exam, not practice scores.
- Experiment notes come from measured runs and say "in our test run".
- The page is one self-contained HTML file, published from claude.ai and (from the Claude Code move on) GitHub Pages via Actions. The Pages site is built from the committed `dist/`, and CI fails if that file is stale, so the file attached to claude.ai and the Pages site can't drift apart.
