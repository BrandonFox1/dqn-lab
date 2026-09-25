# Project history

Items 1–7 below happened in one claude.ai conversation on 2026-09-23; its verbatim record is `chat-log.md`. Items 8 onward happened in Claude Code sessions on this repo. This file is the organized record: what was built, what was measured, and why things are the way they are.

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
10. **Four ghosts** (2026-09-25): "I want to do #4 (the arcade-faithful version), as accurate as possible." Brandon first offered the original ROMs for the ghost logic, mechanics and graphics, then chose **documents only** (below). Built `arcade/` (the 1980 rules, frame by frame, every rule sourced and tested), a PyTorch trainer that plays that same JavaScript engine through a small bridge, a trained brain, and the Workshop's fourth station. After 5 hours of training on 4 CPU cores, the brain clears 5 levels per game on average: 28,975 points over 50 new games, best game level 12, against 718 for random play. Claude Code also republished the claude.ai copy of the page directly with its Artifact tool, first as a preview mid-training.
11. **Four ghosts to 60M decisions** (same day, evening): "keep training the brain to 60M decisions". The run resumed from the 30M checkpoint, optimizer included, for another 5.8 hours. A checkpoint from 57.5M decisions won a new selection round and ships: 29,654 on its final check. Mid-run, at Brandon's request, the claude.ai page carried the 39.75M brain, which measured a statistical tie with 30M. The same day's first Pages deploy was blocked by a race in `qa.js`, fixed in PR #5.

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
- **What we actually built** (item 10) kept most of this and changed three predictions. The network is a 520k-weight MLP over a hand-built observation, not a conv net on pixels, and it trained in hours on 4 CPU cores, not GPU-days. Decisions happen on every new tile, not only at intersections, because the arcade rewards early turns (cornering). Replay is uniform, with 3-step returns instead of prioritized sampling. As predicted, it uses γ = 0.99, compressed rewards, a lost life as the end of the target, a million-move replay memory and sticky-move exams.

## Four ghosts, part 1: the arcade rules (`arcade/`)
### Documents only
Brandon asked to use the original arcade ROMs he had uploaded earlier, for the exact ghost algorithm, the game mechanics and the graphics. Claude laid out the line: the rules of a game can be rebuilt from documents that describe them, while the ROM's program code and artwork are Namco's copyrighted work. Brandon chose **"No ROM, documents only"**:
- rules, ghost AI and maze layout come from public documentation;
- all artwork is original;
- no ROM files or ROM-derived data (graphics, sounds, program bytes) exist in the repo or on the page;
- frightened ghosts use a seeded random generator instead of reading ROM bytes.

### Sources
- *The Pac-Man Dossier* by Jamey Pittman (v1.0.26), which its author verified against the ROM disassembly. The PDF copy used ships in the MIT-licensed [floooh/pacman.c](https://github.com/floooh/pacman.c), and `arcade/SPEC.md` cites its page numbers.
- `pacman.c` itself (Andre Weissflog), a C rewrite from the Dossier, used only to cross-check coordinates. Its embedded ROM graphics were not used.
- `arcade/SPEC.md` numbers every rule (1.1 to 15.2) with its source. Each test in `test_arcade.js` names the rules it checks.

### What the engine does
- One `Game.step(stick)` is one frame at 60 frames per second. The screen is 28×36 tiles of 8 pixels. An actor's tile is the tile holding its center pixel, and a tile's center is pixel (3, 4) inside it.
- **Speeds** are percentages of 1.25 px/frame (80% = 1 px/frame), per level and mode (normal, frightened, tunnel, Cruise Elroy). Eating a dot stops Pac-Man for 1 frame, an energizer for 3.
- **Cornering**: Pac-Man can turn as soon as the next tile that way is open, cutting the corner diagonally. Ghosts turn only at tile centers.
- **Ghost pathfinding**: each ghost decides one tile ahead, never reverses by choice, picks the exit closest in a straight line to its target, and breaks ties up, left, down, right. No turning up in the red zones.
- **Targets**: Blinky goes for Pac-Man. Pinky aims 4 tiles ahead, with the original "facing up" overflow bug (4 up and 4 left). Inky doubles the vector from Blinky through the tile 2 ahead of Pac-Man. Clyde chases until he is within 8 tiles, then heads for his corner. Eyes aim for the door.
- **Waves and fright**: scatter and chase schedules per level; frightened mode pauses the wave clock; mode changes reverse the ghosts at their next tile.
- **Ghost house**: per-ghost dot counters (level 1: Inky 30, Clyde 60), the shared counter after a lost life (7, 17, 32), and the no-dot timer (4 s, 3 s from level 5).
- **Also**: Cruise Elroy, fruit at 70 and 170 dots, the extra life at 10,000, and tile collisions checked once per frame (so a ghost and Pac-Man can pass through each other, as in the arcade).
- **Where the documents are silent**, the choice is listed in SPEC.md. Examples: exact-average speed accumulators instead of the ROM's bit patterns, 1.5 px/frame eyes, the seeded frightened generator, and no level 256.

### Measured against the Dossier
A test runs each actor for thousands of frames and measures its average speed against Table A.1. "With dots" means eating a dot on every tile.

| level and mode | Dossier normal | engine normal | Dossier with dots | engine with dots |
|---|---|---|---|---|
| L1 Pac-Man | 80 | 80.0 | ~71 | 71.1 |
| L1 Pac-Man frightened | 90 | 89.5 | ~79 | 78.5 |
| L2 Pac-Man | 90 | 89.5 | ~79 | 78.5 |
| L2 Pac-Man frightened | 95 | 94.8 | ~83 | 82.6 |
| L5 Pac-Man | 100 | 100.0 | ~87 | 86.5 |
| L21 Pac-Man | 90 | 89.5 | ~79 | 78.5 |

**The stall investigation.** The first version had level 2 frightened Pac-Man with dots at 80.5%, not ~83%. The cause: the stalled frame was simply skipped, and at 95% speed stalls kept landing on frames that would have moved 2 pixels. With the stall *pausing* the speed rhythm instead, every "with dots" value in the table matches. That is now rule 4.3.

### Tests and fixes along the way
- 51 rule tests in `test_arcade.js`: speeds, cornering, pathfinding, every target rule, red zones, waves, fright, the house counters, Elroy, collisions, fruit, scoring, lives, determinism and snapshots, plus the RL wrapper.
- Fixed while testing:
  - endless recursion when the brain's view window wrapped through the tunnel;
  - "no pauses" mode ending a pause instantly, which bled into the next level (it now takes 1 frozen frame);
  - the RL wrapper returning the board from the moment of death (it now fast-forwards to the restart);
  - two tests that were testing the wrong thing: a red-zone test on a tile whose "up" is a wall, and a sticky-move test that compared a game's hash with itself.

## Four ghosts, part 2: training the brain
### One engine, three places
The rules exist once, in `arcade/arcade.js`. It runs:
- in the page (inlined by `build.js`);
- in training: `bridge.js` serves 16 games in lockstep over stdin/stdout to Python with a tiny binary protocol, about 15,800 random decisions per second on its own;
- in every check.

Three tests prove the copies agree:
- The page (frame by frame, with the real pauses) and training (no pauses) make identical decisions and see identical observations, with and without sticky moves (`test_arcade.js`).
- Python and JavaScript give the same Q-values to within 1e-5 and play a whole exam identically, move for move (`pacdqn/tests/test_arcade.py`).
- In a real browser, the page and a Node reference play the same 300 decisions and end with identical score, lives and dots (`workshop/qa.js`).

### The recipe (`pacdqn/pacdqn/arcade_train.py`)
| piece | choice | why |
|---|---|---|
| decisions | one per new tile; the stick is held until the next tile | the brain can turn as early as the rules allow, and 300+ decisions per level stay manageable |
| observation | 1,768 numbers: a 15×15 window around Pac-Man with 7 layers; maze distances along each move to the nearest dot, energizer, fruit and each ghost; 45 game facts including the hidden timers; a coarse map of the remaining dots | the wave and house timers are invisible on screen, so the brain is told them (the partial-observability problem from the 4-ghost answer) |
| network | MLP 1768 → 256 → 256, dueling head folded into one layer for export: 519,684 weights | runs in the page with the workshop's own MLP code |
| targets | Double DQN, 3-step returns, γ = 0.99, Huber loss, Adam lr 1e-4, gradient clip 10, frozen copy refreshed every 2,000 updates | the Bench's target, looking 3 moves ahead |
| replay | about 1M decisions, time-major (each state stored once), 8 samples drawn per decision collected | |
| exploration | 16 games at once, each with its own random-move chance from 40% down to 0.07% (Ape-X), after a 250k-decision warm-up | some games explore hard while others play almost cleanly |
| rewards | √points / 10 (dot 0.32, energizer 0.71, ghosts 1.4–4, fruit 1–7.1); −2 for a lost life, which also ends the target | compresses the 10-to-5,000 point range without clipping ghosts down to a dot |
| sticky stick | in training and exams, each frame has a 25% chance the stick stays where it was | the ghosts are deterministic; see below |
| exams | every 250k decisions: 10 fixed games (seeds 1000–1009), greedy, sticky stick, no learning | the practice-vs-exam lesson from the Arcade station |

Training ran at 1,550–1,700 decisions per second on the sandbox's 4 CPU cores (no GPU).

### Run v0 and the sticky-move finding
The first run (`pacdqn/runs/arcade_v0/`) trained without sticky moves and made exams sticky **per decision**: 25% of the time the previous decision's direction was held for a whole tile. Its exams:

| decisions | 250k | 500k | 750k | 1M | 1.25M | 1.5M | 1.75M | 2M |
|---|---|---|---|---|---|---|---|---|
| exam mean | 3,177 | 4,375 | 6,281 | 5,445 | 6,288 (all 10 cleared level 1) | 5,607 | **3,522** | 8,174 |

The drop to 3,522 looked like the brain falling apart, so it was investigated before going further. The same 1.75M checkpoint under three exam rules:

| exam rule | mean score | what the games looked like |
|---|---|---|
| no sticky moves | 10,370 | near-identical games, all ending on level 2: deterministic ghosts plus a deterministic brain replay one route |
| sticky per decision (v0) | 3,522 | a whole tile in the wrong direction next to a ghost is usually a death |
| sticky per frame (Machado et al. 2018) | 12,692 (median 11,935, mean level 3.3, best level 6) | varied games, and the fair measure of skill |

So the brain was fine and the exam was too harsh. Per-decision stickiness is far stronger than the research standard, which repeats the previous *frame's* input. It also differed from training, where the stick was never sticky. The fix:
- per-frame sticky moves at 25%, in training and exams alike;
- restart training from scratch;
- keep v0's logs in `runs/arcade_v0/` as the record.

The page's Watch mode uses the same sticky stick, so what you watch is what was graded. "You" mode doesn't.

### The run (`pacdqn/runs/arcade/`)
30M decisions in 5.0 hours on the sandbox's 4 CPU cores (about 1,670 decisions per second), 20,639 training games. Exam averages for each 5M decisions:

| decisions | 0–5M | 5–10M | 10–15M | 15–20M | 20–25M | 25–30M |
|---|---|---|---|---|---|---|
| exam mean (20 exams each) | 13,691 | 19,957 | 21,101 | 22,737 | 24,745 | 25,330 |
| mean level reached | 3.3 | 4.5 | 4.7 | 4.9 | 5.4 | 5.2 |

- **A fast start.** The first level was cleared in an exam at 500k decisions, and by 2.5M the exam averaged 17,402.
- **Then a slow climb.** For a long stretch, games mostly ended on level 5, where the ghosts reach 95% speed and blue time drops to 2 seconds. From about 21M decisions, exams more often averaged level 6.
- **Still rising at the end**, slowly: the last 5M decisions added about 600 points.

### Choosing the brain: the best exam was partly luck
The trainer keeps `best.pt`, the checkpoint with the best 10-game exam: 29,695 at 21.25M decisions. With 120 exams of only 10 games each, the highest one is partly luck. So that checkpoint and the final one each played 50 new games (seeds 3000–3049):

| checkpoint | mean ± standard error | median | best game | boards cleared per game | ghosts per game |
|---|---|---|---|---|---|
| best exam (21.25M) | 26,586 ± 1,063 | 26,525 | 42,310 | 4.80 | 29.6 |
| final (30M) | **28,321** ± 1,285 | 27,235 | 45,630 | 4.78 | 30.9 |

The best exam's 29,695 shrank to 26,586 on new games. The final checkpoint scored higher; the gap is about one standard error, so the two are close. It shipped as `runs/arcade/final.pt` (weights only) until the run was continued to 60M decisions (below).

### The final check: 50 games that played no part in the choice
Seeds 2000–2049, sticky stick on. The first row is the exported half-float brain played in JavaScript, exactly as the page runs it:

| | mean | median | best game | levels cleared per game | ghosts per game |
|---|---|---|---|---|---|
| **the 30M brain as the page plays it** (half floats, JavaScript) | **28,975** | 29,290 | 57,930 (level 12) | 5.02 | 31.9 |
| full precision (PyTorch), same games | 28,172 | 27,100 | 42,570 (level 9) | 4.84 | 30.9 |
| random play (100 games) | 718 | | | | |

The two rows differ only through rounding. The half-float brain made the same move on all 4,000 checked decisions (largest Q-value difference 0.011). Over 50 long games, though, an occasional near-tie still goes the other way, and one different move sends a game down a different path. The averages agree within their noise.

### Practice vs exam, once more
The practice dots sit around 16,000 points while exams run near 25,000. Before writing anything about it on the page, the finished brain replayed the training setup with learning switched off: 16 games, each with its own random-move rate, 320,000 decisions, 187 finished games.

| random-move rate | games of the 16 | finished games | share of all finished | mean score |
|---|---|---|---|---|
| 11% to 40% | 4 | 77 | 41% | 7,939 |
| 1% to 7% | 5 | 51 | 27% | 20,422 |
| under 1% | 7 | 59 | 32% | 27,245 |
| all | 16 | 187 | | 17,435 |

So practice runs low for two reasons:
- random moves are expensive (40% random averaged 5,165 points; 0.1% averaged 28,660);
- the most random games end soonest, so they make up far more than their share of the finished games that the practice dots average.

The brain itself plays at exam level. The page's note under the chart makes the same point with the shipped brain's numbers, using the 5%-or-more split.

### Continuing to 60M decisions
`--resume runs/arcade/latest.pt --steps 60000000` picked up the network, Adam's state and the step counter. The replay memory refilled from new games, and the exploration rates were already at their final values. The resumed segment's `minutes` clock restarts at 0, so the export adds the segments together (`total_minutes`, with a test). Exam averages for each 5M decisions:

| decisions | 30–35M | 35–40M | 40–45M | 45–50M | 50–55M | 55–60M |
|---|---|---|---|---|---|---|
| exam mean (20 exams each) | 27,020 | 26,400 | 26,962 | 27,334 | 28,748 | 28,415 |
| mean level reached | 5.6 | 5.5 | 5.6 | 5.5 | 5.8 | 5.8 |

In total: 60M decisions in 10.8 hours and 39,634 training games. Four checkpoints played the same 50 selection games (seeds 3000–3049):

| checkpoint | mean ± standard error | median | best game | boards cleared per game | ghosts per game |
|---|---|---|---|---|---|
| 30M (shipped until now) | 28,321 ± 1,285 | 27,235 | 45,630 | 4.78 | 30.9 |
| 39.75M (best exam at the time) | 30,008 ± 1,309 | 28,915 | 62,120 | 5.16 | 33.0 |
| **57.5M (best exam of the run)** | **31,721** ± 1,419 | 30,490 | 59,620 | 5.38 | 34.4 |
| 60M (end of training) | 30,228 ± 1,169 | 30,695 | 45,040 | 5.14 | 33.1 |

The 57.5M checkpoint now ships as `final.pt`. Its final check on the separate 50 games (seeds 2000–2049), played as the page plays it:

| same 50 games | mean | median | best game | levels cleared per game | ghosts per game |
|---|---|---|---|---|---|
| 30M brain | 28,975 | 29,290 | 57,930 (level 12) | 5.02 | 31.9 |
| **57.5M brain** | **29,654** | 27,900 | 48,000 (level 9) | 4.86 | 32.4 |

- **The gain is real but modest.** Over all 100 test games, the 57.5M brain averages 30,688 against 28,648, and it's ahead on both sets. It earns its extra points mostly by eating more ghosts; levels cleared stayed about the same.
- **Doubling the training bought about 7%.** Most of the curve's rise came in the first 5M decisions.
- **The 39.75M interim.** Brandon asked to see the newer brain before the run finished. The 39.75M checkpoint won its selection round (30,008 vs 28,321) but lost the final check (28,244 vs 28,975): a tie over 100 games (29,126 vs 28,648). It went up on the claude.ai page anyway at his request, with its own numbers on the page.

Practice vs exam for the shipped 57.5M brain, measured the same way as before (16 games with learning off, 320,000 decisions, 179 finished games):

| random-move rate | games of the 16 | finished games | share of all finished | mean score |
|---|---|---|---|---|
| 11% to 40% | 4 | 72 | 40% | 8,286 |
| 1% to 7% | 5 | 53 | 30% | 19,782 |
| under 1% | 7 | 54 | 30% | 30,599 |
| all | 16 | 179 | | 18,421 |

### Shipping the brain
- **Half floats**: the export stores each weight in 16 bits instead of 32. That turns 2.8 MB of base64 into 1.4 MB, and the page from 3 MB into 1.66 MB. The export measures what that costs on 4,000 real decisions: none of the first brain's moves changed (largest Q-value difference 0.001); one mid-run checkpoint changed 8 of 4,000, and the shipped one none (largest Q-value difference 0.011). Any change is a near-tie: a move can flip only when its two best options are within twice that difference.
- **Rounding**: the page rounds each observation to 1/255 steps, exactly like the trainer's replay memory, so the brain sees in the page what it saw in training.
- **The record**: `workshop/brains/arcade_curve.json` holds the exam curve, a thinned practice series, the random-play baseline (718 points) and a final check of 50 new games (seeds 2000–2049) that played no part in picking the checkpoint. The final check plays the exported half-float brain in JavaScript, exactly as the page does, so the numbers on the page describe the brain on the page; the full-precision network's result on the same games is stored next to it.

## Four ghosts, part 3: the Workshop station
- **The game**: the arcade's maze layout drawn in the workshop's style, original character and fruit drawings, and the real pauses (ready, ghost eaten, death, level clear).
- **Who plays**: the brain, or you (arrow keys, swipes or an on-screen pad).
- **Speed**: arcade, 3×, 10× or flat out.
- **Overlays**:
  - "Ghost targets" draws each ghost's current target square, including Inky's pivot and Clyde's 8-tile circle, with arrows for targets off the screen.
  - "What the brain sees" draws its 15×15 window.
- **Side panel**: the brain's Q-values at its latest decision, the level's wave schedule filling as the clock runs, and a card per ghost with its personality rule and what it is doing right now (counting dots in the house, scattering, chasing which tile, frightened, or eyes heading home).
- **Below the game**: the exam curve from training with practice dots, then two drawers. One explains the observation and the recipe; the other lists every rule with its source.
- **QA**: 6 new checks, 29 in total:
  - the brain loads and the screen draws;
  - the 300-decision page-vs-Node parity check;
  - your arrow presses reach the game;
  - flat-out speed;
  - the ghost cards, waves and stats;
  - the training chart.
- **Phone fixes found in screenshots**:
  - a four-tab bar that grew too tall (tabs now scroll sideways, and the selected one scrolls into view);
  - clipped overlay chips (they now wrap);
  - overlapping chart labels;
  - a status line that went stale at flat-out speed;
  - a clipped "15 × 15 view" label.

## Decisions and why
- Build a learning agent (DQN) rather than drive a pretrained model: Brandon wanted something that improves by playing.
- Original maze and characters, no Namco assets.
- Rewards in game points × 0.01 so targets stay in a sane range; Huber loss so big surprises can't yank the weights.
- The workshop uses a small MLP on a walkable-square encoding (116 inputs) instead of a CNN, so it trains in a browser in minutes.
- The Bench uses plain SGD for nudges (step size proportional to the gradient, easy to explain), a deterministic chase ghost (predictable S′), and a live Double DQN / frozen copy toggle.
- The Arcade headline metric is an exam, not practice scores.
- Experiment notes come from measured runs and say "in our test run".
- The page is one self-contained HTML file, published from claude.ai and (from the Claude Code move on) GitHub Pages via Actions. The Pages site is built from the committed `dist/`, and CI fails if that file is stale, so the file attached to claude.ai and the Pages site can't drift apart.
- Four ghosts uses **documents only**: the rules and maze layout from public documents, original artwork, and no ROM data anywhere (Brandon's call, 2026-09-25).
- **One rules engine** (`arcade/arcade.js`) for the page, training and tests, reached from Python through a bridge instead of a Python port, so the page and training can't disagree about the rules or about what the brain sees.
- **Sticky moves per frame**, in training and exams, as in the Atari research standard. Per-decision stickiness measured as far too harsh (3,522 vs 12,692 for the same brain).
- **√points rewards** with a −2 lost-life penalty: ghosts and fruit still matter more than dots, but no single reward swamps the rest.
- **Half-float brain** in the page: half the size, the same moves.
- **Choose on some games, report on others.** The shipped checkpoint was chosen on 50 games (seeds 3000–3049) and reported on 50 different ones (seeds 2000–2049), like the Arcade brain's fine-tune. The best of 120 ten-game exams overstated its checkpoint by about 3,100 points.
