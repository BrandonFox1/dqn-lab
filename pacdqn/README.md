# pacdqn — a Double DQN agent that learns a Pac-Man-style maze from scratch

A complete, tested, dependency-light deep Q-learning project: an original
Pac-Man-style grid environment (pellets, power pellets, ghosts that chase and
flee), a dueling Double-DQN agent in PyTorch, a training loop with evaluation
and checkpoints, and a player that renders the agent in your terminal or to a GIF.

The agent starts knowing nothing and improves purely by playing — the opposite of
a pretrained decision model like Jev.

```
pacdqn/
  pacdqn/
    mazes.py    maze layouts + parser/validator (BFS connectivity check)
    env.py      MazeEnv — Gymnasium-style API, 8-channel grid observation
    model.py    QNetwork — conv trunk + dueling value/advantage heads
    replay.py   ReplayBuffer — uniform replay, uint8 storage
    agent.py    DQNAgent — ε-greedy, Double DQN target, Huber loss, hard target sync
    train.py    CLI training loop, CSV logs, eval, best/latest checkpoints
    play.py     watch a checkpoint in the terminal or record a GIF
    arcade_env.py    many arcade-rules games at once, run by ../arcade/arcade.js through Node
    arcade_train.py  the Four ghosts trainer: dueling Double DQN, 3-step returns, exams, export
  tests/        pytest suite (env rules, replay, model, agent math, CLI smoke, arcade trainer + JS parity)
  runs/small/   a finished demo run (config, logs, checkpoints, GIF)
  runs/arcade/  the Four ghosts run (config, logs, exams, weights-only best.pt)
  run.bat       Windows: install deps, run tests, train, record GIF
```

## Quick start

```bat
pip install -r requirements.txt
python -m pytest -q                      # 42 tests (1 skips without gymnasium), ~10 s on CPU
python -m pacdqn.train --maze small --ghosts 1 --steps 150000 --out runs/small
python -m pacdqn.play  --run runs/small  # watch it in the terminal
python -m pacdqn.play  --run runs/small --gif runs/small/play.gif
```

Or on Windows just double-click `run.bat` (it does all of the above).

If you have an NVIDIA GPU, install the CUDA build of PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cu124`) — the
trainer auto-selects `cuda` when available and is ~10-30× faster.

## How the pieces fit — the DQN loop

```
                 ┌────────────────────────────────────────────────────┐
                 │  for each env step:                                │
   obs ─────────►│  1. a = ε-greedy(Q_online(obs))         agent.act  │
                 │  2. obs', r, done = env.step(a)          env.step  │
                 │  3. buffer.add(obs, a, r·scale, obs', done)        │
                 │  4. every 4 steps: sample batch, one SGD step      │
                 │        target = r + γ·(1-done)·Q_target(s', a*)    │
                 │                 where a* = argmax Q_online(s')     │
                 │        loss = Huber(Q_online(s,a), target)         │
                 │  5. every 1000 steps: Q_target ← Q_online          │
                 └────────────────────────────────────────────────────┘
```

Each idea and why it's there:

| Piece | Why |
|---|---|
| **Replay buffer** | Breaks the correlation between consecutive frames; lets each transition be reused many times. |
| **ε-greedy, decayed** | Explore randomly at first (ε=1), exploit later (ε=0.05). Linear decay over `eps_decay_steps`. |
| **Target network** | The regression target moves every update otherwise; a frozen copy makes training stable. |
| **Double DQN** | Plain DQN's `max` overestimates Q. Choosing the action with the online net and *scoring* it with the target net removes most of that bias. |
| **Dueling head** | Splits Q into V(s) + A(s,a). Learns "this state is bad" without needing to see every action from it. |
| **Huber loss + grad clipping** | Large TD errors (dying = -5) don't blow up the gradient. |
| **Update every 4 steps** | `train_every=4`, as in the original DQN paper. Cheaper, and consecutive frames are near-duplicates anyway. |
| **Reward scaling** | Env rewards are in game points (pellet 10, death -500). The agent trains on `points × 0.01`, keeping targets in a sane range. |
| **Terminal vs. truncated** | Death stops bootstrapping; hitting the step limit does not (the state isn't actually terminal). |

## The environment

Original Pac-Man-style maze (no Namco assets). Legend in `mazes.py`; two layouts
ship (`small` 11×13, `medium` 18×19 with a 3-cell ghost house). Add your own by
writing a string — the parser validates that it's enclosed and every pellet is
reachable.

* **Actions** 0 up · 1 down · 2 left · 3 right. Walking into a wall = stay put.
* **Ghosts** move every step (`ghost_speed`), can't reverse mid-corridor (like
  the arcade), and each has an *aggression* — the probability it takes the
  BFS-shortest step toward you instead of a random one. Ghost 1 is 0.9, ghost 2
  is 0.7, … so more ghosts = more varied pressure.
* **Power pellets** make ghosts flee (they maximise distance) at half speed for
  `power_duration` steps; eating one sends it home for `ghost_respawn` steps.
* **Rewards** (points): pellet +10, power +50, ghost +200, clear +500, death
  -500, every step -1.
* **Observation** `float32 (8, H, W)`: walls, pellets, power pellets, player,
  dangerous ghosts, frightened ghosts, ghosts' previous cells (gives heading),
  and a uniform channel with the power-timer fraction. It's Markov enough that
  no frame stacking is needed.
* `make_gym_env()` returns a proper `gymnasium.Env` if you have gymnasium
  installed (for Stable-Baselines3 etc.). Not required.

## Reading the results

`train.py` writes `log.csv` (one row per episode) and `eval.csv` (every
`--eval-every` steps, 10 greedy episodes with ε=0.02). Quick plot:

```python
import pandas as pd, matplotlib.pyplot as plt
ev = pd.read_csv("runs/small/eval.csv")
ev.plot(x="step", y=["mean_score", "win_rate"], subplots=True); plt.show()
```

### Included demo run (`runs/small/`, 60k steps, single CPU core, ~12 min)

| env steps | greedy eval score (20 eps) | survival (steps) |
|---|---|---|
| random policy | -478 | ~10 |
| 10k | -460 | 11 |
| 20k | -338 | 51 |
| 30k | -276 | 63 |
| 40k | -239 | 59 |
| 50k | -205 | 59 |
| 60k | **-39** | 82 |

`best.pt` scores +54 mean over 20 fresh seeds (vs. -478 random) — it eats 20-30
pellets and evades the ghost for most of the episode. It hasn't learned to clear
the maze yet (0% wins); the curve is still rising steeply at 60k, so run
`--steps 300000` in a fresh run for a much stronger agent. (The shipped `best.pt`
is weights-only and `latest.pt` isn't included, so this demo run can't be resumed.) `learning_curve.png` and `play.gif` are in the same folder.

Training was interrupted once and finished with `--resume` — the checkpoint holds
the network and step counter; the replay buffer is refilled on restart.

## Tuning knobs (all CLI flags)

```
--maze small|medium   --ghosts N   --lives N   --max-episode-steps N   --power-duration N
--steps N             --eval-every N            --eval-episodes N       --seed N   --device cpu|cuda
--gamma --lr --batch-size --buffer-size --learning-starts --train-every --target-update
--eps-start --eps-end --eps-decay-steps --reward-scale --grad-clip --hidden
```

Curriculum that works: `small` + 1 ghost first (150k steps), then `medium` + 2
ghosts with `--eps-decay-steps 300000 --steps 1000000 --buffer-size 200000`.

## The arcade-rules version (the Workshop's Four ghosts)

`arcade_train.py` trains on the real 1980 rules: the 28×31 maze, four ghost personalities, waves, frightened mode, Cruise Elroy, fruit and level speeds. The game itself is `../arcade/arcade.js`, the same file the Workshop page runs. Python starts `node ../arcade/bridge.js` and trades small binary messages with it, so training and the page can't disagree about the rules or about what the network sees. You need Node 18+ on the PATH.

```bash
python -m pacdqn.arcade_train train --out runs/arcade --steps 30000000   # ~1,600 decisions/s on 4 CPU cores
python -m pacdqn.arcade_train exam --run runs/arcade --games 50          # sticky-move exam on fresh seeds
python -m pacdqn.arcade_train export --run runs/arcade_ft/combined --ckpt final.pt --dest ../workshop/brains/arcade
```

| piece | choice |
|---|---|
| decisions | one per tile Pac-Man enters; the stick is held until the next tile |
| observation | 1,768 numbers: 15×15 window × 7 layers, per-move maze distances, 45 game facts, a coarse dot map (see `../arcade/README.md`) |
| network | MLP 1768 → 256 → 256 with a dueling head, folded into a plain 256 → 4 layer for export |
| targets | Double DQN with 3-step returns, γ = 0.99, frozen copy refreshed every 2,000 updates |
| replay | about a million decisions, stored time-major (each state stored once) |
| exploration | 16 games at once, each with its own random-move chance from 40% down to 0.07% (Ape-X) |
| rewards | sqrt(points) / 10, and −2 for a lost life; a lost life ends the target |
| sticky stick | in training and exams, each frame has a 25% chance the stick stays where it was, so a new direction can land a frame or two late (Machado et al., 2018); this stops a memorized route from passing for skill |
| exams | every 250,000 decisions: 10 fixed games (seeds 1000–1009), greedy, sticky stick on, no learning |
| export | half floats in the Workshop brain format. The export measures how often that changes a move, then plays a final check of 50 new games (seeds 2000–2049) with the exported brain in JavaScript, exactly as the page runs it, and the same games at full precision |

`tests/test_arcade.py` checks the replay maths, the folded network, the bridge, and that an exported brain gives identical Q-values and plays identical games in Python and in JavaScript.

**Results (`runs/arcade`, 30M decisions, 5.0 hours on 4 CPU cores).** Exam averages rose from 13,691 over the first 5M decisions to 25,330 over the last 5M, still climbing slowly. `best.pt` (the best 10-game exam, 29,695 at 21.25M) and the final checkpoint each played 50 selection games (seeds 3000–3049): 26,586 vs 28,321, so the final one ships as `final.pt`. Its final check on 50 new games (seeds 2000–2049), played exactly as the page runs it, scored 28,975 on average (median 29,290), cleared 5.02 levels per game, and its best game reached level 12. Random play scores 718. 

**Continued to 60M decisions** (`--resume runs/arcade/latest.pt --steps 60000000`, another 5.8 hours). Exam averages crept from 27,020 (30–35M) to 28,415 (55–60M). Four checkpoints played the 50 selection games: 30M 28,321, 39.75M 30,008, 57.5M 31,721, 60M 30,228. The 57.5M one now ships as `final.pt`. Its final check scored 29,654 on average and cleared 4.86 levels per game; its best game scored 48,000 on level 9. Over all 100 test games it averages 30,688, against 28,648 for the 30M brain: about 7% better for twice the training.

**Fine-tuned at a lower learning rate** (`runs/arcade_ft`: `--resume runs/arcade/final.pt --lr 3e-5 --keep-every 2500000 --steps 67500000`, 1.2 hours). Smaller steps let the weights settle. On the selection games, the 65M checkpoint scored 36,336 against 31,721 for its starting point, and it ships. Its final check scored 33,720 on average, cleared 6.06 levels per game, and its best game scored 68,310 on level 15. That's about 14% better over all 100 test games, from an eighth of the extra training. `--keep-every N` keeps a weights-only checkpoint every N decisions for comparisons like this. The full story is in `docs/HISTORY.md`.

## Where to go next

1. **Prioritized replay** — sample transitions with large TD error more often. `replay.py` is 60 lines; swap the uniform `sample` for a sum-tree.
2. **n-step returns** — bootstrap from `r₁ + γr₂ + γ²r₃ + γ³Q(s₄)` instead of one step. Faster credit assignment for the delayed death penalty. (`arcade_train.py` already does this with n = 3; its `VecReplay` is a working example.)
3. **Noisy nets / distributional (C51)** — the rest of the Rainbow paper.
4. **Real Atari Ms. Pac-Man** — `pip install "gymnasium[atari]"` (ale-py bundles the ROMs). Wrap frames to `(4, 84, 84)` with `gymnasium.wrappers.AtariPreprocessing` + `FrameStackObservation`; `QNetwork` already accepts any `(C, H, W)`, so the agent is unchanged. Expect ~10M frames on a GPU for good play.
5. **Compare to a System-1 model** — feed `env.render()` to Jev each step and ask "which direction?" — no training, instant policy, no improvement. Same harness, opposite philosophy.

## Reference

Mnih et al. 2015 (DQN) · van Hasselt et al. 2016 (Double DQN) · Wang et al. 2016 (Dueling) · Hessel et al. 2018 (Rainbow) · Horgan et al. 2018 (Ape-X, the per-game exploration rates) · Machado et al. 2018 (sticky actions for honest Atari evaluation) · Pittman, *The Pac-Man Dossier* (the arcade rules).
