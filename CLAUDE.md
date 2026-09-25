# dqn-lab: Claude Code handoff

This repo continues a long claude.ai conversation (2026-09-23) in which Brandon and Claude built, from scratch:

1. **`pacdqn/`**: a PyTorch dueling Double-DQN agent that learns an original Pac-Man-style maze by playing it, plus the trainer for the arcade-rules game.
2. **`workshop/`**: "The DQN Workshop", one self-contained HTML page that teaches how a DQN learns. It has a hand-written JavaScript neural network, four interactive stations and pretrained brains.
3. **`arcade/`** (added in Claude Code, 2026-09-25): the 1980 Pac-Man arcade rules as a frame-by-frame JavaScript engine, written from public documents only. It powers the Workshop's **Four ghosts** station and the pacdqn trainer.

Read **`docs/HISTORY.md`** before changing anything: it has the full story, every measured result and why each design decision was made. `docs/chat-log.md` is the conversation itself. `docs/ROADMAP.md` is what comes next.

## Who you're working with
- Brandon, Operations Manager at Clean Team (commercial cleaning). Self-described ML beginner who learns fast. Explain in plain language with concrete numbers; define jargon on first use.
- He often reviews from his phone. Keep summaries short: lead with what changed, what was verified, and what to look at.
- His standard: finished, tested, documented work rather than plans. Run it before you claim it. If a result looks surprising, investigate before writing anything that depends on it (the "practice vs exam" story in HISTORY is the model).
- Windows user. Uses Claude Code desktop with a cloud environment.

## First session in a fresh environment
If `dqn-lab.zip` is sitting at the repo root, the repo was bootstrapped by upload: unzip it into the root (`unzip -o dqn-lab.zip && rm dqn-lab.zip`), run the checks below, and commit.

```bash
cd pacdqn && pip install -r requirements.txt && python -m pytest -q     # 41 tests: 40 pass, 1 skips unless gymnasium is installed (needs node)
cd ../arcade && node test_arcade.js                                      # 51 arcade rule tests
cd ../workshop && npm install && npm test && npm run build && npm run qa # 19 engine tests, build, 29 browser checks
# npm install is blocked by some cloud sandboxes; qa.js then needs puppeteer-core + @sparticuz/chromium from elsewhere
```

## Repo map
```
CLAUDE.md                 this file (auto-loaded)
README.md                 human overview
docs/HISTORY.md           story, results, decisions, gotchas
docs/ROADMAP.md           next steps
docs/chat-log.md          cleaned conversation; chat-transcript-raw.txt = raw export of the first part
explainers/               the step-by-step explainer GIF/PNG built from a real transition of pacdqn's best.pt
arcade/                   the 1980 arcade rules (own README)
  arcade.js               Game (one frame per step) + ArcadeEnv (one decision per tile; the RL wrapper)
  SPEC.md                 every rule, numbered, with its source; test_arcade.js cites the numbers
  vec.js, bridge.js       many games in lockstep; binary stdin/stdout server for the Python trainer
  eval_brain.js           plays an exported brain in pure JS with the Workshop's MLP
pacdqn/                   Python project (own README with every CLI flag)
  pacdqn/{env,mazes,model,replay,agent,train,play}.py
  pacdqn/arcade_env.py    client for arcade/bridge.js; pacdqn/arcade_train.py = Four ghosts trainer/exam/export
  tests/                  pytest suite (test_arcade.py: replay maths, folded net, bridge, Python/JS parity)
  runs/small/             demo run: config, logs, weights-only best.pt, learning_curve.png, play.gif
  runs/arcade/            the Four ghosts run: config, log.csv, exam.csv; weights-only best.pt (best 10-game exam) and final.pt (shipped)
  runs/arcade_v0/         the first, stopped run (harsh sticky exams; see HISTORY)
workshop/
  engine.js               MLP + backprop + SGD/Adam, MiniPac env, GridWorld, TableQ/NetQ, DQN (shared by tests and page)
  test_engine.js          19 tests
  src/index.html|style.css|ui.js   the page (template with placeholders)
  src/experiment-notes.json        Arcade "Watch for" text, written from results/abl_*.txt
  brains/                 brain_A (85k steps) and brain_final (fine-tuned, shipped); arcade.{b64,json} + arcade_curve.json (Four ghosts)
  build.js                inlines everything into dist/dqn-workshop.html
  qa.js, qa_extra.js      headless Chrome QA
  train_brain.js, finetune.js, check_continue.js, run_ablations.sh
  results/                training/fine-tune/ablation logs
  dist/dqn-workshop.html  the built page (what's published)
.github/workflows/workshop.yml   PR checks for workshop/ and arcade/ + GitHub Pages deploy on main
.github/workflows/pacdqn.yml     pytest (CPU torch) + arcade rule tests on PRs touching pacdqn/ or arcade/
```

## Commands
- pacdqn: `python -m pytest -q` · `python -m pacdqn.train --maze small --ghosts 1 --steps 150000 --out runs/small` · `python -m pacdqn.play --run runs/small [--gif out.gif]`
- workshop (Node 18+): `npm test` · `npm run build` · `npm run qa` · `node qa_extra.js` · `node train_brain.js STEPS TAG '{json cfg}'` · `node finetune.js STEPS` · `bash run_ablations.sh`
- arcade: `node test_arcade.js` · `node eval_brain.js ../workshop/brains/arcade [--games 50 --seed-base 2000]`
- Four ghosts brain (from pacdqn/): `python -m pacdqn.arcade_train train --out runs/arcade --steps 30000000` · `... exam --run runs/arcade --games 50` · `... export --run runs/arcade --ckpt final.pt --dest ../workshop/brains/arcade` (then `npm run build` in workshop/)

## Workshop invariants (break these and things silently go wrong)
- `engine.js` is the single source of truth. It sets `module.exports` in Node and a global `Engine` in the page, so tests and page run identical code. Change it → `npm test`.
- `build.js` substitutes `'__BRAIN_B64__'`, `__BRAIN_META__` and `'__NOTARGET__'`-style placeholders and exits non-zero if any `__PLACEHOLDER__` is left.
- Brain format: base64 of one Float32Array, layers in order W0,b0,W1,b1,W2,b2, each W row-major (out × in). Shape MLP [116, 64, 64, 4] = 11,908 weights. Changing the observation, rewards or architecture means retraining (`train_brain.js`, ~100k steps, ~4 min in Node) and re-running the fine-tune.
- MiniPac: 9×7 maze, 29 walkable squares, 27 pellets. Observation = 4 blocks of 29 (pellet, Pac-Man, ghost, ghost's previous square). Actions 0 up, 1 down, 2 left, 3 right. Rewards in points: pellet +10, step −1, caught −300, cleared +200, multiplied by 0.01 for training. 150-step limit. Ghost takes the BFS-shortest step with probability 0.75 and can't reverse; the Bench uses a deterministic chase ghost.
- Arcade learning curve = the **exam**: every 2,500 steps, 10 fixed greedy games (seeds 1000–1009) with no learning. Practice games (ε-greedy) are faint dots. Never go back to plotting training scores as the headline: at ε = 0.05 the trained brain's win rate falls from 99% to 36% purely from random moves.
- Experiment notes are single-seed measurements; phrase them as "in our test run".
- The page must stay self-contained: published claude.ai pages only allow a few script CDNs plus Google Fonts, and no network calls.
- QA hooks: `window.__workshop = {bench, tank, arcade, four, selectTab}`; `arcade._train(n)` runs n training steps synchronously; `four.runDecisions(n)` plays n brain decisions synchronously.

## Four ghosts / arcade invariants
- **Documents only.** Brandon chose: no ROM files and nothing derived from ROMs (no graphics, sounds, program bytes), not in the repo and not on the page. Rules come from the Pac-Man Dossier (PDF in floooh/pacman.c); the maze *layout* is the arcade's; all art is original. Where the documents are silent, the choice is listed at the end of `arcade/SPEC.md`.
- `arcade/arcade.js` is the single source of the rules for the page (inlined by build.js), training (via bridge.js) and all checks. Change a rule → update SPEC.md and a test in `test_arcade.js`.
- Training and the page make identical decisions: `ArcadeEnv.step()` (training, no pauses) and `begin()/tick()` (page, real pauses) are proven equal by a test, with and without sticky moves. Keep it that way: frozen frames never count toward a decision except the last frame of a ghost-eaten pause.
- The observation is 1,768 numbers (`Arcade.OBS`). Changing the observation, actions, rewards or decision rule means retraining (about 5 h on 4 cores for 30M decisions).
- Brain: `workshop/brains/arcade.b64` is half floats (`format: "f16"` in arcade.json) of the folded MLP [1768, 256, 256, 4], layers W0,b0,W1,b1,W2,b2 like the other brains. The page rounds each observation to 1/255 steps exactly like the trainer's replay memory. The shipped checkpoint is the one with the best 10-game exam (`best.pt`); arcade.json records its decisions (`steps`), its own training time (`hours`) and exam, and `arcade_curve.json` holds the whole run plus a 50-game final check on seeds 2000–2049 that played no part in choosing it. The final check plays the exported half-float brain in JavaScript (`eval_brain.js`), exactly as the page runs it; the full-precision result on the same games sits next to it as `full_precision`.
- The exam: 10 greedy games, seeds 1000–1009, sticky moves 0.25 **per frame** (the stick stays put that frame). Per-decision stickiness was far too harsh (see HISTORY). Watch mode uses the same sticky stick; "You" mode doesn't.
- qa.js replays 300 decisions of a seed in the page and in Node and requires identical score, lives and dots, which catches brain-decoding and observation drift.

## Design system
- Light theme = engineering notebook on graph paper; dark = blueprint. Game screens are always dark navy.
- Color = meaning everywhere: guess yellow/amber, target green, gap red, live network blue, frozen copy purple, reward orange.
- Font: Recursive (casual axis for headings, mono axis for numbers). Sentence case, no all-caps labels, no arrows on buttons. Numbered margin steps only for true sequences.
- Original characters and art only; no Namco assets or likenesses. The MiniPac/pacdqn mazes are original; the Four ghosts maze layout follows the arcade (from the documents), drawn in the workshop's style.

## QA gotchas
- `qa.js` uses `@sparticuz/chromium`, a Linux x64 build: fine in cloud environments, not on Windows (swap in `puppeteer` locally).
- Coordinate clicks can be intercepted by the sticky tab bar; use `page.$eval(sel, e => e.click())` for buttons deep in the page.
- Buttons inside closed `<details>` (the Arcade knobs drawer) must be opened before clicking.

## Publishing
Two homes for the page, both serving the same `workshop/dist/dqn-workshop.html`:
- **GitHub Pages** (https://brandonfox1.github.io/dqn-lab/): `.github/workflows/workshop.yml` publishes it on every push to `main` that touches `workshop/` or `arcade/`. Nothing to do by hand: merge the PR.
- **claude.ai** (https://claude.ai/artifact/83NAYFuuY2pjCfeYbm4UG4, Brandon's private artifact): Claude Code sessions that have the Artifact tool republish it directly. Read the artifact first (`action: read` with its `url`; a publish to an artifact the session hasn't read is refused), strip the page's own doctype/`<html>`/`<head>`/`<body>` wrapper (the host adds its own skeleton), then publish with the same `url`. Never ask Brandon to move the file himself.

The workflow also gates every PR that touches `workshop/` or `arcade/`: engine tests, arcade rule tests, build, **committed `dist/` must equal a fresh build** (so always run `npm run build` and commit `dist/`, also after changing arcade.js), `qa.js`, and `qa_extra.js`. QA screenshots are uploaded as the `qa-screenshots` artifact on each run.
- Pages needs a one-time switch: Settings → Pages → Source: GitHub Actions. The repo is private, which needs a paid GitHub plan for Pages (or make the repo public). A Pages site is public either way. Until it's on, the deploy job skips with a warning rather than failing; after switching it on, run Actions → Workshop → Run workflow.

## pacdqn notes
- `runs/small/best.pt` is weights-only (optimizer state stripped, 14.8 MB → 4.9 MB); `DQNAgent.load` skips the optimizer when absent. `latest.pt` isn't included, so `--resume` on runs/small isn't available: start a new run.
- Demo run (60k steps, 1 CPU core): greedy eval −460 at 10k → −39 at 60k; best.pt +54 mean over 20 seeds vs random −478; 0% clears. Still rising steeply: train 300k+ on a GPU.
- Four ghosts run (30M decisions, 5.0 h, 4 CPU cores): exam averages 13.7k (first 5M) → 25.3k (last 5M), still rising slowly. The shipped `final.pt` won a 50-game selection (seeds 3000–3049) against `best.pt`. Its final check on 50 new games (seeds 2000–2049) scored 28,975 mean, 5.02 levels cleared per game, best game level 12; random play scores 718.
