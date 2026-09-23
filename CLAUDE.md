# dqn-lab: Claude Code handoff

This repo continues a long claude.ai conversation (2026-09-23) in which Brandon and Claude built two things from scratch:

1. **`pacdqn/`**: a PyTorch dueling Double-DQN agent that learns an original Pac-Man-style maze by playing it.
2. **`workshop/`**: "The DQN Workshop", one self-contained HTML page that teaches how a DQN learns. It has a hand-written JavaScript neural network, three interactive stations and a pretrained brain.

Read **`docs/HISTORY.md`** before changing anything: it has the full story, every measured result and why each design decision was made. `docs/chat-log.md` is the conversation itself. `docs/ROADMAP.md` is what comes next (including the arcade-faithful 4-ghost version).

## Who you're working with
- Brandon, Operations Manager at Clean Team (commercial cleaning). Self-described ML beginner who learns fast. Explain in plain language with concrete numbers; define jargon on first use.
- He often reviews from his phone. Keep summaries short: lead with what changed, what was verified, and what to look at.
- His standard: finished, tested, documented work rather than plans. Run it before you claim it. If a result looks surprising, investigate before writing anything that depends on it (the "practice vs exam" story in HISTORY is the model).
- Windows user. Uses Claude Code desktop with a cloud environment.

## First session in a fresh environment
If `dqn-lab.zip` is sitting at the repo root, the repo was bootstrapped by upload: unzip it into the root (`unzip -o dqn-lab.zip && rm dqn-lab.zip`), run the checks below, and commit.

```bash
cd pacdqn && pip install -r requirements.txt && python -m pytest -q     # 34 tests: 33 pass, 1 skips unless gymnasium is installed
cd ../workshop && npm install && npm test && npm run build && npm run qa # 19 engine tests, build, 23 browser checks
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
pacdqn/                   Python project (own README with every CLI flag)
  pacdqn/{env,mazes,model,replay,agent,train,play}.py
  tests/                  pytest suite
  runs/small/             demo run: config, logs, weights-only best.pt, learning_curve.png, play.gif
workshop/
  engine.js               MLP + backprop + SGD/Adam, MiniPac env, GridWorld, TableQ/NetQ, DQN (shared by tests and page)
  test_engine.js          19 tests
  src/index.html|style.css|ui.js   the page (template with placeholders)
  src/experiment-notes.json        Arcade "Watch for" text, written from results/abl_*.txt
  brains/                 brain_A (85k steps) and brain_final (fine-tuned, shipped)
  build.js                inlines everything into dist/dqn-workshop.html
  qa.js, qa_extra.js      headless Chrome QA
  train_brain.js, finetune.js, check_continue.js, run_ablations.sh
  results/                training/fine-tune/ablation logs
  dist/dqn-workshop.html  the built page (what's published)
.github/workflows/workshop.yml   PR checks for workshop/ + GitHub Pages deploy on main
```

## Commands
- pacdqn: `python -m pytest -q` · `python -m pacdqn.train --maze small --ghosts 1 --steps 150000 --out runs/small` · `python -m pacdqn.play --run runs/small [--gif out.gif]`
- workshop (Node 18+): `npm test` · `npm run build` · `npm run qa` · `node qa_extra.js` · `node train_brain.js STEPS TAG '{json cfg}'` · `node finetune.js STEPS` · `bash run_ablations.sh`

## Workshop invariants (break these and things silently go wrong)
- `engine.js` is the single source of truth. It sets `module.exports` in Node and a global `Engine` in the page, so tests and page run identical code. Change it → `npm test`.
- `build.js` substitutes `'__BRAIN_B64__'`, `__BRAIN_META__` and `'__NOTARGET__'`-style placeholders and exits non-zero if any `__PLACEHOLDER__` is left.
- Brain format: base64 of one Float32Array, layers in order W0,b0,W1,b1,W2,b2, each W row-major (out × in). Shape MLP [116, 64, 64, 4] = 11,908 weights. Changing the observation, rewards or architecture means retraining (`train_brain.js`, ~100k steps, ~4 min in Node) and re-running the fine-tune.
- MiniPac: 9×7 maze, 29 walkable squares, 27 pellets. Observation = 4 blocks of 29 (pellet, Pac-Man, ghost, ghost's previous square). Actions 0 up, 1 down, 2 left, 3 right. Rewards in points: pellet +10, step −1, caught −300, cleared +200, multiplied by 0.01 for training. 150-step limit. Ghost takes the BFS-shortest step with probability 0.75 and can't reverse; the Bench uses a deterministic chase ghost.
- Arcade learning curve = the **exam**: every 2,500 steps, 10 fixed greedy games (seeds 1000–1009) with no learning. Practice games (ε-greedy) are faint dots. Never go back to plotting training scores as the headline: at ε = 0.05 the trained brain's win rate falls from 99% to 36% purely from random moves.
- Experiment notes are single-seed measurements; phrase them as "in our test run".
- The page must stay self-contained: published claude.ai pages only allow a few script CDNs plus Google Fonts, and no network calls.
- QA hooks: `window.__workshop = {bench, tank, arcade, selectTab}`; `arcade._train(n)` runs n training steps synchronously.

## Design system
- Light theme = engineering notebook on graph paper; dark = blueprint. Game screens are always dark navy.
- Color = meaning everywhere: guess yellow/amber, target green, gap red, live network blue, frozen copy purple, reward orange.
- Font: Recursive (casual axis for headings, mono axis for numbers). Sentence case, no all-caps labels, no arrows on buttons. Numbered margin steps only for true sequences.
- Original characters and maze only; no Namco assets or likenesses.

## QA gotchas
- `qa.js` uses `@sparticuz/chromium`, a Linux x64 build: fine in cloud environments, not on Windows (swap in `puppeteer` locally).
- Coordinate clicks can be intercepted by the sticky tab bar; use `page.$eval(sel, e => e.click())` for buttons deep in the page.
- Buttons inside closed `<details>` (the Arcade knobs drawer) must be opened before clicking.

## Publishing
Two homes for the page, both serving the same `workshop/dist/dqn-workshop.html`:
- **GitHub Pages** (https://brandonfox1.github.io/dqn-lab/): `.github/workflows/workshop.yml` publishes it on every push to `main` that touches `workshop/`. Nothing to do by hand: merge the PR.
- **claude.ai** (https://claude.ai/artifact/83NAYFuuY2pjCfeYbm4UG4): Claude Code can't update claude.ai artifacts. Attach `workshop/dist/dqn-workshop.html` in a claude.ai chat and ask to republish it to that link.

The workflow also gates every PR that touches `workshop/`: engine tests, build, **committed `dist/` must equal a fresh build** (so always run `npm run build` and commit `dist/`), `qa.js`, and `qa_extra.js`. QA screenshots are uploaded as the `qa-screenshots` artifact on each run.
- Pages needs a one-time switch: Settings → Pages → Source: GitHub Actions. The repo is private, which needs a paid GitHub plan for Pages (or make the repo public). A Pages site is public either way. Until it's on, the deploy job skips with a warning rather than failing; after switching it on, run Actions → Workshop → Run workflow.

## pacdqn notes
- `runs/small/best.pt` is weights-only (optimizer state stripped, 14.8 MB → 4.9 MB); `DQNAgent.load` skips the optimizer when absent. `latest.pt` isn't included, so `--resume` on runs/small isn't available: start a new run.
- Demo run (60k steps, 1 CPU core): greedy eval −460 at 10k → −39 at 60k; best.pt +54 mean over 20 seeds vs random −478; 0% clears. Still rising steeply: train 300k+ on a GPU.
