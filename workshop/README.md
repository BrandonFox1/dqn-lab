# The DQN Workshop

A self-contained teaching page for Deep Q-Networks. On the first three stations everything, including training, runs in the browser: an 11,908-weight network with hand-written backpropagation, Adam, replay memory, a frozen target copy and Double DQN. The fourth plays the real arcade rules with a much bigger brain trained offline, still run by the page's own network code.

## Stations
- **The Bench**: one update, up close. Edit a mini maze, pick a move and follow the worked solution: guess, peek at the future (live network picks, frozen copy scores), target, gap and loss, then nudge the weights with plain SGD and watch the guess move. Overlays: Value map (best q per square) and Ripple (how q changed on squares you didn't train on). Brains: Trained, Fresh, From Arcade.
- **Ripple Tank**: a 7×6 grid world with a cherry (+10) and ghost (−10). Table or network brain, explore, step, sweep (synchronous update against a snapshot, so value spreads exactly one square per sweep).
- **Arcade**: full DQN training on the 9×7 maze with Watch, Fast and Turbo speeds, an exam-based learning curve, knobs, and experiments with notes measured from real ablation runs.
- **Four ghosts**: the whole 1980 arcade game (the real maze, four ghost personalities, scatter and chase waves, frightened mode, Cruise Elroy, fruit, lives, level speeds), rebuilt rule by rule from public documents in `../arcade/`, with original artwork. A half-million-weight brain trained offline in PyTorch plays it live: watch it at arcade speed, 3×, 10× or flat out, or take the stick yourself (arrow keys, swipes or the on-screen pad). Overlays show each ghost's current target square and the 15×15 window the brain sees. The side panel shows the brain's Q-value for each move, the wave schedule for the level, and each ghost's state. Below it: the recorded exam curve from training and two drawers that explain the observation, the training recipe, and every rule with its source.

## Build and test
```bash
npm install
npm test          # 19 engine tests
npm run build     # -> dist/dqn-workshop.html
npm run qa        # 29 headless Chrome checks (Linux), screenshots in shots/
node qa_extra.js  # exam schedule check + extra screenshots
node ../arcade/test_arcade.js   # 51 arcade rule tests (the engine the Four ghosts station runs)
```
`build.js` inlines `engine.js`, `../arcade/arcade.js`, `src/style.css`, `src/ui.js`, the brains in `brains/` and `src/experiment-notes.json` into one file. Commit `dist/` after building, including after any change to `../arcade/arcade.js` or the arcade brain: CI rebuilds it and fails if the committed copy differs.

Six of the 29 QA checks cover Four ghosts: the brain loads and the screen draws; the page and a Node reference play the same 300 decisions and must end with identical score, lives and dots (this catches any drift in brain decoding or the observation); your arrow presses reach the game; flat-out speed; the ghost cards, waves and stats; and the training chart.

## Publishing
`.github/workflows/workshop.yml` runs all of the above on every pull request that touches `workshop/` or `arcade/`, and on every such push to `main` publishes `dist/dqn-workshop.html` to GitHub Pages at https://brandonfox1.github.io/dqn-lab/. One-time setup: Settings → Pages → Source: GitHub Actions (a private repo needs a paid GitHub plan for Pages; the site itself is public). The claude.ai copy is republished by attaching the same file in a claude.ai chat.

## Training the brains
```bash
node train_brain.js 100000 A            # from scratch, evaluates every 5k steps, keeps the best -> brains/brain_A.*
node finetune.js 60000                  # fine-tune brain_A at lr 3e-4; select on 100 games, compare on 500 held-out -> brains/brain_final.*
bash run_ablations.sh                   # the experiment runs behind src/experiment-notes.json -> results/
node check_continue.js                  # practice scores when continuing to train a loaded brain
```
Shipped brain: `brains/brain_final` (135k steps total). Over 500 fresh test games it cleared the board 97.8% of the time, average score 407.9. Random play averages −280.

The Four ghosts brain (`brains/arcade.b64`, `arcade.json`, and `arcade_curve.json` for the chart) is trained and exported from `../pacdqn` (see its README):
```bash
cd ../pacdqn
python -m pacdqn.arcade_train train --out runs/arcade --steps 30000000
python -m pacdqn.arcade_train export --run runs/arcade --ckpt final.pt --dest ../workshop/brains/arcade
cd ../workshop && npm run build
```
Shipped Four ghosts brain: the 57.5M-decision checkpoint (10.3 hours on 4 CPU cores) of a 60M-decision run, chosen on 50 selection games. Over 50 new games (seeds 2000–2049) it scored 29,654 on average and cleared 4.86 levels per game; its best game scored 48,000 on level 9. Random play averages 718.
It is stored as half floats (16-bit), which halves the page size. The export measures how many of the brain's moves that changes, and its 50-game final check plays the exported brain in JavaScript exactly as this page does.
