# The DQN Workshop

A self-contained teaching page for Deep Q-Networks. Everything, including training, runs in the browser: an 11,908-weight network with hand-written backpropagation, Adam, replay memory, a frozen target copy and Double DQN.

## Stations
- **The Bench**: one update, up close. Edit a mini maze, pick a move and follow the worked solution: guess, peek at the future (live network picks, frozen copy scores), target, gap and loss, then nudge the weights with plain SGD and watch the guess move. Overlays: Value map (best q per square) and Ripple (how q changed on squares you didn't train on). Brains: Trained, Fresh, From Arcade.
- **Ripple Tank**: a 7×6 grid world with a cherry (+10) and ghost (−10). Table or network brain, explore, step, sweep (synchronous update against a snapshot, so value spreads exactly one square per sweep).
- **Arcade**: full DQN training on the 9×7 maze with Watch, Fast and Turbo speeds, an exam-based learning curve, knobs, and experiments with notes measured from real ablation runs.

## Build and test
```bash
npm install
npm test          # 19 engine tests
npm run build     # -> dist/dqn-workshop.html
npm run qa        # 23 headless Chrome checks (Linux), screenshots in shots/
node qa_extra.js  # exam schedule check + extra screenshots
```
Commit `dist/` after building: CI rebuilds it and fails if the committed copy differs.

## Publishing
`.github/workflows/workshop.yml` runs all of the above on every pull request that touches `workshop/`, and on every push to `main` publishes `dist/dqn-workshop.html` to GitHub Pages at https://brandonfox1.github.io/dqn-lab/. One-time setup: Settings → Pages → Source: GitHub Actions (a private repo needs a paid GitHub plan for Pages; the site itself is public). The claude.ai copy is republished by attaching the same file in a claude.ai chat.

## Training the brains
```bash
node train_brain.js 100000 A            # from scratch, evaluates every 5k steps, keeps the best -> brains/brain_A.*
node finetune.js 60000                  # fine-tune brain_A at lr 3e-4; select on 100 games, compare on 500 held-out -> brains/brain_final.*
bash run_ablations.sh                   # the experiment runs behind src/experiment-notes.json -> results/
node check_continue.js                  # practice scores when continuing to train a loaded brain
```
Shipped brain: `brains/brain_final` (135k steps total). Over 500 fresh test games it cleared the board 97.8% of the time, average score 407.9. Random play averages −280.
