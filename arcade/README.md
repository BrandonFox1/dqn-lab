# arcade: the 1980 Pac-Man rules, frame by frame

`arcade.js` is a simulation of the arcade game's rules at 60 frames per second: the real 28×31 maze, the four ghost personalities, scatter/chase waves, frightened mode, Cruise Elroy, the ghost house counters, fruit, lives and level speeds. It was written from public documents only, mainly *The Pac-Man Dossier*. No ROM data of any kind is used (no graphics, sounds or program bytes).

- **[SPEC.md](SPEC.md)** lists every rule with its source (Dossier page or figure), plus the handful of choices made where the documents are silent.
- **`test_arcade.js`** checks each rule, and each test names the SPEC rule it covers.

The same file runs in three places, which is why there is only one copy of the rules:

| where | how |
|---|---|
| the Workshop page (Four ghosts station) | inlined by `workshop/build.js` as the global `Arcade` |
| training (PyTorch, `pacdqn/pacdqn/arcade_train.py`) | `bridge.js` serves many games over stdin/stdout to `arcade_env.py` |
| checks in Node | `test_arcade.js`, `eval_brain.js`, `workshop/qa.js` |

## Files
```
arcade.js       the game (Game) and the decision-per-tile wrapper for RL (ArcadeEnv)
SPEC.md         every rule, with its source
test_arcade.js  51 tests, one or more per rule
vec.js          many games in lockstep (shared by the bridge and eval_brain.js, so exams match move for move)
bridge.js       binary stdin/stdout server for the Python trainer
eval_brain.js   plays an exported brain in pure JavaScript, with the Workshop's own MLP
```

## Commands (Node 18+)
```bash
node test_arcade.js                                  # 51 rule tests
node eval_brain.js ../workshop/brains/arcade         # 10 exam games with the shipped brain (sticky moves)
node eval_brain.js ../workshop/brains/arcade --games 50 --seed-base 2000   # the final check: mean 28,975, best game level 12 (~50 s)
```
Training, exams and export live in pacdqn (see its README):
```bash
cd ../pacdqn
python -m pacdqn.arcade_train train --out runs/arcade --steps 30000000
python -m pacdqn.arcade_train exam --run runs/arcade --games 50
python -m pacdqn.arcade_train export --run runs/arcade --ckpt final.pt --dest ../workshop/brains/arcade
```

## The game API
```js
const g = new Arcade.Game({ seed: 1, pauses: true }); // pauses: the arcade's ready, ghost-eaten, death and level-clear pauses
g.step(Arcade.UP);                                    // one frame with the stick held up (or null); returns that frame's events
g.pac, g.ghosts, g.dots, g.score, g.level, g.lives, g.phase, g.scatter, g.frightTimer, g.targetOf(ghost), g.elroy()
```
Positions are the actors' center pixels. Their tile is `x >> 3, y >> 3`, and a tile's center pixel is at offset (3, 4) inside the tile (SPEC 1.3).

## The RL wrapper (`ArcadeEnv`)
- **One decision per tile.** The agent picks up, down, left or right, the same action order as the Workshop's MiniPac. The stick is held until Pac-Man enters a new tile (or for 8 frames if he is standing at a wall), so turns get the full cornering advantage.
- **Observation: 1,768 numbers in [0, 1].**
  - A 15×15 window around Pac-Man with 7 layers: wall, dot, energizer, ghost, ghost heading, blue ghost, fruit.
  - For each move, the maze distances (not back through Pac-Man) to the nearest dot, energizer and fruit, and to each ghost, dangerous or blue.
  - 45 global facts: heading, timers, each ghost's state and offset, Elroy, level, dots, lives.
  - A 7×8 map of the remaining dots.
- **Page and training are the same.** `begin(action)` + `tick()` is the frame-by-frame form the page uses with the real pauses on; `step(action)` is what training uses, with no pauses. A test feeds both the same moves and checks that every observation and result is identical.
- Changing the observation, the actions or the rewards means retraining the brain.
