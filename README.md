# dqn-lab

How a Deep Q-Network learns to play Pac-Man, built from scratch: a PyTorch agent, an interactive teaching page, and a rule-by-rule rebuild of the 1980 arcade game that a DQN learned to play.

| Folder | What it is |
|---|---|
| [`pacdqn/`](pacdqn/) | A PyTorch dueling Double-DQN agent with its own maze environment, training loop, player and tests, plus the trainer for the arcade-rules game. |
| [`workshop/`](workshop/) | **The DQN Workshop**: a single HTML page with four interactive stations (the Bench, Ripple Tank, Arcade, Four ghosts) and a hand-written JavaScript neural network. Live at https://brandonfox1.github.io/dqn-lab/ (auto-published from `main`) and https://claude.ai/artifact/83NAYFuuY2pjCfeYbm4UG4 |
| [`arcade/`](arcade/) | The 1980 Pac-Man arcade rules, frame by frame: the real maze, four ghost personalities, waves, frightened mode, Cruise Elroy, fruit and level speeds. Written from public documents only (no ROM data), with every rule sourced in [`SPEC.md`](arcade/SPEC.md) and tested. The same file runs the Workshop's Four ghosts station and the PyTorch training. |
| [`explainers/`](explainers/) | A frame-by-frame picture of one real learning step. |
| [`docs/`](docs/) | The project history, roadmap and the full conversation that produced all of this. |

## Quick start
```bash
cd pacdqn && pip install -r requirements.txt && python -m pytest -q
python -m pacdqn.play --run runs/small            # watch the demo agent in the terminal

cd ../arcade && node test_arcade.js                # the arcade rules, one test per rule (Node 18+)

cd ../workshop && npm install && npm test && npm run build
# open workshop/dist/dqn-workshop.html in a browser
```

Working with Claude Code? It reads `CLAUDE.md` automatically.
