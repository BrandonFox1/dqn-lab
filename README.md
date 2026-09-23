# dqn-lab

Two ways to see how a Deep Q-Network learns to play a Pac-Man-style game, both built from scratch.

| Folder | What it is |
|---|---|
| [`pacdqn/`](pacdqn/) | A PyTorch dueling Double-DQN agent with its own maze environment, training loop, player and tests. |
| [`workshop/`](workshop/) | **The DQN Workshop**: a single HTML page with three interactive stations (the Bench, Ripple Tank, Arcade) and a hand-written JavaScript neural network. Live at https://claude.ai/artifact/83NAYFuuY2pjCfeYbm4UG4 |
| [`explainers/`](explainers/) | A frame-by-frame picture of one real learning step. |
| [`docs/`](docs/) | The project history, roadmap and the full conversation that produced all of this. |

## Quick start
```bash
cd pacdqn && pip install -r requirements.txt && python -m pytest -q
python -m pacdqn.play --run runs/small            # watch the demo agent in the terminal

cd ../workshop && npm install && npm test && npm run build
# open workshop/dist/dqn-workshop.html in a browser
```

Working with Claude Code? It reads `CLAUDE.md` automatically.
