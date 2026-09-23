"""Watch a trained agent play in the terminal, or record a GIF.

    python -m pacdqn.play --run runs/small               # animate in terminal
    python -m pacdqn.play --run runs/small --gif out.gif # record 3 episodes to a GIF
    python -m pacdqn.play --run runs/small --random      # baseline: random policy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .agent import DQNAgent
from .env import ACTION_NAMES, EnvConfig, MazeEnv, Rewards

CELL = 24  # pixels per cell in GIF frames


def load_run(run: str | Path, checkpoint: str = "best.pt", device: str | None = None):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    env_cfg_dict = dict(cfg["env"])
    env_cfg_dict["rewards"] = Rewards(**env_cfg_dict["rewards"])
    env_cfg_dict["ghost_aggression"] = tuple(env_cfg_dict["ghost_aggression"])
    env_cfg = EnvConfig(**env_cfg_dict)
    ckpt = run / checkpoint
    if not ckpt.exists():
        ckpt = run / "latest.pt"
    agent = DQNAgent.load(ckpt, device=device)
    return agent, env_cfg


def frame_to_image(env: MazeEnv, q: np.ndarray | None = None):
    from PIL import Image, ImageDraw

    h, w = env.height, env.width
    img = Image.new("RGB", (w * CELL, h * CELL + 18), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            x0, y0 = x * CELL, y * CELL
            if env.walls[y, x]:
                d.rectangle([x0 + 1, y0 + 1, x0 + CELL - 2, y0 + CELL - 2], fill=(40, 60, 200))
            elif env.pellets[y, x]:
                c = CELL // 2
                d.ellipse([x0 + c - 2, y0 + c - 2, x0 + c + 2, y0 + c + 2], fill=(255, 220, 160))
            elif env.powers[y, x]:
                c = CELL // 2
                d.ellipse([x0 + c - 6, y0 + c - 6, x0 + c + 6, y0 + c + 6], fill=(255, 200, 120))
    for i, g in enumerate(env.ghosts):
        if g.active:
            y, x = g.pos
            color = (70, 90, 255) if env.power_timer > 0 else [(255, 60, 60), (255, 150, 200), (60, 220, 220), (255, 160, 60)][i % 4]
            d.rounded_rectangle([x * CELL + 3, y * CELL + 3, x * CELL + CELL - 3, y * CELL + CELL - 3], radius=8, fill=color)
    y, x = env.player
    d.ellipse([x * CELL + 3, y * CELL + 3, x * CELL + CELL - 3, y * CELL + CELL - 3], fill=(250, 230, 60))
    text = f"score {env.score:.0f}  left {env.pellets_left}  t {env.steps}"
    if q is not None:
        text += "  " + " ".join(f"{n[0]}{v:+.1f}" for n, v in zip(ACTION_NAMES, q))
    d.text((4, h * CELL + 2), text, fill=(230, 230, 230))
    return img


def run_episode(env: MazeEnv, act, seed: int, delay: float, frames: list | None, show_q):
    obs, info = env.reset(seed=seed)
    done, ret = False, 0.0
    while not done:
        q = show_q(obs) if show_q else None
        if frames is not None:
            frames.append(frame_to_image(env, q))
        if delay >= 0:
            sys.stdout.write("\x1b[2J\x1b[H" + env.render() + "\n")
            if q is not None:
                sys.stdout.write("  ".join(f"{n}:{v:+.2f}" for n, v in zip(ACTION_NAMES, q)) + "\n")
            sys.stdout.flush()
            time.sleep(delay)
        obs, r, term, trunc, info = env.step(act(obs))
        ret += r
        done = term or trunc
    if frames is not None:
        frames.append(frame_to_image(env))
    return ret, info


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Watch or record a trained agent.")
    p.add_argument("--run", default="runs/default", help="run directory produced by pacdqn.train")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eps", type=float, default=0.0, help="exploration during play")
    p.add_argument("--delay", type=float, default=0.08, help="seconds per frame in the terminal; -1 to disable")
    p.add_argument("--gif", default=None, help="write episodes to this GIF instead of animating")
    p.add_argument("--random", action="store_true", help="play a uniformly random policy instead")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    agent, env_cfg = load_run(args.run, args.checkpoint, args.device)
    env = MazeEnv(env_cfg)
    rng = np.random.default_rng(args.seed)
    if args.random:
        act = lambda obs: int(rng.integers(env.n_actions))  # noqa: E731
        show_q = None
    else:
        act = lambda obs: agent.act(obs, epsilon=args.eps)  # noqa: E731
        show_q = agent.q_values

    frames = [] if args.gif else None
    delay = -1 if args.gif else args.delay
    results = []
    for ep in range(args.episodes):
        ret, info = run_episode(env, act, args.seed + ep, delay, frames, show_q)
        results.append((ret, info["score"], info["won"], info["steps"]))
        print(f"episode {ep + 1}: return {ret:.0f}  score {info['score']:.0f}  won={info['won']}  steps={info['steps']}")
    print(f"mean score {np.mean([r[1] for r in results]):.1f}  win rate {np.mean([r[2] for r in results]):.0%}")

    if frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=90, loop=0)
        print(f"wrote {args.gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
