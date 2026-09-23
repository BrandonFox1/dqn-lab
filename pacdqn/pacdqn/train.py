"""Train a Double DQN agent on the maze environment.

    python -m pacdqn.train --maze small --ghosts 1 --steps 150000 --out runs/small

Writes to ``--out``:
  config.json   full env + agent config
  log.csv       one row per finished episode (step, return, score, won, eps, loss)
  eval.csv      one row per evaluation (step, mean_return, mean_score, win_rate)
  latest.pt     most recent checkpoint
  best.pt       checkpoint with the best evaluation mean_score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env import EnvConfig, MazeEnv


@dataclass
class EvalResult:
    mean_return: float
    mean_score: float
    win_rate: float
    mean_steps: float


def evaluate(agent: DQNAgent, env_cfg: EnvConfig, episodes: int, seed: int, epsilon: float = 0.02) -> EvalResult:
    env = MazeEnv(env_cfg)
    returns, scores, wins, lengths = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, epsilon=epsilon))
            ret += r
            done = term or trunc
        returns.append(ret)
        scores.append(info["score"])
        wins.append(float(info["won"]))
        lengths.append(info["steps"])
    return EvalResult(float(np.mean(returns)), float(np.mean(scores)), float(np.mean(wins)), float(np.mean(lengths)))


@dataclass
class TrainConfig:
    steps: int = 150_000
    seed: int = 0
    out: str = "runs/default"
    device: str | None = None
    eval_every: int = 10_000
    eval_episodes: int = 10
    log_every: int = 20  # print every N episodes
    resume: bool = False  # continue from <out>/latest.pt (replay buffer is refilled, not restored)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: TrainConfig, quiet: bool = False) -> DQNAgent:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    env = MazeEnv(cfg.env)
    resuming = cfg.resume and (out / "latest.pt").exists()
    if resuming:
        agent = DQNAgent.load(out / "latest.pt", device=cfg.device, seed=cfg.seed + 1)
        if agent.obs_shape != env.observation_shape or agent.n_actions != env.n_actions:
            raise ValueError("checkpoint in --out does not match this environment")
    else:
        agent = DQNAgent(env.observation_shape, env.n_actions, cfg.agent, device=cfg.device, seed=cfg.seed)
    start_step = agent.steps
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if not quiet:
        print(f"device={agent.device}  maze={cfg.env.maze} {env.height}x{env.width}  ghosts={cfg.env.n_ghosts}  "
              f"params={sum(p.numel() for p in agent.online.parameters()):,}")

    episode, best_score = 0, -float("inf")
    if resuming:
        episode, best_score = _read_progress(out)
        if not quiet:
            print(f"resuming from step {start_step:,} (episode {episode}, best eval score {best_score:.1f})")
    mode = "a" if resuming else "w"
    log_f = open(out / "log.csv", mode, newline="")
    log = csv.writer(log_f)
    eval_f = open(out / "eval.csv", mode, newline="")
    ev = csv.writer(eval_f)
    if not resuming:
        log.writerow(["step", "episode", "return", "score", "won", "length", "epsilon", "loss"])
        ev.writerow(["step", "mean_return", "mean_score", "win_rate", "mean_steps"])

    obs, _ = env.reset(seed=cfg.seed + start_step)
    ep_return, ep_losses = 0.0, []
    recent_returns: list[float] = []
    t0 = time.time()
    next_eval = (start_step // cfg.eval_every + 1) * cfg.eval_every

    try:
        for step in range(start_step + 1, cfg.steps + 1):
            action = agent.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            # Only true terminals stop bootstrapping; time-limit truncation does not.
            loss = agent.observe(obs, action, reward, next_obs, term)
            if loss is not None:
                ep_losses.append(loss)
            ep_return += reward
            obs = next_obs

            if term or trunc:
                episode += 1
                mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
                log.writerow([step, episode, f"{ep_return:.1f}", f"{info['score']:.0f}", int(info["won"]),
                              info["steps"], f"{agent.epsilon():.3f}", f"{mean_loss:.5f}"])
                recent_returns.append(ep_return)
                if not quiet and episode % cfg.log_every == 0:
                    sps = (step - start_step) / max(1e-9, time.time() - t0)
                    print(f"step {step:>8,}  ep {episode:>5}  avg_return(last {cfg.log_every}) "
                          f"{np.mean(recent_returns[-cfg.log_every:]):>8.1f}  eps {agent.epsilon():.3f}  "
                          f"loss {mean_loss:.4f}  {sps:,.0f} steps/s")
                obs, _ = env.reset()
                ep_return, ep_losses = 0.0, []

            if step >= next_eval or step == cfg.steps:
                next_eval += cfg.eval_every
                res = evaluate(agent, cfg.env, cfg.eval_episodes, seed=10_000 + step)
                ev.writerow([step, f"{res.mean_return:.1f}", f"{res.mean_score:.1f}", f"{res.win_rate:.2f}", f"{res.mean_steps:.1f}"])
                eval_f.flush()
                log_f.flush()
                agent.save(out / "latest.pt")
                if res.mean_score > best_score:
                    best_score = res.mean_score
                    agent.save(out / "best.pt")
                if not quiet:
                    print(f"  [eval @ {step:,}] mean_score {res.mean_score:.1f}  win_rate {res.win_rate:.0%}  "
                          f"mean_len {res.mean_steps:.0f}{'  *best*' if res.mean_score >= best_score else ''}")
    finally:
        log_f.close()
        eval_f.close()
        agent.save(out / "latest.pt")
    return agent


def _read_progress(out: Path) -> tuple[int, float]:
    """Episode count and best eval score recorded so far in ``out`` (for --resume)."""
    episode, best = 0, -float("inf")
    log_p, eval_p = out / "log.csv", out / "eval.csv"
    if log_p.exists():
        with open(log_p) as f:
            for row in csv.DictReader(f):
                episode = int(row["episode"])
    if eval_p.exists():
        with open(eval_p) as f:
            for row in csv.DictReader(f):
                best = max(best, float(row["mean_score"]))
    return episode, best


def _add_dataclass_args(parser: argparse.ArgumentParser, dc, prefix: str = "", skip=()) -> None:
    for f in fields(dc):
        if f.name in skip:
            continue
        default = f.default if f.default is not f.default_factory else None  # type: ignore[attr-defined]
        if isinstance(default, (int, float, str)) and not isinstance(default, bool):
            parser.add_argument(f"--{prefix}{f.name.replace('_', '-')}", type=type(default), default=default,
                                help=f"(default: {default})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Double DQN agent on a Pac-Man-style maze.")
    p.add_argument("--maze", default="small", help="small | medium")
    p.add_argument("--ghosts", type=int, default=1)
    p.add_argument("--lives", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--power-duration", type=int, default=20)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue training from <out>/latest.pt")
    _add_dataclass_args(p, AgentConfig)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    agent_kwargs = {f.name: getattr(args, f.name) for f in fields(AgentConfig) if hasattr(args, f.name)}
    env_cfg = EnvConfig(maze=args.maze, n_ghosts=args.ghosts, lives=args.lives,
                        max_steps=args.max_episode_steps, power_duration=args.power_duration)
    return TrainConfig(steps=args.steps, seed=args.seed, out=args.out, device=args.device,
                       eval_every=args.eval_every, eval_episodes=args.eval_episodes, resume=args.resume,
                       env=env_cfg, agent=AgentConfig(**agent_kwargs))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    train(config_from_args(args), quiet=args.quiet)


if __name__ == "__main__":
    main()
