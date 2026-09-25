"""Train a DQN on the arcade-rules game and export it for the Workshop page.

    python -m pacdqn.arcade_train train --out runs/arcade [--steps 20000000]
    python -m pacdqn.arcade_train exam --run runs/arcade [--games 50] [--sticky 0.25]
    python -m pacdqn.arcade_train export --run runs/arcade --dest ../workshop/brains/arcade

The game runs in the JavaScript engine (see arcade_env.py); this file only holds the learner:

* Network: an MLP over the 1,768-number observation with a dueling head (value + advantages). The head is folded
  into one plain linear layer for export, so the page's hand-written MLP runs it unchanged.
* Double DQN targets with 3-step returns; a replay memory of about a million decisions; a frozen copy
  refreshed every 2,000 updates.
* Exploration: each of the 16 games has its own random-move chance, from 40% down to under 0.1%
  (the Ape-X recipe), reached after a short warm-up at 100%.
* Rewards: sqrt(points)/10 per decision (a dot 0.32, an energizer 0.71, ghosts 1.4 to 4, fruit 1 to 7.1)
  and -2 for a lost life. A lost life ends the target (no bootstrapping), as in the Atari DQN work.
  Scores and exams are always reported in real arcade points.
* Sticky moves (Machado et al. 2018), in training and exams: each frame there is a 25% chance the stick stays
  where it was, so a new direction sometimes lands a frame or two late. The ghosts are deterministic, so without
  this every exam game would be nearly the same game and a memorised route could fake skill.
* Exams: 10 fixed games (seeds 1000-1009) played greedily.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .arcade_env import REPO, ArcadeVecEnv


@dataclass
class ArcadeConfig:
    envs: int = 16
    steps: int = 20_000_000
    hidden: list = field(default_factory=lambda: [256, 256])
    lr: float = 1e-4
    gamma: float = 0.99
    n_step: int = 3
    batch: int = 128
    replay_ratio: float = 8.0  # samples drawn per decision collected
    buffer: int = 1_000_000
    learning_starts: int = 50_000
    target_update: int = 2_000  # gradient updates between frozen-copy refreshes
    eps_base: float = 0.4
    eps_alpha: float = 7.0
    eps_warmup: int = 250_000
    death_penalty: float = -2.0
    sticky: float = 0.25  # per-frame sticky moves in training games too (Machado et al. 2018)
    grad_clip: float = 10.0
    exam_every: int = 250_000
    exam_games: int = 10
    exam_sticky: float = 0.25
    exam_max_decisions: int = 50_000
    seed: int = 1
    threads: int = 3


def shaped_reward(points: np.ndarray, life_lost: np.ndarray, cfg: ArcadeConfig) -> np.ndarray:
    return (np.sqrt(np.maximum(points, 0.0)) / 10.0 + cfg.death_penalty * life_lost).astype(np.float32)


# ---------------------------------------------------------------- network
class ArcadeQNet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int = 4, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], obs_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)

    def folded_layers(self):
        """[(W, b), ...] of a plain MLP computing exactly the same Q-values (dueling head folded into one layer)."""
        out = [(m.weight.detach().clone(), m.bias.detach().clone()) for m in self.trunk if isinstance(m, nn.Linear)]
        wa, ba = self.advantage.weight.detach(), self.advantage.bias.detach()
        wv, bv = self.value.weight.detach(), self.value.bias.detach()
        out.append((wa - wa.mean(dim=0, keepdim=True) + wv, ba - ba.mean() + bv))
        return out


# ----------------------------------------------------------------- replay
class VecReplay:
    """Time-major replay for N parallel games: obs[t, e] is the state before decision t in game e.

    n-step targets read the following slots of the same game, stopping at the first lost life or game over,
    so the next state is never stored twice (half the memory of storing (s, s') pairs).
    """

    def __init__(self, capacity: int, n_envs: int, obs_size: int, n_step: int, gamma: float, seed: int = 0):
        self.T = max(n_step + 2, capacity // n_envs)
        self.N, self.n, self.gamma = n_envs, n_step, gamma
        self.obs = np.zeros((self.T, n_envs, obs_size), dtype=np.uint8)
        self.act = np.zeros((self.T, n_envs), dtype=np.int64)
        self.rew = np.zeros((self.T, n_envs), dtype=np.float32)
        self.term = np.zeros((self.T, n_envs), dtype=np.bool_)
        self.total = 0  # slots written so far
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return min(self.total, self.T) * self.N

    def add(self, obs, actions, rewards, terminal) -> None:
        s = self.total % self.T
        self.obs[s], self.act[s], self.rew[s], self.term[s] = obs, actions, rewards, terminal
        self.total += 1

    def sample(self, batch: int):
        lo = max(0, self.total - self.T)
        hi = self.total - 1 - self.n  # slot t+n must already hold an observation
        t = self.rng.integers(lo, hi + 1, size=batch)
        e = self.rng.integers(0, self.N, size=batch)
        ret = np.zeros(batch, dtype=np.float32)
        alive = np.ones(batch, dtype=np.bool_)
        for k in range(self.n):
            s = (t + k) % self.T
            ret += alive * (self.gamma ** k) * self.rew[s, e]
            alive &= ~self.term[s, e]
        disc = alive * (self.gamma ** self.n)
        s0, sn = t % self.T, (t + self.n) % self.T
        return self.obs[s0, e], self.act[s0, e], ret, self.obs[sn, e], disc.astype(np.float32)


# ------------------------------------------------------------------ exams
@torch.no_grad()
def run_exam(net: ArcadeQNet, games: int = 10, sticky: float = 0.25, seed_base: int = 1000, max_decisions: int = 50_000,
             device: str = "cpu") -> dict:
    env = ArcadeVecEnv(games, seed_base=seed_base, sticky=sticky, auto_reset=False)
    active = np.ones(games, dtype=np.bool_)
    score = np.zeros(games, dtype=np.int64)
    level = np.ones(games, dtype=np.int64)
    ghosts = np.zeros(games, dtype=np.int64)
    obs = env.obs
    for _ in range(max_decisions):
        q = net(torch.from_numpy(obs.astype(np.float32)).to(device) / 255.0)
        res = env.step(q.argmax(dim=1).cpu().numpy())
        score = np.where(active, res.score, score)
        level = np.where(active, res.level, level)
        ghosts += active * res.ghosts
        active &= ~res.game_over
        obs = res.obs
        if not active.any():
            break
    env.close()
    return {
        "mean_score": float(score.mean()), "median_score": float(np.median(score)), "max_score": int(score.max()),
        "mean_level": float(level.mean()), "max_level": int(level.max()), "levels_cleared": int((level - 1).sum()),
        "ghosts_per_game": float(ghosts.mean()), "unfinished": int(active.sum()), "scores": score.tolist(), "levels": level.tolist(),
    }


# ------------------------------------------------------------------ train
def epsilons(step: int, cfg: ArcadeConfig) -> np.ndarray:
    final = cfg.eps_base ** (1 + cfg.eps_alpha * np.arange(cfg.envs) / max(1, cfg.envs - 1))
    f = min(1.0, step / max(1, cfg.eps_warmup))
    return 1.0 + f * (final - 1.0)


def train(cfg: ArcadeConfig, out: Path, resume: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = ArcadeVecEnv(cfg.envs, seed_base=cfg.seed * 1_000_000, sticky=cfg.sticky)
    online = ArcadeQNet(env.obs_size, env.n_actions, cfg.hidden)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr, eps=1.5e-4)
    steps = updates = 0
    best = -math.inf
    if resume is not None:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        online.load_state_dict(ck["online"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        steps, updates, best = ck["steps"], ck["updates"], ck.get("best", -math.inf)
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    buf = VecReplay(cfg.buffer, cfg.envs, env.obs_size, cfg.n_step, cfg.gamma, seed=cfg.seed)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log_new = not (out / "log.csv").exists() or resume is None
    log = open(out / "log.csv", "w" if log_new else "a", newline="")
    exam_log = open(out / "exam.csv", "w" if log_new else "a", newline="")
    lw, ew = csv.writer(log), csv.writer(exam_log)
    if log_new:
        lw.writerow(["steps", "updates", "minutes", "decisions_per_s", "eps_mean", "loss", "q_mean", "games", "practice_score", "practice_level"])
        ew.writerow(["steps", "minutes", "mean_score", "median_score", "max_score", "mean_level", "max_level", "levels_cleared", "ghosts_per_game", "unfinished"])

    obs = env.obs
    t0 = time.time()
    last_log, last_steps, next_exam = t0, steps, (steps // cfg.exam_every + 1) * cfg.exam_every
    recent_scores, recent_levels, games = [], [], 0
    losses, qs = [], []
    upd_credit = 0.0
    loss_fn = nn.SmoothL1Loss()
    while steps < cfg.steps:
        eps = epsilons(steps, cfg)
        with torch.no_grad():
            q = online(torch.from_numpy(obs.astype(np.float32)) / 255.0)
        actions = q.argmax(dim=1).numpy()
        qs.append(float(q.max(dim=1).values.mean()))
        rand = rng.random(cfg.envs) < eps
        actions = np.where(rand, rng.integers(0, env.n_actions, size=cfg.envs), actions).astype(np.uint8)
        res = env.step(actions)
        buf.add(obs, actions, shaped_reward(res.points, res.life_lost, cfg), res.life_lost | res.game_over)
        obs = res.obs
        steps += cfg.envs
        for i in np.flatnonzero(res.game_over):
            games += 1
            recent_scores.append(int(res.score[i])); recent_levels.append(int(res.level[i]))
        recent_scores, recent_levels = recent_scores[-100:], recent_levels[-100:]

        if len(buf) >= cfg.learning_starts:
            upd_credit += cfg.envs * cfg.replay_ratio / cfg.batch
            while upd_credit >= 1.0:
                upd_credit -= 1.0
                o, a, r, o2, disc = buf.sample(cfg.batch)
                o = torch.from_numpy(o.astype(np.float32)) / 255.0
                o2 = torch.from_numpy(o2.astype(np.float32)) / 255.0
                a, r, disc = torch.as_tensor(a), torch.as_tensor(r), torch.as_tensor(disc)
                qa = online(o).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():  # Double DQN: the live network picks, the frozen copy scores
                    a2 = online(o2).argmax(dim=1, keepdim=True)
                    tgt = r + disc * target(o2).gather(1, a2).squeeze(1)
                loss = loss_fn(qa, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                opt.step()
                updates += 1
                losses.append(loss.item())
                if updates % cfg.target_update == 0:
                    target.load_state_dict(online.state_dict())

        now = time.time()
        if now - last_log >= 30:
            dps = (steps - last_steps) / (now - last_log)
            lw.writerow([steps, updates, round((now - t0) / 60, 2), round(dps), round(float(eps.mean()), 4),
                         round(float(np.mean(losses)), 5) if losses else "", round(float(np.mean(qs)), 3) if qs else "",
                         games, round(float(np.mean(recent_scores)), 1) if recent_scores else "",
                         round(float(np.mean(recent_levels)), 2) if recent_levels else ""])
            log.flush()
            last_log, last_steps, losses, qs = now, steps, [], []

        if steps >= next_exam:
            next_exam += cfg.exam_every
            online.eval()
            ex = run_exam(online, cfg.exam_games, cfg.exam_sticky, max_decisions=cfg.exam_max_decisions)
            online.train()
            ew.writerow([steps, round((time.time() - t0) / 60, 2), ex["mean_score"], ex["median_score"], ex["max_score"], ex["mean_level"],
                         ex["max_level"], ex["levels_cleared"], ex["ghosts_per_game"], ex["unfinished"]])
            exam_log.flush()
            ck = {"online": online.state_dict(), "optimizer": opt.state_dict(), "steps": steps, "updates": updates,
                  "config": asdict(cfg), "obs_size": env.obs_size, "exam": ex, "best": max(best, ex["mean_score"])}
            torch.save(ck, out / "latest.pt")
            if ex["mean_score"] > best:
                best = ex["mean_score"]
                torch.save({k: v for k, v in ck.items() if k != "optimizer"}, out / "best.pt")
            print(f"[{steps:>10,}] exam mean {ex['mean_score']:.0f} (best {best:.0f}) level {ex['mean_level']:.2f} "
                  f"practice {np.mean(recent_scores) if recent_scores else 0:.0f}", flush=True)
    env.close()
    log.close()
    exam_log.close()


# ----------------------------------------------------------------- export
def load_net(path: Path) -> tuple[ArcadeQNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    net = ArcadeQNet(ck["obs_size"], 4, cfg.get("hidden", [256, 256]))
    net.load_state_dict(ck["online"])
    net.eval()
    return net, ck


def export_brain(ckpt: Path, dest: Path, fmt: str = "f32", extra: dict | None = None) -> dict:
    """Write dest.b64 (W0,b0,W1,b1,... row-major, the Workshop brain format; float32 or half floats) and dest.json."""
    net, ck = load_net(ckpt)
    layers = net.folded_layers()
    flat = np.concatenate([np.concatenate([w.numpy().ravel(), b.numpy().ravel()]) for w, b in layers]).astype("<f4")
    sizes = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    raw = flat.astype("<f2").tobytes() if fmt == "f16" else flat.tobytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".b64").write_text(base64.b64encode(raw).decode("ascii"))
    meta = {"sizes": sizes, "weights": int(flat.size), "format": fmt, "steps": int(ck["steps"]), "exam": ck.get("exam"), **(extra or {})}
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def half_agreement(ckpt: Path, games: int = 8, decisions: int = 4000) -> dict:
    """How often rounding the weights to half floats changes the brain's move, measured on states it really visits."""
    net, _ = load_net(ckpt)
    half = copy.deepcopy(net)
    with torch.no_grad():
        for p in half.parameters():
            p.copy_(p.half().float())
    same = total = 0
    worst = 0.0
    env = ArcadeVecEnv(games, seed_base=7000)
    obs = env.obs
    for _ in range(decisions // games):
        x = torch.from_numpy(obs.astype(np.float32)) / 255.0
        with torch.no_grad():
            q, qh = net(x), half(x)
        same += int((q.argmax(1) == qh.argmax(1)).sum()); total += games
        worst = max(worst, float((q - qh).abs().max()))
        obs = env.step(q.argmax(1).numpy()).obs
    env.close()
    return {"decisions": total, "same_move": same / total, "max_q_diff": worst}


def js_exam(brain: Path, games: int, seed_base: int, sticky: float = 0.25, max_decisions: int = 50_000) -> dict:
    """Play an exported brain exactly as the Workshop page runs it: the stored (half-float) weights in the page's own
    hand-written MLP, with observations rounded like the trainer's (arcade/eval_brain.js)."""
    out = subprocess.run([os.environ.get("NODE", "node"), str(REPO / "arcade" / "eval_brain.js"), str(brain), "--games", str(games),
                          "--seed-base", str(seed_base), "--sticky", str(sticky), "--max", str(max_decisions)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def random_baseline(games: int = 100, seed_base: int = 5000) -> float:
    env = ArcadeVecEnv(games, seed_base=seed_base, auto_reset=False)
    rng = np.random.default_rng(seed_base)
    score = np.zeros(games)
    active = np.ones(games, dtype=np.bool_)
    while active.any():
        r = env.step(rng.integers(0, 4, games))
        score = np.where(active, r.score, score)
        active &= ~r.game_over
    env.close()
    return float(score.mean())


def total_minutes(rows: list[dict]) -> list[float]:
    """Training minutes since the run began, for each log row. A resumed run restarts its clock at 0, so each
    segment is added onto the time the earlier segments had reached."""
    out, offset, last = [], 0.0, 0.0
    for r in rows:
        m = float(r["minutes"])
        if m < last:  # the clock restarted: a resumed segment
            offset += last
        last = m
        out.append(offset + m)
    return out


def checkpoint_hours(run: Path, steps: int) -> float | None:
    """Training time (hours) when the checkpoint at `steps` was saved, from the exam log."""
    with open(run / "exam.csv") as f:
        rows = list(csv.DictReader(f))
    for r, m in zip(rows, total_minutes(rows)):
        if int(r["steps"]) == steps:
            return round(m / 60, 2)
    return None


def export_record(run: Path, dest: Path, final: dict | None = None, random_score: float | None = None) -> dict:
    """The learning curve for the page: exam points, a thinned practice series, and a summary."""
    with open(run / "exam.csv") as f:
        exams = [[int(r["steps"]), float(r["mean_score"]), float(r["mean_level"])] for r in csv.DictReader(f)]
    with open(run / "log.csv") as f:
        log = list(csv.DictReader(f))
    minutes = total_minutes(log)[-1] if log else 0.0
    rows = [r for r in log if r["practice_score"]]
    stride = max(1, len(rows) // 400)
    practice = [[int(r["steps"]), round(float(r["practice_score"]))] for r in rows[::stride]]
    rec = {"exams": exams, "practice": practice, "random": round(random_score if random_score is not None else random_baseline()),
           "summary": {"hours": round(minutes / 60, 2), "decisions": exams[-1][0] if exams else 0, **({"final": final} if final else {})}}
    dest.write_text(json.dumps(rec, separators=(",", ":")))
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--resume", type=Path)
    for k, v in asdict(ArcadeConfig()).items():
        if k != "hidden":
            t.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    t.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    e = sub.add_parser("exam")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--ckpt", default="best.pt")
    e.add_argument("--games", type=int, default=50)
    e.add_argument("--sticky", type=float, default=0.25)
    e.add_argument("--seed-base", type=int, default=2000)
    x = sub.add_parser("export")
    x.add_argument("--run", type=Path, required=True)
    x.add_argument("--ckpt", default="best.pt")
    x.add_argument("--dest", type=Path, required=True, help="path without extension, e.g. ../workshop/brains/arcade")
    x.add_argument("--format", choices=["f32", "f16"], default="f16")
    x.add_argument("--final-games", type=int, default=50, help="fresh sticky-action games for the summary (0 to skip)")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        cfg = ArcadeConfig(**{k: getattr(args, k) for k in asdict(ArcadeConfig())})
        train(cfg, args.out, args.resume)
    elif args.cmd == "exam":
        net, _ = load_net(args.run / args.ckpt)
        print(json.dumps(run_exam(net, args.games, args.sticky, seed_base=args.seed_base), indent=1))
    else:
        ck = args.run / args.ckpt
        agree = half_agreement(ck) if args.format == "f16" else None
        steps = int(torch.load(ck, map_location="cpu", weights_only=False)["steps"])
        extra = {"hours": checkpoint_hours(args.run, steps), **({"half_agreement": agree} if agree else {})}
        meta = export_brain(ck, args.dest, args.format, extra)
        final = None
        if args.final_games:  # fresh games that played no part in choosing the checkpoint
            summary = lambda r: {k: v for k, v in r.items() if k not in ("scores", "levels", "seconds")}
            page = js_exam(args.dest, args.final_games, seed_base=2000)  # the brain exactly as the page plays it
            net, _ = load_net(ck)
            full = run_exam(net, args.final_games, 0.25, seed_base=2000)  # the same games at full precision
            final = {"games": args.final_games, **summary(page), "full_precision": summary(full)}
        rec = export_record(args.run, args.dest.with_name(args.dest.name + "_curve.json"), final)
        print(json.dumps({"meta": meta, "summary": rec["summary"], "random": rec["random"]}, indent=1))


if __name__ == "__main__":
    main()
