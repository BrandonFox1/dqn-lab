import csv
import json

import numpy as np

from pacdqn import play, train
from pacdqn.agent import AgentConfig
from pacdqn.env import EnvConfig
from pacdqn.train import TrainConfig, evaluate


def test_train_smoke_writes_artifacts(tmp_path):
    cfg = TrainConfig(
        steps=300, seed=0, out=str(tmp_path / "run"), device="cpu", eval_every=150, eval_episodes=2,
        env=EnvConfig(maze="small", n_ghosts=1, max_steps=40),
        agent=AgentConfig(buffer_size=500, learning_starts=50, batch_size=8, target_update=50, hidden=32),
    )
    agent = train.train(cfg, quiet=True)
    assert agent.steps == 300
    run = tmp_path / "run"
    assert (run / "latest.pt").exists() and (run / "best.pt").exists()
    conf = json.loads((run / "config.json").read_text())
    assert conf["env"]["maze"] == "small" and conf["agent"]["hidden"] == 32
    with open(run / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 300 // 40 - 1
    assert all(float(r["epsilon"]) <= 1.0 for r in rows)
    with open(run / "eval.csv") as f:
        ev = list(csv.DictReader(f))
    assert [int(r["step"]) for r in ev] == [150, 300]


def test_evaluate_returns_reasonable_stats():
    cfg = EnvConfig(maze="small", n_ghosts=1, max_steps=30)
    agent = train.DQNAgent((8, 11, 13), 4, AgentConfig(hidden=16, buffer_size=10), device="cpu", seed=0)
    res = evaluate(agent, cfg, episodes=3, seed=0, epsilon=1.0)
    assert 0.0 <= res.win_rate <= 1.0
    assert 1 <= res.mean_steps <= 30
    assert np.isfinite(res.mean_return) and np.isfinite(res.mean_score)


def test_cli_train_then_play_gif(tmp_path, capsys):
    out = tmp_path / "cli"
    train.main(["--maze", "small", "--ghosts", "1", "--steps", "120", "--out", str(out), "--device", "cpu",
                "--eval-every", "60", "--eval-episodes", "1", "--max-episode-steps", "30",
                "--buffer-size", "300", "--learning-starts", "20", "--batch-size", "4", "--hidden", "16", "--quiet"])
    gif = tmp_path / "play.gif"
    play.main(["--run", str(out), "--episodes", "1", "--gif", str(gif), "--device", "cpu"])
    assert gif.exists() and gif.stat().st_size > 0
    captured = capsys.readouterr().out
    assert "mean score" in captured
    play.main(["--run", str(out), "--episodes", "1", "--random", "--delay", "-1", "--device", "cpu"])
    assert "episode 1" in capsys.readouterr().out


def test_resume_continues_from_checkpoint(tmp_path):
    out = tmp_path / "resume"
    base = dict(seed=0, out=str(out), device="cpu", eval_every=100, eval_episodes=1,
                env=EnvConfig(maze="small", n_ghosts=1, max_steps=25),
                agent=AgentConfig(buffer_size=300, learning_starts=20, batch_size=4, target_update=50, hidden=16))
    train.train(TrainConfig(steps=200, **base), quiet=True)
    with open(out / "log.csv") as f:
        n_before = len(list(csv.DictReader(f)))
    agent = train.train(TrainConfig(steps=400, resume=True, **base), quiet=True)
    assert agent.steps == 400
    with open(out / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > n_before
    assert int(rows[-1]["step"]) == 400 or int(rows[-1]["step"]) > 200
    assert int(rows[-1]["episode"]) > int(rows[n_before - 1]["episode"])  # episode numbering continues
    with open(out / "eval.csv") as f:
        ev = list(csv.DictReader(f))
    assert [int(r["step"]) for r in ev] == [100, 200, 300, 400]
