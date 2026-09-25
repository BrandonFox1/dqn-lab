"""The arcade-rules trainer: replay maths, the folded network, the Node bridge, and Python/JavaScript parity."""
import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from pacdqn.arcade_env import BRIDGE, REPO, ArcadeVecEnv
from pacdqn.arcade_train import (ArcadeConfig, ArcadeQNet, VecReplay, checkpoint_hours, export_brain, export_record, js_exam,
                                  run_exam, shaped_reward)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_folded_network_gives_identical_q_values():
    torch.manual_seed(0)
    net = ArcadeQNet(30, 4, [16, 8])
    x = torch.rand(64, 30)
    h = x
    for i, (w, b) in enumerate(net.folded_layers()):
        h = h @ w.T + b
        if i < 2:
            h = torch.relu(h)
    torch.testing.assert_close(h, net(x), atol=1e-5, rtol=1e-5)


def test_replay_n_step_returns_stop_at_terminals():
    buf = VecReplay(capacity=40, n_envs=2, obs_size=1, n_step=3, gamma=0.5, seed=0)
    rewards = [[1, 10], [2, 20], [4, 40], [8, 80], [16, 160], [32, 320], [0, 0]]
    terms = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
    for t, (r, d) in enumerate(zip(rewards, terms)):
        buf.add(np.array([[t], [100 + t]], dtype=np.uint8), np.array([0, 0]), np.array(r, dtype=np.float32), np.array(d, dtype=bool))
    obs, _, ret, nxt, disc = buf.sample(2000)
    for o, g, n, d in zip(obs[:, 0], ret, nxt[:, 0], disc):
        env, t = (0, int(o)) if o < 100 else (1, int(o) - 100)
        want, alive = 0.0, True
        for k in range(3):
            if alive:
                want += 0.5 ** k * rewards[t + k][env]
                alive = not terms[t + k][env]
        assert g == pytest.approx(want), (env, t)
        assert d == pytest.approx(0.125 if alive else 0.0)
        assert int(n) == (t + 3 if env == 0 else 103 + t)


def test_shaped_reward_uses_sqrt_points_and_life_penalty():
    cfg = ArcadeConfig()
    r = shaped_reward(np.array([10.0, 1600.0, 0.0]), np.array([False, False, True]), cfg)
    np.testing.assert_allclose(r, [np.sqrt(10) / 10, 4.0, -2.0], rtol=1e-6)


@needs_node
def test_bridge_steps_restarts_and_repeats_exactly():
    with ArcadeVecEnv(3, seed_base=11) as env:
        assert env.obs.shape == (3, env.obs_size) and env.obs_size == 1768
        rng = np.random.default_rng(1)
        over = 0
        for _ in range(3000):
            r = env.step(rng.integers(0, 4, 3))
            assert r.obs.dtype == np.uint8 and (r.points >= 0).all()
            over += int(r.game_over.sum())
        assert over > 0, "random play should finish games (and restart them)"

    def trace(sticky):
        with ArcadeVecEnv(2, seed_base=5, sticky=sticky, auto_reset=False) as env:
            out = []
            for k in range(400):
                r = env.step(np.array([k % 4, (k // 3) % 4]))
                out.append((r.score.tolist(), r.level.tolist(), r.obs.sum()))
            return out

    assert trace(0.25) == trace(0.25)
    assert trace(0.25) != trace(0.0)


@needs_node
def test_exported_brain_runs_identically_in_javascript(tmp_path):
    torch.manual_seed(3)
    net = ArcadeQNet(1768, 4, [32, 16])
    ck = tmp_path / "net.pt"
    torch.save({"online": net.state_dict(), "config": {"hidden": [32, 16]}, "obs_size": 1768, "steps": 0}, ck)
    meta = export_brain(ck, tmp_path / "brain")
    assert meta["sizes"] == [1768, 32, 16, 4]
    with ArcadeVecEnv(4, seed_base=21) as env:
        obs = [env.obs.copy()]
        for _ in range(30):
            obs.append(env.step(np.array([2, 3, 0, 1])).obs.copy())
    obs = np.concatenate(obs)
    js = subprocess.run(["node", str(REPO / "arcade" / "eval_brain.js"), str(tmp_path / "brain"), "--q"], input=obs.tobytes(),
                        capture_output=True, check=True)
    q_js = np.array(json.loads(js.stdout))
    with torch.no_grad():
        q_py = net(torch.from_numpy(obs.astype(np.float32)) / 255.0).numpy()
    np.testing.assert_allclose(q_js, q_py, atol=1e-5)
    assert (q_js.argmax(1) == q_py.argmax(1)).all()
    # and a whole exam: Python's run_exam and the pure-JavaScript player agree move for move
    py = run_exam(net, games=3, sticky=0.25, seed_base=1000, max_decisions=600)
    js = subprocess.run(["node", str(REPO / "arcade" / "eval_brain.js"), str(tmp_path / "brain"), "--games", "3", "--max", "600"],
                        capture_output=True, check=True)
    jr = json.loads(js.stdout)
    assert jr["scores"] == py["scores"] and jr["levels"] == py["levels"]
    # the export's final check calls the same player through js_exam
    je = js_exam(tmp_path / "brain", games=3, seed_base=1000, max_decisions=600)
    assert je["scores"] == py["scores"] and je["median_score"] == py["median_score"]


def test_export_record_and_checkpoint_hours(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "exam.csv").write_text(
        "steps,minutes,mean_score,median_score,max_score,mean_level,max_level,levels_cleared,ghosts_per_game,unfinished\n"
        "250000,2.5,1000.0,900.0,2000,1.0,1,0,1.0,0\n"
        "500000,6.0,3000.0,2800.0,5000,1.5,2,5,3.0,0\n")
    log = ["steps,updates,minutes,decisions_per_s,eps_mean,loss,q_mean,games,practice_score,practice_level",
           "20000,0,0.2,1600,1.0,,,0,,"]  # before any game finished: no practice score yet
    log += [f"{s},{s // 16},{s / 100000:.2f},1600,0.07,0.1,1.0,{s // 1000},{s / 200:.1f},1.2" for s in range(50000, 550000, 50000)]
    (run / "log.csv").write_text("\n".join(log) + "\n")
    assert checkpoint_hours(run, 500000) == 0.1  # 6 minutes, when that checkpoint was saved
    assert checkpoint_hours(run, 123) is None
    rec = export_record(run, tmp_path / "curve.json", final={"games": 2, "mean_score": 1.0}, random_score=700.4)
    assert rec["exams"] == [[250000, 1000.0, 1.0], [500000, 3000.0, 1.5]]
    assert rec["random"] == 700
    assert rec["practice"][0] == [50000, 250] and len(rec["practice"]) == 10  # the row without a score is skipped
    assert rec["summary"] == {"hours": 0.08, "decisions": 500000, "final": {"games": 2, "mean_score": 1.0}}
    assert json.loads((tmp_path / "curve.json").read_text()) == rec


def test_bridge_path_exists():
    assert BRIDGE.exists()
