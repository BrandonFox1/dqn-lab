import numpy as np
import pytest
import torch

from pacdqn.agent import AgentConfig, DQNAgent
from pacdqn.env import EnvConfig, MazeEnv
from pacdqn.model import QNetwork
from pacdqn.replay import ReplayBuffer

SHAPE = (8, 11, 13)


# ---------------------------------------------------------------- replay
def test_replay_roundtrip_and_wraparound():
    buf = ReplayBuffer(capacity=5, obs_shape=SHAPE, seed=0)
    for i in range(7):
        o = np.full(SHAPE, i / 10, dtype=np.float32)
        buf.add(o, i % 4, float(i), o + 0.05, done=(i == 6))
    assert len(buf) == 5
    assert buf.pos == 2
    b = buf.sample(32)
    assert b.obs.shape == (32, *SHAPE) and b.obs.dtype == np.float32
    assert b.next_obs.shape == (32, *SHAPE)
    assert b.actions.dtype == np.int64 and b.actions.shape == (32,)
    assert b.rewards.shape == (32,) and b.dones.shape == (32,)
    # The two oldest transitions (rewards 0 and 1) were overwritten.
    assert set(np.unique(b.rewards)).issubset({2.0, 3.0, 4.0, 5.0, 6.0})
    # Quantisation error is bounded by half a step.
    assert np.abs(b.obs - np.round(b.obs * 255) / 255).max() < 1e-6


def test_replay_errors():
    with pytest.raises(ValueError):
        ReplayBuffer(0, SHAPE)
    with pytest.raises(ValueError):
        ReplayBuffer(3, SHAPE).sample(1)


# ----------------------------------------------------------------- model
@pytest.mark.parametrize("shape", [SHAPE, (8, 18, 19), (4, 84, 84)])
def test_qnetwork_output_shape(shape):
    net = QNetwork(shape, 4, hidden=32)
    x = torch.zeros(3, *shape)
    y = net(x)
    assert y.shape == (3, 4)


def test_dueling_advantage_is_zero_mean():
    net = QNetwork(SHAPE, 4)
    x = torch.rand(2, *SHAPE)
    f = net.features(x)
    a = net.advantage(f)
    assert torch.allclose((net(x) - net.value(f)).mean(dim=1), torch.zeros(2), atol=1e-5)
    assert a.shape == (2, 4)


# ----------------------------------------------------------------- agent
def make_agent(**kw):
    cfg = AgentConfig(buffer_size=200, learning_starts=10, batch_size=8, target_update=20, eps_decay_steps=100, hidden=32, train_every=1, **kw)
    return DQNAgent(SHAPE, 4, cfg, device="cpu", seed=0)


def test_epsilon_schedule():
    agent = make_agent()
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(50) == pytest.approx(0.525)
    assert agent.epsilon(100) == pytest.approx(0.05)
    assert agent.epsilon(10_000) == pytest.approx(0.05)


def test_act_returns_valid_actions_and_greedy_is_argmax():
    agent = make_agent()
    obs = np.random.default_rng(0).random(SHAPE, dtype=np.float32)
    for _ in range(20):
        assert 0 <= agent.act(obs) < 4
    assert agent.act(obs, epsilon=0.0) == int(np.argmax(agent.q_values(obs)))


def test_observe_trains_after_learning_starts_and_syncs_target():
    agent = make_agent()
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1))
    obs, _ = env.reset(seed=0)
    losses = []
    for step in range(1, 41):
        a = agent.act(obs)
        nobs, r, term, trunc, _ = env.step(a)
        loss = agent.observe(obs, a, r, nobs, term)
        if step < 10:
            assert loss is None
        else:
            assert loss is not None and np.isfinite(loss)
            losses.append(loss)
        obs = nobs
        if term or trunc:
            obs, _ = env.reset()
    assert agent.steps == 40 and agent.updates == 31
    # target was synced at step 40 -> identical parameters
    for p, q in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(p, q)


def test_train_step_reduces_loss_on_fixed_batch():
    agent = make_agent(lr=1e-3)
    rng = np.random.default_rng(0)
    for i in range(50):
        o = rng.random(SHAPE, dtype=np.float32)
        agent.buffer.add(o, int(rng.integers(4)), float(rng.normal()), rng.random(SHAPE, dtype=np.float32), bool(i % 5 == 0))
    batch = agent.buffer.sample(16)
    first = agent.train_step(batch)
    for _ in range(30):
        last = agent.train_step(batch)
    assert last < first


def test_double_dqn_target_uses_online_argmax():
    """The target must be r + γ·Q_target(s', argmax_a Q_online(s', a)), not max_a Q_target."""
    agent = make_agent(gamma=0.9)
    rng = np.random.default_rng(1)
    # make online/target disagree
    for p in agent.target.parameters():
        p.data.add_(torch.randn_like(p) * 0.5)
    obs = rng.random((4, *SHAPE), dtype=np.float32)
    nobs = rng.random((4, *SHAPE), dtype=np.float32)
    rewards = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    dones = np.array([0, 0, 1, 0], dtype=np.float32)
    with torch.no_grad():
        n_t = torch.as_tensor(nobs)
        a_star = agent.online(n_t).argmax(1, keepdim=True)
        expected = torch.as_tensor(rewards) + 0.9 * (1 - torch.as_tensor(dones)) * agent.target(n_t).gather(1, a_star).squeeze(1)
        naive = torch.as_tensor(rewards) + 0.9 * (1 - torch.as_tensor(dones)) * agent.target(n_t).max(1).values
    # Recompute via the agent's own code path by monkeypatching the loss to capture the target.
    captured = {}

    def capture(q, target):
        captured["target"] = target.detach().clone()
        return ((q - target) ** 2).mean()

    agent.loss_fn = capture
    from pacdqn.replay import Batch

    agent.train_step(Batch(obs, np.zeros(4, dtype=np.int64), rewards, nobs, dones))
    assert torch.allclose(captured["target"], expected, atol=1e-5)
    assert not torch.allclose(captured["target"], naive, atol=1e-5)
    assert captured["target"][2] == pytest.approx(0.3)  # terminal: no bootstrap


def test_save_load_roundtrip(tmp_path):
    agent = make_agent()
    obs = np.random.default_rng(0).random(SHAPE, dtype=np.float32)
    agent.steps, agent.updates = 123, 45
    path = tmp_path / "ckpt.pt"
    agent.save(path)
    loaded = DQNAgent.load(path, device="cpu")
    assert loaded.steps == 123 and loaded.updates == 45
    assert loaded.cfg == agent.cfg
    assert np.allclose(loaded.q_values(obs), agent.q_values(obs))
