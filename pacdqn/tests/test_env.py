import numpy as np
import pytest

from pacdqn.env import ACTIONS, N_ACTIONS, NUM_CHANNELS, EnvConfig, MazeEnv
from pacdqn.mazes import LAYOUTS, parse_layout, reachable_from


# ----------------------------------------------------------------- mazes
@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_layouts_are_valid_and_connected(name):
    lay = LAYOUTS[name]
    seen = reachable_from(lay.walls, lay.player_start)
    assert seen[lay.pellets | lay.powers].all()
    for g in lay.ghost_starts:
        assert seen[g]
    assert not lay.walls[lay.player_start]
    assert lay.n_pellets > 0


def test_parse_rejects_bad_layouts():
    with pytest.raises(ValueError, match="enclosed"):
        parse_layout("x", "#####\n#P.G.\n#####")
    with pytest.raises(ValueError, match="no player"):
        parse_layout("x", "#####\n#..G#\n#####")
    with pytest.raises(ValueError, match="no ghost"):
        parse_layout("x", "#####\n#P..#\n#####")
    with pytest.raises(ValueError, match="unreachable"):
        parse_layout("x", "#######\n#P.#.G#\n#######")
    with pytest.raises(ValueError, match="same width"):
        parse_layout("x", "#####\n#P.G#\n####")
    with pytest.raises(ValueError, match="invalid char"):
        parse_layout("x", "#####\n#PxG#\n#####")


# ------------------------------------------------------------------- env
def test_reset_obs_shape_and_channels():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1))
    obs, info = env.reset(seed=0)
    assert obs.shape == (NUM_CHANNELS, env.height, env.width)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert obs[3].sum() == 1  # exactly one player
    assert obs[4].sum() == 1  # one dangerous ghost
    assert obs[5].sum() == 0  # no frightened ghosts
    assert obs[7].max() == 0  # no power timer
    assert info["pellets_left"] == env.layout.n_pellets
    assert np.array_equal(obs[0].astype(bool), env.walls)


def test_deterministic_given_seed():
    def rollout(seed):
        env = MazeEnv(EnvConfig(maze="medium", n_ghosts=3))
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trace = []
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(rng.integers(N_ACTIONS))
            trace.append((r, env.player, tuple(g.pos for g in env.ghosts)))
            done = term or trunc
        return trace

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_cannot_walk_through_walls():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=0))
    env.reset(seed=0)
    # In the small maze the player starts at (9,6), walled left/right/below.
    for a in (1, 2, 3):  # down, left, right
        pos = env.player
        env.step(a)
        assert env.player == pos
    env.step(0)  # up is open
    assert env.player == (8, 6)


def test_pellets_reward_and_step_penalty():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=0))
    env.reset(seed=0)
    R = env.cfg.rewards
    left_before = env.pellets_left
    _, r, *_ = env.step(0)  # up onto a pellet
    assert r == pytest.approx(R.pellet + R.step)
    assert env.pellets_left == left_before - 1
    _, r, *_ = env.step(1)  # back down to the (empty) start cell
    assert r == pytest.approx(R.step)


def test_ghost_collision_ends_episode_with_death_penalty():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1, ghost_speed=0.0))
    env.reset(seed=0)
    # Teleport the ghost directly above the player, then step into it.
    env.ghosts[0].pos = env.ghosts[0].prev = (8, 6)
    obs, r, term, trunc, info = env.step(0)
    assert term and not trunc
    assert r == pytest.approx(env.cfg.rewards.death + env.cfg.rewards.pellet + env.cfg.rewards.step)
    assert info["lives"] == 0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_multiple_lives_respawn():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1, lives=2, ghost_speed=0.0))
    env.reset(seed=0)
    env.ghosts[0].pos = env.ghosts[0].prev = (8, 6)
    _, r, term, trunc, info = env.step(0)
    assert not term
    assert info["lives"] == 1
    assert env.player == env.layout.player_start
    assert env.ghosts[0].pos == env.ghosts[0].home


def test_power_pellet_makes_ghosts_edible():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1, ghost_speed=0.0, frightened_speed=0.0, power_duration=5, ghost_respawn=3))
    env.reset(seed=0)
    # Move player next to the power pellet at (9,1): go to (9,2) via teleport.
    env.player = (9, 2)
    obs, r, *_ = env.step(2)  # left onto 'o'
    assert env.power_timer > 0
    assert obs[7].max() > 0
    assert obs[5].sum() == 1 and obs[4].sum() == 0  # ghost now frightened in obs
    # Put the frightened ghost in our path and eat it.
    env.ghosts[0].pos = env.ghosts[0].prev = (9, 2)
    obs, r, term, trunc, info = env.step(3)
    assert not term
    assert info["ghosts_eaten"] == 1
    assert r == pytest.approx(env.cfg.rewards.ghost + env.cfg.rewards.pellet + env.cfg.rewards.step)
    assert not env.ghosts[0].active
    assert obs[4].sum() == 0 and obs[5].sum() == 0  # off-board while respawning
    for _ in range(3):
        env.step(2)
    assert env.ghosts[0].active and env.ghosts[0].pos == env.ghosts[0].home


def test_clearing_all_pellets_wins():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=0))
    env.reset(seed=0)
    env.pellets[:] = False
    env.powers[:] = False
    env.pellets[8, 6] = True  # one pellet directly above the player
    obs, r, term, trunc, info = env.step(0)
    assert term and info["won"]
    assert r == pytest.approx(env.cfg.rewards.pellet + env.cfg.rewards.clear + env.cfg.rewards.step)


def test_time_limit_truncates():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=0, max_steps=7))
    env.reset(seed=0)
    for i in range(7):
        _, _, term, trunc, _ = env.step(1)  # bump into wall repeatedly
    assert trunc and not term


def test_ghosts_chase_when_aggressive():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1, ghost_aggression=(1.0,)))
    env.reset(seed=0)
    d0 = env._distances(env.player)[env.ghosts[0].pos]
    env.step(1)  # player bumps wall, stays put
    d1 = env._distances(env.player)[env.ghosts[0].pos]
    assert d1 == d0 - 1


def test_frightened_ghosts_flee():
    env = MazeEnv(EnvConfig(maze="small", n_ghosts=1, ghost_aggression=(1.0,), frightened_speed=1.0, power_duration=50))
    env.reset(seed=0)
    env.power_timer = 50
    env.ghosts[0].pos = env.ghosts[0].prev = (7, 6)  # open corridor
    env.ghosts[0].direction = None
    d0 = env._distances(env.player)[env.ghosts[0].pos]
    env.step(1)
    d1 = env._distances(env.player)[env.ghosts[0].pos]
    assert d1 > d0


def test_legal_actions_and_render():
    env = MazeEnv(EnvConfig(maze="small"))
    env.reset(seed=0)
    assert env.legal_actions() == [0]
    text = env.render()
    assert "P" in text and "G" in text and text.count("\n") == env.height


def test_invalid_config_and_action():
    with pytest.raises(TypeError):
        MazeEnv(nonsense=1)
    with pytest.raises(KeyError):
        MazeEnv(EnvConfig(maze="nope"))
    env = MazeEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(9)


def test_gym_wrapper_if_available():
    gym = pytest.importorskip("gymnasium")
    from pacdqn.env import make_gym_env

    env = make_gym_env(EnvConfig(maze="small"))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
