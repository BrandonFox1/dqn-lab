"""Pac-Man-style maze environment.

Gymnasium-compatible API (``reset(seed) -> (obs, info)``,
``step(a) -> (obs, reward, terminated, truncated, info)``) with no dependency on
gymnasium itself. Fully deterministic given a seed.

Observation: ``float32`` tensor of shape ``(NUM_CHANNELS, H, W)``:

=====  ===============================================
index  channel
=====  ===============================================
0      walls
1      pellets
2      power pellets
3      player
4      ghosts (dangerous)
5      ghosts (frightened / edible)
6      ghosts' previous positions (encodes heading)
7      power timer fraction (uniform fill, 0..1)
=====  ===============================================

Actions: 0 = up, 1 = down, 2 = left, 3 = right. Moving into a wall leaves the
player in place (the step still counts and ghosts still move).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mazes import Layout, get_layout

NUM_CHANNELS = 8
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("up", "down", "left", "right")
N_ACTIONS = len(ACTIONS)
_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


@dataclass
class Rewards:
    """Reward table, in game points."""

    pellet: float = 10.0
    power: float = 50.0
    ghost: float = 200.0
    clear: float = 500.0
    death: float = -500.0
    step: float = -1.0


@dataclass
class EnvConfig:
    maze: str = "small"
    n_ghosts: int = 1
    lives: int = 1
    max_steps: int = 500
    power_duration: int = 20
    ghost_respawn: int = 10
    ghost_speed: float = 1.0
    frightened_speed: float = 0.5
    # Probability that ghost i chases (vs. moves randomly). Cycled if fewer than n_ghosts.
    ghost_aggression: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3)
    rewards: Rewards = field(default_factory=Rewards)


@dataclass
class Ghost:
    home: tuple[int, int]
    aggression: float
    pos: tuple[int, int]
    prev: tuple[int, int]
    direction: int | None = None
    respawn_in: int = 0  # >0 means eaten and waiting off-board

    @property
    def active(self) -> bool:
        return self.respawn_in == 0


class MazeEnv:
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | None = None, **overrides):
        cfg = config or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown EnvConfig field {k!r}")
            setattr(cfg, k, v)
        if cfg.n_ghosts < 0:
            raise ValueError("n_ghosts must be >= 0")
        self.cfg = cfg
        self.layout: Layout = get_layout(cfg.maze)
        self.walls = self.layout.walls
        self.height, self.width = self.walls.shape
        self.observation_shape = (NUM_CHANNELS, self.height, self.width)
        self.n_actions = N_ACTIONS
        self._rng = np.random.default_rng()
        self._dist_cache: dict[tuple[int, int], np.ndarray] = {}
        self._reset_state()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action {action}")
        R = self.cfg.rewards
        reward = R.step
        self.steps += 1
        self.ghosts_eaten_this_step = 0

        # --- player moves
        dy, dx = ACTIONS[action]
        ny, nx = self.player[0] + dy, self.player[1] + dx
        if not self.walls[ny, nx]:
            self.player = (ny, nx)

        # --- eat
        if self.pellets[self.player]:
            self.pellets[self.player] = False
            reward += R.pellet
            self.score += R.pellet
        elif self.powers[self.player]:
            self.powers[self.player] = False
            reward += R.power
            self.score += R.power
            self.power_timer = self.cfg.power_duration
            for g in self.ghosts:
                if g.active:
                    g.direction = _OPPOSITE.get(g.direction, g.direction)

        # --- collisions after player move
        reward += self._resolve_collisions()
        if not self.done:
            # --- ghosts move
            self._move_ghosts()
            reward += self._resolve_collisions()

        if self.power_timer > 0:
            self.power_timer -= 1

        terminated = False
        truncated = False
        if self.done:  # killed with no lives left
            terminated = True
        elif self.pellets_left == 0:
            reward += R.clear
            self.score += R.clear
            self.won = True
            self.done = True
            terminated = True
        elif self.steps >= self.cfg.max_steps:
            truncated = True
            self.done = True

        return self._obs(), float(reward), terminated, truncated, self._info()

    def render(self) -> str:
        grid = np.where(self.walls, "#", " ").astype(object)
        grid[self.pellets] = "."
        grid[self.powers] = "o"
        for g in self.ghosts:
            if g.active:
                grid[g.pos] = "g" if self.power_timer > 0 else "G"
        grid[self.player] = "P"
        rows = ["".join(r) for r in grid]
        status = f"score={self.score:.0f} pellets_left={self.pellets_left} steps={self.steps} lives={self.lives}"
        return "\n".join(rows + [status])

    def legal_actions(self) -> list[int]:
        return [a for a, (dy, dx) in enumerate(ACTIONS) if not self.walls[self.player[0] + dy, self.player[1] + dx]]

    # ------------------------------------------------------------- internals
    @property
    def pellets_left(self) -> int:
        return int(self.pellets.sum() + self.powers.sum())

    def _reset_state(self) -> None:
        self.pellets = self.layout.pellets.copy()
        self.powers = self.layout.powers.copy()
        self.lives = self.cfg.lives
        self.score = 0.0
        self.steps = 0
        self.power_timer = 0
        self.done = False
        self.won = False
        self.ghosts_eaten_this_step = 0
        self._place_actors()

    def _place_actors(self) -> None:
        self.player = self.layout.player_start
        homes = self.layout.ghost_starts
        aggr = self.cfg.ghost_aggression
        self.ghosts = [
            Ghost(home=homes[i % len(homes)], aggression=aggr[i % len(aggr)], pos=homes[i % len(homes)], prev=homes[i % len(homes)])
            for i in range(self.cfg.n_ghosts)
        ]

    def _resolve_collisions(self) -> float:
        reward = 0.0
        R = self.cfg.rewards
        for g in self.ghosts:
            if not g.active or g.pos != self.player:
                continue
            if self.power_timer > 0:
                reward += R.ghost
                self.score += R.ghost
                self.ghosts_eaten_this_step += 1
                g.respawn_in = self.cfg.ghost_respawn
                g.pos = g.prev = g.home
                g.direction = None
            else:
                reward += R.death
                self.score += R.death
                self.lives -= 1
                if self.lives <= 0:
                    self.done = True
                else:
                    self._place_actors()
                    self.power_timer = 0
                break
        return reward

    def _move_ghosts(self) -> None:
        frightened = self.power_timer > 0
        for g in self.ghosts:
            if not g.active:
                g.respawn_in -= 1
                if g.respawn_in == 0:
                    g.pos = g.prev = g.home
                continue
            speed = self.cfg.frightened_speed if frightened else self.cfg.ghost_speed
            g.prev = g.pos
            if self._rng.random() >= speed:
                continue
            legal = self._ghost_legal_moves(g)
            if not legal:
                continue
            if self._rng.random() < g.aggression:
                dist = self._distances(self.player)
                scored = [(dist[g.pos[0] + ACTIONS[a][0], g.pos[1] + ACTIONS[a][1]], a) for a in legal]
                best = min(scored)[0] if not frightened else max(scored)[0]
                choices = [a for d, a in scored if d == best]
                a = int(self._rng.choice(choices))
            else:
                a = int(self._rng.choice(legal))
            g.direction = a
            g.pos = (g.pos[0] + ACTIONS[a][0], g.pos[1] + ACTIONS[a][1])

    def _ghost_legal_moves(self, g: Ghost) -> list[int]:
        legal = [a for a, (dy, dx) in enumerate(ACTIONS) if not self.walls[g.pos[0] + dy, g.pos[1] + dx]]
        if g.direction is not None and len(legal) > 1:
            rev = _OPPOSITE[g.direction]
            no_reverse = [a for a in legal if a != rev]
            if no_reverse:
                legal = no_reverse
        return legal

    def _distances(self, target: tuple[int, int]) -> np.ndarray:
        """BFS distance from every cell to ``target`` (cached per target)."""
        cached = self._dist_cache.get(target)
        if cached is not None:
            return cached
        dist = np.full(self.walls.shape, np.iinfo(np.int32).max // 2, dtype=np.int32)
        dist[target] = 0
        q: deque[tuple[int, int]] = deque([target])
        while q:
            y, x = q.popleft()
            for dy, dx in ACTIONS:
                ny, nx = y + dy, x + dx
                if not self.walls[ny, nx] and dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))
        self._dist_cache[target] = dist
        return dist

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        obs[0] = self.walls
        obs[1] = self.pellets
        obs[2] = self.powers
        obs[3][self.player] = 1.0
        frightened = self.power_timer > 0
        for g in self.ghosts:
            if g.active:
                obs[5 if frightened else 4][g.pos] = 1.0
                obs[6][g.prev] = 1.0
        if self.cfg.power_duration > 0:
            obs[7] = self.power_timer / self.cfg.power_duration
        return obs

    def _info(self) -> dict:
        return {
            "score": self.score,
            "pellets_left": self.pellets_left,
            "steps": self.steps,
            "lives": self.lives,
            "won": self.won,
            "power_timer": self.power_timer,
            "ghosts_eaten": self.ghosts_eaten_this_step,
        }


def make_gym_env(config: EnvConfig | None = None, **overrides):
    """Wrap :class:`MazeEnv` as a real ``gymnasium.Env`` if gymnasium is installed."""
    import gymnasium as gym  # optional dependency

    class GymMazeEnv(gym.Env):
        metadata = MazeEnv.metadata

        def __init__(self):
            super().__init__()
            self.env = MazeEnv(config, **overrides)
            self.observation_space = gym.spaces.Box(0.0, 1.0, self.env.observation_shape, np.float32)
            self.action_space = gym.spaces.Discrete(N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.env.reset(seed=seed)

        def step(self, action):
            return self.env.step(action)

        def render(self):
            return self.env.render()

    return GymMazeEnv()
