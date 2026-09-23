"""Uniform experience replay.

Observations are quantised to ``uint8`` (values are in ``[0, 1]``) so a
100k-transition buffer for the medium maze fits in ~550 MB instead of ~2.2 GB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray  # float32 (B, C, H, W)
    actions: np.ndarray  # int64 (B,)
    rewards: np.ndarray  # float32 (B,)
    next_obs: np.ndarray  # float32 (B, C, H, W)
    dones: np.ndarray  # float32 (B,)  1.0 if terminal (bootstrapping stops)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _encode(obs: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def _decode(obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.pos
        self.obs[i] = self._encode(obs)
        self.next_obs[i] = self._encode(next_obs)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._rng.integers(0, self.size, size=batch_size)
        return Batch(
            obs=self._decode(self.obs[idx]),
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self._decode(self.next_obs[idx]),
            dones=self.dones[idx],
        )
