"""Dueling Q-network for grid observations of shape (C, H, W).

Works on any spatial size (the flatten size is computed from ``obs_shape``),
so the same network serves the small maze, the medium maze, or a downsampled
Atari frame stack.
"""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden: int = 256, channels: int = 32, squeeze: int = 16):
        super().__init__()
        c, h, w = obs_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            # 1x1 "squeeze" keeps the flattened vector small so the heads stay cheap.
            nn.Conv2d(channels * 2, squeeze, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.value = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.n_actions = n_actions
        self.obs_shape = tuple(obs_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)
