"""Double DQN agent with a dueling network, Huber loss and a hard target update.

The agent is deliberately small and readable; the classic DQN pieces are all
here and labelled: replay buffer, ε-greedy exploration, a frozen target network,
and the Double-DQN target (online net picks the action, target net scores it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .model import QNetwork
from .replay import Batch, ReplayBuffer


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 32
    buffer_size: int = 50_000
    learning_starts: int = 2_000
    train_every: int = 4  # gradient update every N env steps (DQN default)
    target_update: int = 1_000  # env steps between hard target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    reward_scale: float = 0.01  # env points -> training reward
    grad_clip: float = 10.0
    hidden: int = 256


class DQNAgent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        config: AgentConfig | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        self.cfg = config or AgentConfig()
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.online = QNetwork(self.obs_shape, self.n_actions, self.cfg.hidden).to(self.device)
        self.target = QNetwork(self.obs_shape, self.n_actions, self.cfg.hidden).to(self.device)
        self.sync_target()
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr, eps=1.5e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.obs_shape, seed=seed)
        self.steps = 0  # environment steps observed
        self.updates = 0  # gradient updates performed

    # ------------------------------------------------------------ acting
    def epsilon(self, step: int | None = None) -> float:
        s = self.steps if step is None else step
        frac = min(1.0, s / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, epsilon: float | None = None) -> int:
        eps = self.epsilon() if epsilon is None else epsilon
        if self._rng.random() < eps:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    # ----------------------------------------------------------- learning
    def observe(self, obs, action, reward, next_obs, done) -> float | None:
        """Store a transition and (maybe) train. Returns the loss if a step was taken."""
        self.buffer.add(obs, action, reward * self.cfg.reward_scale, next_obs, done)
        self.steps += 1
        loss = None
        if len(self.buffer) >= self.cfg.learning_starts and self.steps % self.cfg.train_every == 0:
            loss = self.train_step(self.buffer.sample(self.cfg.batch_size))
        if self.steps % self.cfg.target_update == 0:
            self.sync_target()
        return loss

    def train_step(self, batch: Batch) -> float:
        d = self.device
        obs = torch.as_tensor(batch.obs, device=d)
        actions = torch.as_tensor(batch.actions, device=d)
        rewards = torch.as_tensor(batch.rewards, device=d)
        next_obs = torch.as_tensor(batch.next_obs, device=d)
        dones = torch.as_tensor(batch.dones, device=d)

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network chooses, target network evaluates.
            next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------- persistence
    def save(self, path) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "config": asdict(self.cfg),
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str | None = None, seed: int | None = None) -> "DQNAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = AgentConfig(**ckpt["config"])
        agent = cls(tuple(ckpt["obs_shape"]), ckpt["n_actions"], cfg, device=device, seed=seed)
        agent.online.load_state_dict(ckpt["online"])
        agent.sync_target()
        if "optimizer" in ckpt:  # weights-only checkpoints (like the shipped demo best.pt) skip this
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.steps = ckpt["steps"]
        agent.updates = ckpt["updates"]
        return agent
