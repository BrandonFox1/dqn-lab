"""Many arcade-rules Pac-Man games at once, run by the JavaScript engine in ``arcade/arcade.js``.

The game logic lives in exactly one place: the same file the Workshop page runs. This module starts
``node arcade/bridge.js`` and exchanges small binary messages with it, so training and the page can never
disagree about the rules or about what the network sees.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "arcade" / "bridge.js"
ACTION_NAMES = ("up", "down", "left", "right")


@dataclass
class StepResult:
    obs: np.ndarray  # uint8 (N, D): observation for the next decision (after any reset)
    points: np.ndarray  # float32 (N,): arcade points scored during this decision
    life_lost: np.ndarray  # bool (N,)
    game_over: np.ndarray  # bool (N,): the game ended (and restarted, if auto-reset is on)
    cleared: np.ndarray  # bool (N,): a level was cleared
    level: np.ndarray  # uint16 (N,)
    score: np.ndarray  # uint32 (N,): score at the end of this decision (the final score if game_over)
    frames: np.ndarray  # uint16 (N,)
    ghosts: np.ndarray  # uint8 (N,): ghosts eaten during this decision


class ArcadeVecEnv:
    """``count`` independent games. Seeds run from ``seed_base``; restarted games take the next unused seed."""

    def __init__(self, count: int, seed_base: int = 1, sticky: float = 0.0, auto_reset: bool = True, node: str | None = None):
        self.count = int(count)
        node = node or os.environ.get("NODE", "node")
        self.proc = subprocess.Popen([node, str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        hello = json.loads(self.proc.stdout.readline())
        self.obs_size, self.n_actions = int(hello["obsSize"]), int(hello["nActions"])
        self._layout()
        self.proc.stdin.write(b"I" + struct.pack("<IIfB", self.count, int(seed_base), float(sticky), 1 if auto_reset else 0))
        self.obs = self._read()[0]

    def _layout(self) -> None:
        n, d = self.count, self.obs_size
        obs = n * d
        pts = obs + (4 - obs % 4) % 4
        flags = pts + 4 * n
        level = flags + n + n % 2
        score = level + 2 * n
        frames = score + 4 * n
        ghosts = frames + 2 * n
        self._off = dict(pts=pts, flags=flags, level=level, score=score, frames=frames, ghosts=ghosts)
        self._total = ghosts + n

    def _read_exact(self, size: int) -> bytes:
        chunks, got = [], 0
        while got < size:
            b = self.proc.stdout.read(size - got)
            if not b:
                raise RuntimeError("arcade bridge closed its output")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def _read(self):
        raw = self._read_exact(self._total)
        n, o = self.count, self._off
        obs = np.frombuffer(raw, dtype=np.uint8, count=n * self.obs_size).reshape(n, self.obs_size)
        flags = np.frombuffer(raw, dtype=np.uint8, count=n, offset=o["flags"])
        return (
            obs,
            np.frombuffer(raw, dtype="<f4", count=n, offset=o["pts"]),
            flags,
            np.frombuffer(raw, dtype="<u2", count=n, offset=o["level"]),
            np.frombuffer(raw, dtype="<u4", count=n, offset=o["score"]),
            np.frombuffer(raw, dtype="<u2", count=n, offset=o["frames"]),
            np.frombuffer(raw, dtype=np.uint8, count=n, offset=o["ghosts"]),
        )

    def step(self, actions) -> StepResult:
        a = np.asarray(actions, dtype=np.uint8)
        if a.shape != (self.count,):
            raise ValueError(f"expected {self.count} actions, got shape {a.shape}")
        self.proc.stdin.write(b"S" + a.tobytes())
        obs, pts, flags, level, score, frames, ghosts = self._read()
        self.obs = obs
        return StepResult(obs, pts, (flags & 1) > 0, (flags & 2) > 0, (flags & 4) > 0, level, score, frames, ghosts)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"X")
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
