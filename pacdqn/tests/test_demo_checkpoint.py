"""The shipped demo checkpoint is weights-only (optimizer state stripped to keep the repo small)."""
from pathlib import Path

import pytest
import torch

from pacdqn.play import load_run

DEMO = Path(__file__).resolve().parents[1] / "runs" / "small"


@pytest.mark.skipif(not (DEMO / "best.pt").exists(), reason="demo run not included")
def test_demo_checkpoint_is_weights_only_and_loads():
    ckpt = torch.load(DEMO / "best.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in ckpt and "online" in ckpt
    agent, env_cfg = load_run(DEMO)
    assert agent is not None and env_cfg is not None
