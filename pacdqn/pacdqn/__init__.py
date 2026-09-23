"""pacdqn: a Double DQN agent for a Pac-Man-style maze game."""

from .agent import AgentConfig, DQNAgent
from .env import ACTION_NAMES, N_ACTIONS, NUM_CHANNELS, EnvConfig, MazeEnv, Rewards, make_gym_env
from .mazes import LAYOUTS, Layout, get_layout, parse_layout
from .model import QNetwork
from .replay import Batch, ReplayBuffer

__all__ = [
    "ACTION_NAMES", "N_ACTIONS", "NUM_CHANNELS", "LAYOUTS", "AgentConfig", "Batch", "DQNAgent",
    "EnvConfig", "Layout", "MazeEnv", "QNetwork", "ReplayBuffer", "Rewards", "get_layout",
    "make_gym_env", "parse_layout",
]
__version__ = "0.1.0"
