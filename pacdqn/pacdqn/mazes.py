"""Maze layouts.

Legend
------
``#``  wall
``.``  pellet
``o``  power pellet
``P``  player start (exactly one)
``G``  ghost spawn (one or more; ghosts cycle through them)
`` ``  empty corridor (walkable, no pellet)

Layouts are original designs for a Pac-Man-style pellet/ghost game. Every
layout is validated on import: rectangular, fully enclosed by walls, exactly
one ``P``, at least one ``G``, and every pellet/ghost cell reachable from ``P``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

WALL, PELLET, POWER, PLAYER, GHOST, EMPTY = "#", ".", "o", "P", "G", " "
VALID_CHARS = {WALL, PELLET, POWER, PLAYER, GHOST, EMPTY}


@dataclass(frozen=True)
class Layout:
    name: str
    walls: np.ndarray  # bool (H, W)
    pellets: np.ndarray  # bool (H, W) regular pellets
    powers: np.ndarray  # bool (H, W) power pellets
    player_start: tuple[int, int]
    ghost_starts: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        return int(self.walls.shape[0])

    @property
    def width(self) -> int:
        return int(self.walls.shape[1])

    @property
    def n_pellets(self) -> int:
        return int(self.pellets.sum() + self.powers.sum())


def parse_layout(name: str, text: str) -> Layout:
    rows = [r for r in text.strip("\n").split("\n")]
    if not rows:
        raise ValueError(f"{name}: empty layout")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"{name}: all rows must have the same width")
    height = len(rows)

    walls = np.zeros((height, width), dtype=bool)
    pellets = np.zeros((height, width), dtype=bool)
    powers = np.zeros((height, width), dtype=bool)
    player: tuple[int, int] | None = None
    ghosts: list[tuple[int, int]] = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in VALID_CHARS:
                raise ValueError(f"{name}: invalid char {ch!r} at ({y},{x})")
            if ch == WALL:
                walls[y, x] = True
            elif ch == PELLET:
                pellets[y, x] = True
            elif ch == POWER:
                powers[y, x] = True
            elif ch == PLAYER:
                if player is not None:
                    raise ValueError(f"{name}: more than one player start")
                player = (y, x)
            elif ch == GHOST:
                ghosts.append((y, x))

    if player is None:
        raise ValueError(f"{name}: no player start 'P'")
    if not ghosts:
        raise ValueError(f"{name}: no ghost spawn 'G'")
    border = np.concatenate([walls[0], walls[-1], walls[:, 0], walls[:, -1]])
    if not border.all():
        raise ValueError(f"{name}: layout must be enclosed by walls")

    layout = Layout(name, walls, pellets, powers, player, tuple(ghosts))
    unreachable = _unreachable_targets(layout)
    if unreachable:
        raise ValueError(f"{name}: unreachable cells from player start: {unreachable[:5]}")
    return layout


def reachable_from(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """BFS over non-wall cells; returns a bool mask of reachable cells."""
    h, w = walls.shape
    seen = np.zeros_like(walls, dtype=bool)
    seen[start] = True
    q: deque[tuple[int, int]] = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not walls[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def _unreachable_targets(layout: Layout) -> list[tuple[int, int]]:
    seen = reachable_from(layout.walls, layout.player_start)
    targets = layout.pellets | layout.powers
    for g in layout.ghost_starts:
        targets[g] = True
    bad = np.argwhere(targets & ~seen)
    return [tuple(int(v) for v in p) for p in bad]


SMALL = """
#############
#o....#....o#
#.##.###.##.#
#...........#
#.##.#.#.##.#
#....#G#....#
#.##.###.##.#
#...........#
#.##.#.#.##.#
#o...#P#...o#
#############
"""

MEDIUM = """
###################
#o.......#.......o#
#.###.##.#.##.###.#
#.................#
#.###.#.###.#.###.#
#.....#.....#.....#
#####.#######.#####
#####.#.....#.#####
#####.#.GGG.#.#####
#####.#.###.#.#####
#.....#.....#.....#
#.###.###.###.###.#
#o..#.........#..o#
###.#.#.###.#.#.###
#.....#.....#.....#
#.#####.###.#####.#
#........P........#
###################
"""

LAYOUTS: dict[str, Layout] = {
    "small": parse_layout("small", SMALL),
    "medium": parse_layout("medium", MEDIUM),
}


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unknown maze {name!r}; choose from {sorted(LAYOUTS)}") from None
