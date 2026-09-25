# Arcade rules spec

The rules `arcade.js` implements, one numbered rule per line, each with its source. `test_arcade.js` cites these numbers.

**Sources.** Brandon chose "documents only": no ROM files and nothing extracted from ROMs (no tile, sprite, palette, sound or program bytes).

- **[D]** *The Pac-Man Dossier* by Jamey Pittman, v1.0.26 (2011). The author verified it against the ROM disassembly. A PDF copy ships in [floooh/pacman.c](https://github.com/floooh/pacman.c) as `pacman-dossier.pdf`, and page numbers below refer to that PDF. Rules marked *fig.* were measured from the Dossier's figures.
- **[P]** [floooh/pacman.c](https://github.com/floooh/pacman.c) (MIT license), a C reimplementation written from the Dossier. Used only as a cross-check for coordinates. Its embedded ROM graphics were not used.
- **[C]** Our choice, where neither source says. Kept to the smallest reasonable option and listed at the end.

All artwork on the Workshop page is original. The maze *layout* (which tiles are walls) matches the arcade, but it is drawn in the Workshop's own style.

## 1. Screen, tiles, time
- **1.1** The screen is 224×288 pixels, or 28×36 tiles of 8×8. [D p.22]
- **1.2** An actor occupies exactly one tile: the tile that contains its center point. [D p.22]
- **1.3** Every tile's center pixel is at offset (3, 4) inside the tile: pixel (8·tx+3, 8·ty+4). [D p.16 fig.: approaching a turn from the left gives 3 pre-turn pixels and 4 post-turn; from the right 4 and 3; from above 4 and 3; from below 3 and 4.]
- **1.4** Game logic runs once per frame. Timers given in seconds are counted as 60 frames per second (the Dossier writes a frame as 1/60 s). The real refresh rate is 60.606 Hz, which only matters for wall-clock time. [D p.14, p.41]

## 2. Maze
- **2.1** The layout is `MAZE` in `arcade.js`: screen rows 3–33, with 240 dots (10 points) and 4 energizers (50 points) = 244. [D p.10; layout cross-checked with P]
- **2.2** Energizers sit at tiles (1,6), (26,6), (1,26), (26,26). [P]
- **2.3** Row 17 is the side tunnel. Leaving one edge wraps to the other. [D p.20, glossary]
- **2.4** Ghosts are slowed in the tunnel zone: row 17 with x ≤ 4 or x ≥ 23. Pac-Man is never slowed. [D p.20 fig. (measured 5 tiles per side); P uses 6]
- **2.5** Red zones are rows 14 and 26 for x 11–16. In scatter and chase mode, a ghost may not choose "up" there. Frightened ghosts ignore red zones. [D p.20, fig. matches P] Eyes also ignore them. [P]
- **2.6** The ghost house door is tiles (13,15) and (14,15). Only ghosts leaving or entering the house pass through it. For Pac-Man, and for ghosts roaming the maze, it is a wall. [D p.19, P]

## 3. Start positions (pixels of the center point)
- **3.1** Pac-Man starts at (111, 212), facing left. [P, shifted by the 1.3 center convention]
- **3.2** Blinky starts outside the house at (111, 116), facing left. Pinky starts in the house at (111, 140), Inky at (95, 140), Clyde at (127, 140). [D p.19 order, P positions]

## 4. Speed
- **4.1** Speeds are percentages of the maximum. 100% = 1.25 px/frame, so 80% = 1 px/frame. [Derived from D p.15: the "dots" column, e.g. 80% → ~71%, only works out with a 1-frame stop per 8-pixel tile at 1 px/frame. Test 4.1 reproduces the whole column.]
- **4.2** Each frame an actor's percentage is added to an accumulator, and it moves one pixel per 80 units. This gives the exact average speed; the arcade's frame-by-frame bit patterns are ROM data we don't use. [C]
- **4.3** Eating a dot stops Pac-Man for 1 frame, an energizer for 3. His movement rhythm (4.2) waits during those frames rather than skipping ahead. [D p.15] We first skipped the stalled frame's movement instead. That gave 80.5% instead of the Dossier's ~83% at 95% speed, because stalls kept landing on 2-pixel frames. Waiting reproduces every "dots" value in Table A.1.
- **4.4** Speeds per level, Pac-Man normal/fright and ghost normal/fright/tunnel: L1 80/90, 75/50/40. L2–4 90/95, 85/55/45. L5–20 100/100, 95/60/50. L21+ 90/–, 95/–/50. [D p.15, Table A.1]
- **4.5** Pac-Man uses his fright speed while the fright timer runs. [D p.15]
- **4.6** Ghost speed has this priority: eyes, then tunnel, then frightened, then Elroy (Blinky), then normal. [D p.20 "always enforced"; C for eyes]
- **4.7** Eyes move at 1.5 px/frame (120%). Ghosts inside the house move at 0.5 px/frame (40%). [P estimates; D is silent]

## 5. Pac-Man movement
- **5.1** The joystick has one direction at a time, or none. [D p.41]
- **5.2** Pac-Man can reverse at any moment. [D p.16]
- **5.3** He turns into a new direction as soon as the tile next to his current tile in that direction is open. That can be anywhere in the tile, which is what makes pre-turns and post-turns possible. [D p.16–17 fig.]
- **5.4** While turning, each step moves 1 px in the new direction and 1 px toward the center line of the new path, until he is on it. [D p.16]
- **5.5** If the tile ahead is a wall, he stops at the center of his tile. Holding the stick into a wall does nothing. [D p.16, P]

## 6. Ghost pathfinding
- **6.1** When a ghost enters a new tile, it looks ahead to the next tile along the way it will leave this one, and decides its exit from that tile. When it reaches that tile's center, it turns. [D p.26]
- **6.2** Candidate exits exclude the reverse direction, walls, and "up" in a red zone (2.5). [D p.26–27]
- **6.3** It picks the exit whose next tile has the smallest straight-line distance to the target. Ties go up, left, down, right, in that order. [D p.27]
- **6.4** A frightened ghost draws a random direction instead. If that is a wall or the reverse, it tries the next one clockwise until one works. [D p.14] The arcade's generator reads bits from its own program ROM, which we don't use. We use a seeded generator reset to the same seed at every level start and every lost life, which keeps the arcade's property that frightened paths repeat. [C]
- **6.5** Ghosts can't cut corners. They turn only at tile centers. [D p.16]

## 7. Targets
- **7.1** Scatter targets are outside the maze: Blinky (25,0), Pinky (2,0), Inky (27,35), Clyde (0,35). [D p.27 fig. (measured); P has 34 for the bottom row]
- **7.2** In chase mode, Blinky targets Pac-Man's tile. [D p.30]
- **7.3** Pinky targets 4 tiles ahead of Pac-Man. When Pac-Man faces up, it is 4 up and 4 left (the overflow bug). [D p.30–31]
- **7.4** Inky: take the tile 2 ahead of Pac-Man (2 up and 2 left when facing up), and double the vector from Blinky's tile to it. [D p.32–33]
- **7.5** Clyde targets Pac-Man when he is 8 or more tiles away (Euclidean distance), otherwise his own scatter target. [D p.34]
- **7.6** Eyes target (13,14), just above the left half of the door. [D p.27]

## 8. Scatter and chase
- **8.1** Schedule in seconds, alternating scatter and chase and ending in chase for the rest of the level. L1: 7, 20, 7, 20, 5, 20, 5. L2–4: 7, 20, 7, 20, 5, 1033, 1/60. L5+: 5, 20, 5, 20, 5, 1037, 1/60. [D p.13]
- **8.2** The schedule restarts at the start of every level and after every lost life, starting in scatter. [D p.13]
- **8.3** The mode timer pauses while ghosts are frightened. [D p.13]
- **8.4** These changes force ghosts to reverse: chase to scatter, scatter to chase, and either one to frightened. Leaving frightened mode does not reverse them. [D p.12]
- **8.5** A reversal takes effect when the ghost next enters a new tile. [D p.12]

## 9. Frightened mode
- **9.1** Eating an energizer reverses the ghosts and frightens them for this many seconds by level: 6, 5, 4, 3, 2, 5, 2, 2, 1, 5, 2, 1, 1, 3, 1, 1, 0, 1, then 0 from level 19 on. [D p.14 and Table A.1 (the PDF cuts off the last two columns); P levelspec_table has the same times]
- **9.2** On levels with 0 seconds the ghosts only reverse. [D p.14]
- **9.3** Ghosts flash 5 times before recovering, except 3 times on levels 9, 12, 13, 15, 16 and 18. We use 28 frames per flash. This is only visual. [D Table A.1 web edition; C for the flash length]
- **9.4** Ghosts eaten from one energizer are worth 200, 400, 800, then 1600. A new energizer resets the chain. [D p.10]
- **9.5** Eyes return to the house, and the revived ghost is not frightened even if the timer is still running. [D p.10, P]
- **9.6** Ghosts waiting in the house also become frightened and stay frightened after they leave. [D p.10, "all turn the same shade of blue"]

## 10. Ghost house
- **10.1** Blinky never waits. Revived Blinky leaves at once. [D p.19]
- **10.2** Pinky, Inky and Clyde are preferred in that order. Only the most preferred ghost waiting inside counts dots. [D p.19]
- **10.3** Dot limits: Pinky 0 always. Inky 30, 0, then 0. Clyde 60, 50, then 0, for levels 1, 2 and 3+. A ghost leaves when its count reaches or passes its limit. Counts reset at each level start. [D p.19]
- **10.4** After a lost life, the personal counters pause (without resetting) and a global counter starts at 0. It releases Pinky at 7 and Inky at 17. If Clyde is inside when it reaches 32, the global counter switches off and personal counters resume. Otherwise it keeps counting and never releases anyone again. [D p.19–20]
- **10.5** A timer counts time since the last dot. At 4 s (3 s from level 5 on) it releases the most preferred waiting ghost and restarts from 0. [D p.19]
- **10.6** Ghosts leave the house heading left, or right if the mode changed at least once while they were inside. [D p.20]
- **10.7** Energizers count as dots for 10.3, 10.4, 10.5, 12.1 and 15.1. [C, following D's "244 dots"; P counts only small dots for 10.3 and 10.5]

## 11. Collisions
- **11.1** Collisions are checked once per frame, after everyone has moved. Sharing a tile is a collision. So when Pac-Man and a ghost swap tiles in the same frame, they pass through each other, as in the arcade. [D p.25]
- **11.2** Pac-Man eats a frightened ghost. A chasing or scattering ghost kills Pac-Man. Eyes do nothing. [D p.10]

## 12. Cruise Elroy (Blinky)
- **12.1** Elroy 1 starts when this many dots are left, Elroy 2 at half that: L1 20, L2 30, L3–5 40, L6–8 50, L9–11 60, L12–14 80, L15–18 100, L19+ 120. [D Table A.1]
- **12.2** Elroy speeds (Elroy 1 / Elroy 2): L1 80/85, L2–4 90/95, L5+ 100/105. [D Table A.1]
- **12.3** As Elroy, Blinky keeps targeting Pac-Man during scatter mode (he still reverses at mode changes). [D p.30]
- **12.4** After a lost life, Elroy is suspended until Clyde starts to leave the house. [D p.30]

## 13. Scoring, lives, levels
- **13.1** Dots are 10 points, energizers 50, fruit as in 15.2. [D p.10]
- **13.2** The game starts with 3 lives and awards 1 extra life at 10,000 points (factory settings). [D; P]
- **13.3** Clearing all 244 dots moves to the next level: a fresh maze, positions reset, and the level's speeds and timers. [D p.10]
- **13.4** Level 256 (the "split screen") comes from ROM data spilling onto the screen, so it is not emulated. The game ends after level 255. [D p.35–37; C]

## 14. Pauses
- **14.1** Eating a ghost freezes play for 60 frames. Nothing moves and no timers run. [P]
- **14.2** A lost life or a cleared level pauses play, then resets positions. The RL wrapper skips these frames. [P, C]

## 15. Fruit
- **15.1** Fruit appears after 70 and after 170 dots are eaten, below the house at tile (13,20), for a random 9–10 seconds. [D p.10; P tile]
- **15.2** Symbols and points: cherries 100, strawberry 300, peach 500 (L3–4), apple 700 (L5–6), grapes 1000 (L7–8), galaxian 2000 (L9–10), bell 3000 (L11–12), key 5000 (L13+). The page draws its own generic icons, not Namco's. [D Table A.1]

## Our choices where the documents are silent
| rule | choice |
|---|---|
| 4.2 | Exact-average accumulator instead of the ROM's speed bit patterns |
| 4.7 | Eyes 1.5 px/frame and in-house 0.5 px/frame (pacman.c's estimates) |
| 6.4 | Seeded generator instead of reading ROM bytes; still reset per level and life |
| 7.1 | Bottom scatter row 35, measured from the Dossier figure |
| 9.3 | 28 frames per flash |
| 10.7 | Energizers count toward the house counters, the dot timer, Elroy and fruit |
| 13.4 | No level 256 |
| 14.1 | Everything freezes while a ghost is eaten |
