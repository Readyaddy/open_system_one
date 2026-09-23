"""Pure game logic for Maze Runner -- no rendering, no model code. Mirrors
snake_env.py's split (rules here, exp7 bridge in maze_agent.py) so the two
games share the same shape: a background loop feeds MazeGame.move_facts()
per candidate move to the same describe-an-option-and-pick trick Snake uses.

Unlike Snake, a maze generated as a perfect maze (spanning tree over the
grid -- exactly one path between any two cells, no loops) has no dead ends
that trap the agent forever: every non-goal cell has at least one open
passage, so the agent can always move somewhere, including back the way it
came. That means Maze Runner has no "collision" failure mode -- only
"reached the goal" or "timed out" -- which makes it a cleaner test of pure
navigation than Snake, where survival and navigation are tangled together.
"""
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


DIRECTION_ORDER = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


@dataclass
class MoveFacts:
    """Everything true about one candidate move -- same shape as snake_env's
    MoveFacts (open_one_agent.py's rendering code is written against this
    interface, not against Snake specifically)."""
    direction: Direction
    free_space: int     # reachable cells beyond this move (flood fill within the maze's passages)
    room_ratio: float    # free_space / total maze cells -- how much of the maze this opens onto
    goal_delta: int      # change in Manhattan distance to the goal (-1 closer, +1 farther)
    goal_dist: int        # Manhattan distance to the goal after the move


class MazeGame:
    def __init__(self, width=12, height=12, seed=None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.passages = self._generate_maze()  # frozenset of frozenset({cellA, cellB}) -- an
        # open edge between two orthogonally-adjacent cells.
        self.start = (0, 0)
        self.goal = (self.width - 1, self.height - 1)
        self.pos = self.start
        self.steps = 0
        self.alive = True   # False once finished (reached goal OR timed out)
        self.won = False
        self.path = [self.pos]
        self._dist_from_goal = self._bfs_distances(self.goal)  # see move_facts()
        return self

    def _bfs_distances(self, source):
        """True shortest-path distance (in steps, through actual passages)
        from every cell to `source`. Computed once per maze -- the maze and
        goal are both fixed for the whole episode, so this never goes
        stale. This is NOT the same as Manhattan distance: a first version
        of this file used Manhattan distance as the "closer to the goal"
        signal, which is actively wrong inside a maze (walls mean the
        straight-line-closer neighbor is often the one that leads into a
        dead end). Verified: a policy that always picks the option with the
        smallest Manhattan distance solved 0/20 test mazes, stuck
        oscillating around walls it couldn't see. True graph distance does
        not have that failure mode -- see move_facts()."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            c = queue.popleft()
            x, y = c
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (x + dx, y + dy)
                if (0 <= nb[0] < self.width and 0 <= nb[1] < self.height
                        and nb not in dist and self._has_edge(c, nb)):
                    dist[nb] = dist[c] + 1
                    queue.append(nb)
        return dist

    # ------------------------------------------------------------------
    # Maze generation -- randomized depth-first backtracker (the standard
    # "recursive backtracker" algorithm): carve passages between cells by
    # DFS-walking an initially-fully-walled grid, only stepping into
    # unvisited cells, backtracking when stuck. The result is always a
    # spanning tree over the grid graph -- connected, no cycles, no
    # isolated cells -- which is what guarantees no dead-end traps (see
    # module docstring).
    # ------------------------------------------------------------------
    def _generate_maze(self):
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        visited = {c: False for c in cells}
        passages = set()

        start = (0, 0)
        stack = [start]
        visited[start] = True
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not visited[(nx, ny)]:
                    neighbors.append((nx, ny))
            if not neighbors:
                stack.pop()
                continue
            nxt = self.rng.choice(neighbors)
            passages.add(frozenset({(x, y), nxt}))
            visited[nxt] = True
            stack.append(nxt)
        return passages

    def _has_edge(self, a, b):
        return frozenset({a, b}) in self.passages

    def legal_directions(self):
        """Every direction with an open passage from the current cell --
        unlike Snake, reversing is allowed (there is no body to run into),
        so nothing is excluded on that basis. A cell in a perfect maze
        always has at least one open passage, so this is never empty."""
        x, y = self.pos
        out = []
        for d in DIRECTION_ORDER:
            dx, dy = d.value
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < self.width and 0 <= nb[1] < self.height and self._has_edge(self.pos, nb):
                out.append(d)
        return out

    def step(self, direction: Direction):
        if not self.alive:
            return self.alive
        x, y = self.pos
        dx, dy = direction.value
        nb = (x + dx, y + dy)
        self.steps += 1
        if self._has_edge(self.pos, nb):
            self.pos = nb
            self.path.append(self.pos)
        # An illegal direction (no passage) is simply a no-op tick -- like
        # Snake's fatal-move filter, open_one_agent.py never offers one of
        # these to the model in the first place, so this only matters for a
        # human/heuristic player pressing an invalid key.

        if self.pos == self.goal:
            self.alive = False
            self.won = True
        elif self.steps > self.width * self.height * 4:
            # Timeout guard, same constant Snake's starvation guard uses --
            # keeps a poorly-performing agent from running forever instead
            # of it being a meaningful "4x cells" threshold in itself.
            self.alive = False
            self.won = False
        return self.alive

    def free_space(self, direction):
        """Flood-fills the maze's passage graph starting from the cell one
        step in `direction`, counting reachable cells. Because the maze is
        a spanning tree, this is actually the same number (total cells - 1)
        for every open direction from every cell -- there is only one
        connected component. It's kept anyway (rather than hard-coded) so
        move_facts has the same shape as Snake's, and so this stays correct
        if maze generation ever stops guaranteeing a perfect maze (e.g. if
        loops are added later for difficulty)."""
        x, y = self.pos
        dx, dy = direction.value
        start = (x + dx, y + dy)
        if not (0 <= start[0] < self.width and 0 <= start[1] < self.height) or not self._has_edge(self.pos, start):
            return 0
        seen = {start}
        queue = deque([start])
        count = 0
        while queue:
            cx, cy = queue.popleft()
            count += 1
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (cx + ddx, cy + ddy)
                if (0 <= nb[0] < self.width and 0 <= nb[1] < self.height
                        and nb not in seen and self._has_edge((cx, cy), nb)):
                    seen.add(nb)
                    queue.append(nb)
        return count

    def move_facts(self, direction):
        x, y = self.pos
        dx, dy = direction.value
        nx, ny = x + dx, y + dy
        before = self._dist_from_goal[(x, y)]
        after = self._dist_from_goal[(nx, ny)]  # true remaining path length, not
        # straight-line distance -- see _bfs_distances() docstring for why that
        # distinction matters in a maze.
        space = self.free_space(direction)
        return MoveFacts(
            direction=direction, free_space=space,
            room_ratio=space / max(1, self.width * self.height),
            goal_delta=after - before, goal_dist=after,
        )

    def describe(self):
        """Human-readable board state -- for the UI/debugging only, same
        role as SnakeGame.describe(): NOT sent to the model (see
        maze_agent.py for why -- same reasoning as Snake's README)."""
        x, y = self.pos
        gx, gy = self.goal
        return (f"Maze {self.width}x{self.height}. Agent at ({x},{y}), "
                f"goal at ({gx},{gy}), {self.steps} steps taken.")
