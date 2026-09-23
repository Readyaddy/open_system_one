"""Pure game logic for Snake -- no rendering, no model code.

Kept separate from game.py (pygame) and open_one_agent.py (exp7 bridge) so the
rules of the game don't get tangled up with either the graphics or the
model-specific text encoding. `SnakeGame.describe()` is the only method the
agent side needs to call.
"""
import random
from collections import deque
from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Order fed to the model as the four "choice" options -- fixed, so the
# option->direction mapping never has to be reconstructed elsewhere.
DIRECTION_ORDER = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


@dataclass
class MoveFacts:
    """Everything that is true about one candidate move, computed by the game
    itself. This is the whole interface the model side consumes.

    The split matters: the exp7 model is a text option-scorer, not a spatial
    reasoner -- feeding it raw coordinates and expecting it to derive these
    facts does not work (it ignores the board entirely; see README). So the
    rules engine states the facts and the model does what it was actually
    trained to do: read several described options and pick the one that best
    matches what was asked for.
    """
    direction: "Direction"
    fatal: bool         # moving here ends the game this tick
    free_space: int     # reachable empty cells after the move (flood fill)
    room_ratio: float   # free_space / snake length -- 1.0 = only just fits
    food_delta: int     # change in Manhattan distance to food (-1 closer, +1 away)
    food_dist: int      # Manhattan distance to food after the move


class SnakeGame:
    def __init__(self, width=20, height=20, seed=None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.alive = True
        self.food = self._spawn_food()
        return self

    def _spawn_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        if not free:
            return None
        return self.rng.choice(free)

    def legal_directions(self):
        """Every direction except the immediate reverse -- reversing is a
        guaranteed self-collision, not a meaningful choice."""
        return [d for d in DIRECTION_ORDER if d is not OPPOSITE[self.direction]]

    def step(self, direction: Direction):
        """Advances the game one tick. Reversing into your own neck is
        treated as if the previous direction were kept, matching classic
        Snake rules, rather than an instant death -- the death still comes
        from the resulting body collision if the model insists on it."""
        if not self.alive:
            return self.alive

        if direction is not OPPOSITE[self.direction]:
            self.direction = direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        self.steps += 1
        self.steps_since_food += 1

        if (
            new_head[0] < 0 or new_head[0] >= self.width
            or new_head[1] < 0 or new_head[1] >= self.height
            or new_head in self.snake
        ):
            self.alive = False
            return self.alive

        self.snake.appendleft(new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:
                self.alive = False  # board filled -- won
        else:
            self.snake.pop()

        # Starvation guard: prevents an agent that loops forever from
        # running the game indefinitely.
        if self.steps_since_food > self.width * self.height * 4:
            self.alive = False

        return self.alive

    def free_space(self, direction):
        """Flood-fills the empty region the head would land in. The tail cell
        is treated as free because it moves out of the way on the same tick --
        without that, following your own tail (the standard way to survive a
        near-full board) scores as a dead end."""
        head_x, head_y = self.snake[0]
        dx, dy = direction.value
        start = (head_x + dx, head_y + dy)
        if (start[0] < 0 or start[0] >= self.width
                or start[1] < 0 or start[1] >= self.height or start in self.snake):
            return 0
        blocked = set(list(self.snake)[:-1])
        seen = {start}
        queue = deque([start])
        count = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxt = (x + ddx, y + ddy)
                if (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height
                        and nxt not in blocked and nxt not in seen):
                    seen.add(nxt)
                    queue.append(nxt)
        return count

    def move_facts(self, direction):
        """The full fact-set for one candidate move -- see MoveFacts."""
        head_x, head_y = self.snake[0]
        dx, dy = direction.value
        nx, ny = head_x + dx, head_y + dy
        fatal = (nx < 0 or nx >= self.width or ny < 0 or ny >= self.height
                 or (nx, ny) in self.snake)
        fx, fy = self.food if self.food else (head_x, head_y)
        before = abs(fx - head_x) + abs(fy - head_y)
        after = abs(fx - nx) + abs(fy - ny)
        space = 0 if fatal else self.free_space(direction)
        return MoveFacts(
            direction=direction, fatal=fatal, free_space=space,
            room_ratio=space / max(1, len(self.snake)),
            food_delta=after - before, food_dist=after,
        )

    def describe(self):
        """Renders the current state as plain text -- this is the `context`
        string fed into the exp7 model's packed sequence (see open_one_agent.py).
        Deliberately verbose/explicit rather than a compact encoding: the
        model has never seen a game-state string in training, so plain
        English coordinates give it the best chance of the backbone's
        pretrained spatial/numeric priors actually transferring."""
        head = self.snake[0]
        body = list(self.snake)[1:]
        fx, fy = self.food if self.food else (-1, -1)
        dx, dy = fx - head[0], fy - head[1]

        horiz = f"{abs(dx)} cells to the {'right' if dx > 0 else 'left'}" if dx != 0 else "aligned horizontally"
        vert = f"{abs(dy)} cells {'down' if dy > 0 else 'up'}" if dy != 0 else "aligned vertically"

        lines = [
            f"Snake game on a {self.width}x{self.height} grid, "
            f"columns 0-{self.width - 1} left-to-right, rows 0-{self.height - 1} top-to-bottom.",
            f"Snake head is at column {head[0]}, row {head[1]}, currently moving {self.direction.name}.",
            f"Snake body (excluding head, in order from neck to tail): "
            f"{body if body else 'none'}.",
            f"Snake length is {len(self.snake)}.",
            f"Food is at column {fx}, row {fy}, which is {horiz} and {vert} from the head.",
        ]

        danger = []
        for d in DIRECTION_ORDER:
            ddx, ddy = d.value
            nx, ny = head[0] + ddx, head[1] + ddy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or (nx, ny) in self.snake:
                danger.append(d.name)
        if danger:
            lines.append(f"Moving {', '.join(danger)} would immediately hit a wall or the snake's own body.")
        else:
            lines.append("No immediately adjacent direction is fatal this turn.")

        return "\n".join(lines)
