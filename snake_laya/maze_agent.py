"""Bridge between exp7 and Maze Runner -- same "describe each surviving
option, let the choice model pick" trick as open_one_agent.py, adapted for
a task with a genuinely different signal shape.

Two things are simpler here than in Snake, both because a maze is a
spanning tree (see maze_env.py):

  - There is no "room" axis. free_space after any move is always the same
    constant (the entire rest of the maze is reachable from anywhere) --
    unlike Snake, where a move can genuinely wall itself into a small
    pocket. Presenting a "room" descriptor that never varies would be
    describing something without ever giving the model a real reason to
    care about it, so it's left out entirely.
  - goal_delta is always exactly -1 or +1 -- moving along any edge changes
    true distance-to-goal by exactly one step, never more (verified in
    maze_env.py's module docstring). So the whole task reduces to one
    binary distinction per option: does this lead toward the goal or away
    from it. That is the same kind of distinction exp7 already does
    reasonably well in Snake's food descriptor (Part A of the capability
    probe: it ranked "closing in on food" above "drifting away" in every
    clean-cut case tested) -- so this is a fair, honestly-easier task, not
    a rigged one.

Every offered option is survivable by construction (a maze has no fatal
moves -- see maze_env.py's docstring on why it can't trap the agent), so
build_request here only has two modes, never Snake's "trapped": "model"
(a real choice, 2+ options) and "forced" (exactly one open passage, e.g. a
dead-end corridor -- nothing to choose between, so the model isn't called).
"""
from maze_env import Direction, DIRECTION_ORDER  # noqa: F401 -- DIRECTION_ORDER kept
# for symmetry with open_one_agent.py's imports even though callers here
# mostly go through build_request.

CONTEXT_REQUEST = (
    "A robot is exploring a maze and wants to reach the marked exit as "
    "efficiently as possible, moving only through open passages."
)
INSTRUCTIONS = "Pick the option that best matches this request."


def _direction_label(d: Direction) -> str:
    return {
        Direction.UP: "moving up", Direction.DOWN: "moving down",
        Direction.LEFT: "moving left", Direction.RIGHT: "moving right",
    }[d]


def render_direction_option(d, facts) -> str:
    """label -- Category: <toward/away>. -- same rendering family as
    open_one_agent.py's render_direction_option (data.py's label_description
    mode + dataset_v7's "Category: {x}." template), so this checkpoint sees
    the same format it was trained on."""
    label = _direction_label(d)
    if facts.goal_delta < 0:
        detail = "an open passage that leads toward the goal, one step closer."
    else:
        detail = "an open passage that leads away from the goal, one step farther."
    return f"{label} -- Category: {detail}"


def build_request(game):
    """Mirrors open_one_agent.py's build_request. Returns (mode, moves,
    option_texts): "model" for a real choice, "forced" when the maze leaves
    only one open passage (a dead-end corridor) and there is nothing to
    decide, so the model is not called. There is no "trapped" mode -- see
    module docstring for why a maze can never do that to the agent."""
    legal = game.legal_directions()
    facts = [game.move_facts(d) for d in legal]

    if len(legal) == 1:
        return "forced", legal, []
    option_texts = [render_direction_option(d, f) for d, f in zip(legal, facts)]
    return "model", legal, option_texts
