# snake_laya

Watch the `exp7_hybrid_decision` model ("Open-One") play Snake.

## How Open-One actually plays

exp7 is a text `choice` model: context + instructions + N text options ->
probability over options. It was trained on intent routing, MCQ, bool and
score tasks. It has no notion of a grid, and -- measured, not assumed -- it
cannot acquire one from a board description:

| framing | picks a safe move | dies |
|---|---|---|
| board coordinates via `describe()` | 0.47 | **0.53** |
| *random legal move (chance)* | *0.66* | *0.34* |

Feeding it the board was worse than random. Two measurements explain why:

1. **It never read the board.** Swapping in a *different* board's description
   left the model's output unchanged (best-move rate 0.71 real vs 0.74
   mismatched, chance 0.45). The apparent skill was a fixed prior over the
   direction labels.
2. **`describe()` pointed it at the moves that kill it.** The line naming
   which directions are fatal made those options *more* attractive -- a text
   matcher has no representation of negation, it is drawn to words it can
   see. Removing that line restored chance (0.68); inverting it to name the
   safe directions instead beat chance (0.82).

So the turn is reshaped into the question exp7 can answer. `SnakeGame`
computes the facts about each candidate move (`move_facts`: fatal, flood-fill
free space, room ratio, change in distance to food); `open_one_agent` renders each
move as an option string describing what that move *is*; the context is the
**request** -- what the snake wants. The model routes the request to the
best-matching option, which is exactly intent routing.

```
context: The snake is hunting food on a grid and must stay alive. It wants to
         move through open space that it can still get out of, on a route
         that takes it toward the food.
options: up    -- Category: a wide open region with room to spare, drifting away from the food.
         left  -- Category: a tight passage that only just fits, closing in on the food.
         down  -- Category: an open route with comfortable room, closing in on the food.
```

The descriptors are graded, not four canned strings, so the model is weighing
"narrow but closing on the food" against "wide open but drifting away" -- a
real preference judgment over option texts, not a bucket lookup. Same
architecture, same checkpoint, no retraining.

**What this does and does not claim.** Open-One is doing the selection; it is not
doing the spatial reasoning. The flood fill and the distance arithmetic happen
in `snake_env.py`. The honest description is "exp7 chooses between described
options", which is what it was built for -- not "exp7 understands a game
board", which the measurements above rule out.

## Files

- `snake_env.py` -- game rules, flood-fill free space, and `move_facts()`. No pygame, no model code.
- `open_one_agent.py` -- builds the `PackedExample` per tick and runs it through `HybridDecisionModel`. Falls back to a greedy heuristic if no checkpoint is given.
- `game.py` -- pygame rendering + main loop, with a HUD showing the model's confidence per direction.
- `server.py` -- web front-end (`/`) plus a playground (`/playground`) that shows the exact tokenized request the model receives.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python server.py                        # web UI, exp7a checkpoint, GPU if available
python game.py                          # pygame window
python game.py --ckpt ../checkpoints/exp7a_best_zeroshot_epoch6.pt
python game.py --human                  # play it yourself, arrow keys / WASD
```

Press `R` to restart after a game over, `Esc` to quit.

`--k` is clamped to the checkpoint's trained recurrent depth. exp7a was
trained at `k_max=1`; running more passes than the block was fit for is
off-distribution and degrades the output, so asking for more logs a warning
and clamps.
