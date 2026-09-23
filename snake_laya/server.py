"""Web front-end for snake_laya.

Runs the Snake game loop in a background thread on the server side (where the
GPU and the exp7 model live), and exposes the live state -- board, chosen
move, per-direction confidence, and wall-clock decision latency -- as JSON
over a tiny polling API. index.html renders it on a <canvas> and re-fetches
a few times a second, so it looks and feels like a live dashboard (board +
latency readout) rather than a static screenshot.

Usage:
    python server.py                         # exp7a checkpoint, GPU if available
    python server.py --ckpt "" --human       # heuristic agent (no model)
    python server.py --port 8000 --no-browser
"""
import argparse
import os
import random
import threading
import time
import webbrowser
from collections import deque

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

from snake_env import SnakeGame, DIRECTION_ORDER
from open_one_agent import INFERENCE_LOCK  # see its module docstring: every forward
# pass through the shared model/tokenizer -- Snake's own OpenOneAgent.decide(), and
# every inline model call below (quiz/maze/blackjack loops, and /api/inspect) -- must
# hold this lock. The HF fast tokenizer is not safe to call from two threads at once.
from quiz_agent import QuizBank, QUIZ_TYPES
from maze_env import MazeGame, DIRECTION_ORDER as MAZE_DIRECTION_ORDER
import maze_agent
from blackjack_env import BlackjackGame, hand_value
import blackjack_agent

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "exp7_latest.pt")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=None)

STATE_LOCK = threading.Lock()
STATE = {
    "board": {"width": 20, "height": 20},
    "snake": [],
    "food": None,
    "direction": "RIGHT",
    "score": 0,
    "steps": 0,
    "alive": True,
    "agent_label": "loading...",
    "probs": {},
    "chosen": None,
    "latency_ms": None,
    "avg_latency_ms": None,
    "tick": 0,
    "generation": 0,   # bumped on every reset, so the client can detect restarts
    "history": [],     # completed games this run, most recent first -- see _record_result
}
CONTROL = {"paused": False, "restart": False, "fps": 8.0}
MAX_HISTORY = 50
HISTORY = deque(maxlen=MAX_HISTORY)

# --------------------------------------------------------------------------
# Quiz mode -- exp7 doing the task it was actually trained on. Runs in its
# own background thread, independent of the Snake game loop, and shares the
# SAME loaded model/tokenizer/builder (AGENT.model etc., set once Snake's
# loop finishes loading) rather than loading a second copy -- see
# open_one_agent.py's _MODEL_CACHE for why a second load would be wasteful.
# --------------------------------------------------------------------------
QUIZ_LOCK = threading.Lock()
QUIZ_STATE = {
    "ready": False,       # False until AGENT has finished loading
    "qtype": None, "source": None, "prompt": None, "question": None,
    "options": [], "gold_idx": None, "chosen_idx": None, "probs": [],
    "correct": None, "latency_ms": None, "tick": 0,
    "stats": {t: {"n": 0, "correct": 0} for t in QUIZ_TYPES},
}
QUIZ_CONTROL = {"paused": False, "mode": "auto", "dwell": 3.0}

# --------------------------------------------------------------------------
# Maze Runner -- same shared-model pattern as quiz mode. See maze_env.py /
# maze_agent.py: unlike Snake, a maze can never trap the agent (it's a
# spanning tree), so the only failure mode is a timeout, not a collision.
# --------------------------------------------------------------------------
MAZE_LOCK = threading.Lock()
MAZE_HISTORY = deque(maxlen=MAX_HISTORY)
MAZE_STATE = {
    "ready": False, "width": 0, "height": 0, "pos": None, "goal": None,
    "path": [], "passages": [], "steps": 0, "alive": True, "won": False,
    "probs": {}, "chosen": None, "latency_ms": None, "avg_latency_ms": None,
    "tick": 0, "generation": 0, "history": [],
}
MAZE_CONTROL = {"paused": False, "restart": False, "fps": 4.0}

# --------------------------------------------------------------------------
# Blackjack -- same shared-model pattern again. See blackjack_env.py /
# blackjack_agent.py. Rounds resolve and restart on their own (no manual
# restart needed the way Snake/Maze have one) -- CONTROL only has pause/fps.
# --------------------------------------------------------------------------
BLACKJACK_LOCK = threading.Lock()
BLACKJACK_HISTORY = deque(maxlen=MAX_HISTORY)
BLACKJACK_STATE = {
    "ready": False, "player": [], "dealer": [], "phase": None, "result": None,
    "probs": {}, "chosen": None, "strategy_match": None, "latency_ms": None,
    "rounds_played": 0, "rounds_won": 0, "strategy_matches": 0, "strategy_decisions": 0,
    "tick": 0, "history": [],
}
BLACKJACK_CONTROL = {"paused": False, "dwell": 1.5}

# Set once by run_game_loop, then read (never mutated) by the /api/inspect and
# /api/live_context playground routes -- lets the playground query the SAME
# loaded model/tokenizer/builder the live game uses, and prefill from the
# actual live board, without spinning up a second copy of a ~400M-param model.
AGENT = None
LIVE_GAME = None


def run_game_loop(ckpt_path, width, height, use_human_heuristic_only, k, fixed_fps):
    global AGENT, LIVE_GAME
    game = SnakeGame(width=width, height=height)
    agent = None
    agent_label = "HEURISTIC"
    if not use_human_heuristic_only:
        from open_one_agent import OpenOneAgent
        agent = OpenOneAgent(ckpt_path=ckpt_path, k=k)
        agent_label = agent.label if agent.available else "HEURISTIC (no checkpoint loaded)"
    else:
        from open_one_agent import OpenOneAgent
        agent = OpenOneAgent(ckpt_path=None, k=k)  # forces the built-in heuristic

    AGENT = agent
    LIVE_GAME = game

    with STATE_LOCK:
        STATE["agent_label"] = agent_label
        STATE["board"] = {"width": width, "height": height}

    latencies = []
    result_recorded = False  # guards against re-appending the same finished
    # game to HISTORY on every subsequent poll while it sits dead/paused --
    # the loop keeps ticking (paused branch below) long after game.alive
    # goes False, so this can only be reset by an actual restart.

    while True:
        if CONTROL["restart"]:
            game.reset()
            latencies.clear()
            result_recorded = False
            CONTROL["restart"] = False
            with STATE_LOCK:
                STATE["generation"] += 1

        if CONTROL["paused"] or not game.alive:
            if game.alive is False and not result_recorded:
                _record_result(game, sum(latencies) / len(latencies) if latencies else None)
                result_recorded = True
            time.sleep(0.05)
            _publish(game, agent_label, probs=None, chosen=None, latency_ms=None, latencies=latencies)
            continue

        t0 = time.perf_counter()
        move, probs = agent.decide(game)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        game.step(move)

        latencies.append(latency_ms)
        if len(latencies) > 200:
            latencies.pop(0)

        _publish(game, agent_label, probs, move, latency_ms, latencies)

        # Pace ticks so a fast model doesn't make the board flicker
        # unwatchably fast -- but never wait longer than the model itself
        # took, so the UI still reflects real inference latency.
        target_dt = 1.0 / max(0.1, CONTROL["fps"])
        elapsed = time.perf_counter() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


def _record_result(game, avg_latency_ms):
    """One row per finished game -- board size travels with the row since
    score is not comparable across board sizes. Cause is read off the same
    signals SnakeGame.step() itself used to end the game (see snake_env.py),
    not re-derived heuristically."""
    if game.food is None:
        cause = "won"          # board filled completely
    elif game.steps_since_food > game.width * game.height * 4:
        cause = "starved"      # SnakeGame's starvation guard
    else:
        cause = "collision"    # wall or self
    HISTORY.appendleft({
        "score": game.score, "steps": game.steps, "cause": cause,
        "board": f"{game.width}x{game.height}",
        "avg_latency_ms": round(avg_latency_ms, 1) if avg_latency_ms else None,
    })
    with STATE_LOCK:
        STATE["history"] = list(HISTORY)


def _publish(game, agent_label, probs, chosen, latency_ms, latencies):
    with STATE_LOCK:
        STATE["snake"] = [list(p) for p in game.snake]
        STATE["food"] = list(game.food) if game.food else None
        STATE["direction"] = game.direction.name
        STATE["score"] = game.score
        STATE["steps"] = game.steps
        STATE["alive"] = game.alive
        STATE["agent_label"] = agent_label
        STATE["probs"] = {d.name: (probs.get(d, 0.0) if probs else 0.0) for d in DIRECTION_ORDER}
        STATE["chosen"] = chosen.name if chosen else None
        STATE["latency_ms"] = latency_ms
        STATE["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else None
        STATE["tick"] += 1


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/playground")
def playground_page():
    return send_from_directory(STATIC_DIR, "playground.html")


@app.route("/api/live_context")
def api_live_context():
    """Prefills the playground with exactly what the live game would send
    the model right now: the same request-shaped context and the same
    fact-derived option strings open_one_agent.decide() builds, just computed
    here instead of consumed by a real forward pass. The raw board text is
    returned alongside as `board`, for reading only -- the model is not
    given it (see open_one_agent.py for why)."""
    if LIVE_GAME is None:
        return jsonify({"error": "game not started yet"}), 503
    from open_one_agent import build_request, CONTEXT_REQUEST, INSTRUCTIONS

    game = LIVE_GAME
    mode, moves, option_texts = build_request(game)
    return jsonify({
        "context": CONTEXT_REQUEST,
        "instructions": INSTRUCTIONS,
        "board_text": game.describe(),  # shown in the playground for reading
        # only -- the model is NOT given the board (see open_one_agent.py).
        "options": option_texts,
        "option_labels": [d.name for d in moves],
        "mode": mode,  # "model" = a real choice was offered; "forced"/"trapped"
        # = the rules left nothing to decide and the model is not consulted.
        "board": {"width": game.width, "height": game.height},
        "snake": [list(p) for p in game.snake],
        "food": list(game.food) if game.food else None,
        "direction": game.direction.name,
    })


@app.route("/api/state_to_query", methods=["POST"])
def api_state_to_query():
    """Lets the playground build a query from a hand-specified board (grid
    size, snake body, food, heading) instead of free-typed English -- goes
    through the SAME SnakeGame.describe() / render_* functions the live game
    uses, so the generated context/options and the rendered board are
    guaranteed consistent with each other and with what the real game would
    actually send for that exact state."""
    from snake_env import SnakeGame, Direction
    from open_one_agent import build_request, CONTEXT_REQUEST, INSTRUCTIONS

    body = request.get_json(force=True, silent=True) or {}
    width = int(body.get("width", 20))
    height = int(body.get("height", 20))
    snake = [tuple(p) for p in body.get("snake", [])]
    food = body.get("food")
    direction_name = body.get("direction", "RIGHT")

    if len(snake) < 1:
        return jsonify({"error": "snake needs at least one segment (the head)"}), 400
    try:
        direction = Direction[direction_name]
    except KeyError:
        return jsonify({"error": f"unknown direction {direction_name!r}"}), 400

    game = SnakeGame(width=width, height=height)
    from collections import deque
    game.snake = deque(snake)
    game.direction = direction
    game.food = tuple(food) if food else None
    game.alive = True

    mode, moves, option_texts = build_request(game)
    return jsonify({
        "context": CONTEXT_REQUEST,
        "instructions": INSTRUCTIONS,
        "board_text": game.describe(),
        "options": option_texts,
        "option_labels": [d.name for d in moves],
        "mode": mode,  # "model" = a real choice was offered; "forced"/"trapped"
        # = the rules left nothing to decide and the model is not consulted.
        "board": {"width": game.width, "height": game.height},
        "snake": [list(p) for p in game.snake],
        "food": list(game.food) if game.food else None,
        "direction": game.direction.name,
    })


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """The actual playground: takes a context/instructions/option-list,
    builds the exact same PackedExample -> PackedSequenceBuilder pipeline
    model.py uses, and returns the full token-by-token breakdown (which
    positions are [MASK], which span is "context") alongside the model's
    real output -- so the request/response can be checked by eye instead of
    trusted blind."""
    if AGENT is None or not AGENT.available:
        return jsonify({"error": "no exp7 checkpoint loaded (agent unavailable)"}), 503

    body = request.get_json(force=True, silent=True) or {}
    context = (body.get("context") or "").strip()
    instructions = body.get("instructions") or ""
    options = [o for o in (body.get("options") or []) if o.strip()]
    if len(options) < 1:
        return jsonify({"error": "provide at least one option"}), 400

    from model import PackedExample

    ex = PackedExample(context=context, instructions=instructions, option_texts=options,
                        qtype="choice", answer_idx=0)

    # Same builder the live game uses -- build_one() is the unbatched,
    # pre-padding step, exactly what NOTES.md Sec 2.2 describes.
    ids, mask_positions, context_token_mask = AGENT.builder.build_one(ex)
    tokens = AGENT.tokenizer.convert_ids_to_tokens(ids)
    mask_pos_set = set(mask_positions)
    token_rows = [
        {
            "idx": i,
            "token": tok,
            "is_mask": i in mask_pos_set,
            "is_context": bool(context_token_mask[i]),
            "option_index": mask_positions.index(i) if i in mask_pos_set else None,
        }
        for i, tok in enumerate(tokens)
    ]

    with INFERENCE_LOCK:
        batch = AGENT.builder.build_batch([ex], AGENT.device)
        with torch.no_grad():
            logits_per_depth = AGENT.model(AGENT.tokenizer, batch, AGENT.device, k=AGENT.k)
    probs = F.softmax(logits_per_depth[-1][0].float(), dim=-1).tolist()
    top_idx = int(torch.tensor(probs).argmax().item())

    return jsonify({
        "tokens": token_rows,
        "sequence_length": len(ids),
        "num_mask_tokens": len(mask_positions),
        "num_options": len(options),
        "options": options,
        "probs": probs,
        "top_index": top_idx,
        "top_option": options[top_idx],
        "depths_returned": len(logits_per_depth),
    })


@app.route("/api/control", methods=["POST"])
def api_control():
    body = request.get_json(force=True, silent=True) or {}
    if "paused" in body:
        CONTROL["paused"] = bool(body["paused"])
    if "restart" in body and body["restart"]:
        CONTROL["restart"] = True
    if "fps" in body:
        try:
            CONTROL["fps"] = max(0.5, min(30.0, float(body["fps"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "control": CONTROL})


def run_quiz_loop():
    """Runs independently of the Snake game loop -- waits for AGENT to
    finish loading (Snake's loop owns that load), then samples and answers
    quiz questions forever, reusing AGENT's already-loaded model/tokenizer/
    builder. Heuristic mode (no checkpoint) has nothing to quiz with, so the
    loop just stays not-ready."""
    while AGENT is None or not AGENT.available:
        time.sleep(0.2)

    from model import PackedExample
    import torch.nn.functional as _F
    from torch.amp import autocast as _autocast

    bank = QuizBank(seed=random.randrange(1 << 30))
    model, tok, builder, device = AGENT.model, AGENT.tokenizer, AGENT.builder, AGENT.device

    with QUIZ_LOCK:
        QUIZ_STATE["ready"] = True

    while True:
        if QUIZ_CONTROL["paused"]:
            time.sleep(0.1)
            continue

        mode = QUIZ_CONTROL["mode"]
        qtype = random.choice(QUIZ_TYPES) if mode == "auto" else mode
        ex, meta = bank.sample(qtype)

        t0 = time.perf_counter()
        with INFERENCE_LOCK:
            with torch.no_grad():
                batch = builder.build_batch([ex], device)
                with _autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits_per_depth = model(tok, batch, device, k=AGENT.k)
                probs = _F.softmax(logits_per_depth[-1][0].float(), dim=-1).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        chosen_idx = max(range(len(probs)), key=lambda i: probs[i])
        correct = chosen_idx == ex.answer_idx

        with QUIZ_LOCK:
            st = QUIZ_STATE["stats"][meta["kind"]]
            st["n"] += 1
            st["correct"] += int(correct)
            QUIZ_STATE.update({
                "qtype": meta["kind"], "source": meta["source"],
                "prompt": meta["prompt"], "question": meta.get("question"),
                "options": ex.option_texts, "gold_idx": ex.answer_idx,
                "chosen_idx": chosen_idx, "probs": probs, "correct": correct,
                "latency_ms": latency_ms,
            })
            QUIZ_STATE["tick"] += 1

        time.sleep(max(0.2, QUIZ_CONTROL["dwell"]))


@app.route("/api/quiz_state")
def api_quiz_state():
    with QUIZ_LOCK:
        return jsonify(QUIZ_STATE)


@app.route("/api/quiz_control", methods=["POST"])
def api_quiz_control():
    body = request.get_json(force=True, silent=True) or {}
    if "paused" in body:
        QUIZ_CONTROL["paused"] = bool(body["paused"])
    if "mode" in body and body["mode"] in ("auto", *QUIZ_TYPES):
        QUIZ_CONTROL["mode"] = body["mode"]
    if "dwell" in body:
        try:
            QUIZ_CONTROL["dwell"] = max(0.5, min(15.0, float(body["dwell"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "control": QUIZ_CONTROL})


def _record_maze_result(game, avg_latency_ms):
    """One row per finished maze run. `optimal_steps` is the true
    shortest-path length computed once at reset() (see maze_env.py's
    _bfs_distances) -- comparing actual steps to it is a genuine efficiency
    measure, not a guess, since greedy-on-true-distance is provably optimal
    in a spanning-tree maze (verified when maze_env.py was written)."""
    optimal = game._dist_from_goal.get(game.start)
    MAZE_HISTORY.appendleft({
        "won": game.won, "steps": game.steps, "optimal_steps": optimal,
        "board": f"{game.width}x{game.height}",
        "avg_latency_ms": round(avg_latency_ms, 1) if avg_latency_ms else None,
    })
    with MAZE_LOCK:
        MAZE_STATE["history"] = list(MAZE_HISTORY)


def _passages_json(game):
    """The maze's open edges as [[x1,y1],[x2,y2]] pairs -- the client needs
    this to draw walls (every adjacent cell pair NOT in this list is a
    wall). Cheap to recompute per tick (at most width*height-1 edges -- a
    maze is a spanning tree, see maze_env.py) and it only actually changes
    on a new maze, but recomputing avoids a second piece of restart-tracking
    state getting out of sync with the game itself."""
    return [[list(a), list(b)] for a, b in (tuple(edge) for edge in game.passages)]


def run_maze_loop(width=12, height=12, fixed_fps=4.0):
    """Same shared-model pattern as run_quiz_loop -- waits for Snake's loop
    to finish loading AGENT, then runs Maze Runner forever in its own
    thread. "forced" moves (a dead-end corridor with exactly one open
    passage -- see maze_agent.py) skip the model call entirely, same
    reasoning as Snake's fatal-move filter: nothing to choose between."""
    while AGENT is None or not AGENT.available:
        time.sleep(0.2)

    from model import PackedExample
    import torch.nn.functional as _F
    from torch.amp import autocast as _autocast

    model, tok, builder, device = AGENT.model, AGENT.tokenizer, AGENT.builder, AGENT.device
    game = MazeGame(width=width, height=height)

    with MAZE_LOCK:
        MAZE_STATE["ready"] = True
        MAZE_STATE["width"], MAZE_STATE["height"] = width, height

    MAZE_CONTROL["fps"] = fixed_fps
    latencies = []
    result_recorded = False

    while True:
        if MAZE_CONTROL["restart"]:
            game.reset()
            latencies.clear()
            result_recorded = False
            MAZE_CONTROL["restart"] = False
            with MAZE_LOCK:
                MAZE_STATE["generation"] += 1

        if MAZE_CONTROL["paused"] or not game.alive:
            if game.alive is False and not result_recorded:
                _record_maze_result(game, sum(latencies) / len(latencies) if latencies else None)
                result_recorded = True
            time.sleep(0.05)
            with MAZE_LOCK:
                MAZE_STATE.update({
                    "pos": list(game.pos), "goal": list(game.goal), "path": [list(p) for p in game.path],
                    "passages": _passages_json(game), "steps": game.steps, "alive": game.alive, "won": game.won,
                })
            continue

        t0 = time.perf_counter()
        mode, moves, option_texts = maze_agent.build_request(game)
        probs_by_dir = {d.name: 0.0 for d in MAZE_DIRECTION_ORDER}
        if mode == "forced":
            chosen = moves[0]
            probs_by_dir[chosen.name] = 1.0
        else:
            ex = PackedExample(context=maze_agent.CONTEXT_REQUEST, instructions=maze_agent.INSTRUCTIONS,
                                option_texts=option_texts, qtype="choice", answer_idx=0)
            with INFERENCE_LOCK:
                with torch.no_grad():
                    batch = builder.build_batch([ex], device)
                    with _autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        logits_per_depth = model(tok, batch, device, k=AGENT.k)
                    probs = _F.softmax(logits_per_depth[-1][0].float(), dim=-1).tolist()
            for d, p in zip(moves, probs):
                probs_by_dir[d.name] = p
            chosen = moves[max(range(len(probs)), key=lambda i: probs[i])]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        game.step(chosen)

        latencies.append(latency_ms)
        if len(latencies) > 200:
            latencies.pop(0)

        with MAZE_LOCK:
            MAZE_STATE.update({
                "pos": list(game.pos), "goal": list(game.goal), "path": [list(p) for p in game.path],
                "passages": _passages_json(game), "steps": game.steps, "alive": game.alive, "won": game.won,
                "probs": probs_by_dir, "chosen": chosen.name, "latency_ms": latency_ms,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            })
            MAZE_STATE["tick"] += 1

        target_dt = 1.0 / max(0.1, MAZE_CONTROL["fps"])
        elapsed = time.perf_counter() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


@app.route("/api/maze_state")
def api_maze_state():
    with MAZE_LOCK:
        return jsonify(MAZE_STATE)


@app.route("/api/maze_control", methods=["POST"])
def api_maze_control():
    body = request.get_json(force=True, silent=True) or {}
    if "paused" in body:
        MAZE_CONTROL["paused"] = bool(body["paused"])
    if "restart" in body and body["restart"]:
        MAZE_CONTROL["restart"] = True
    if "fps" in body:
        try:
            MAZE_CONTROL["fps"] = max(0.5, min(20.0, float(body["fps"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "control": MAZE_CONTROL})


def _record_blackjack_result(game):
    BLACKJACK_HISTORY.appendleft({
        "result": game.result, "player_total": hand_value(game.player)[0],
        "dealer_total": hand_value(game.dealer)[0],
    })
    with BLACKJACK_LOCK:
        BLACKJACK_STATE["history"] = list(BLACKJACK_HISTORY)


def run_blackjack_loop():
    """Same shared-model pattern again. Rounds resolve and a new one starts
    automatically -- there is no "restart" control the way Snake/Maze have
    one, since a finished round isn't a failure state to recover from, just
    the normal end of a turn."""
    while AGENT is None or not AGENT.available:
        time.sleep(0.2)

    from model import PackedExample
    import torch.nn.functional as _F
    from torch.amp import autocast as _autocast

    model, tok, builder, device = AGENT.model, AGENT.tokenizer, AGENT.builder, AGENT.device
    game = BlackjackGame(rng=random.Random(random.randrange(1 << 30)))
    result_recorded = False

    with BLACKJACK_LOCK:
        BLACKJACK_STATE["ready"] = True

    def _publish(chosen=None, probs_by_action=None, strategy_match=None, latency_ms=None):
        reveal_dealer = game.phase != "player_turn"
        with BLACKJACK_LOCK:
            BLACKJACK_STATE.update({
                "player": [list(c) for c in game.player],
                "dealer": [list(c) for c in game.dealer] if reveal_dealer else [list(game.dealer[0]), None],
                "phase": game.phase, "result": game.result,
                "probs": probs_by_action or {}, "chosen": chosen,
                "strategy_match": strategy_match, "latency_ms": latency_ms,
                "rounds_played": game.rounds_played, "rounds_won": game.rounds_won,
            })
            BLACKJACK_STATE["tick"] += 1

    while True:
        if BLACKJACK_CONTROL["paused"]:
            time.sleep(0.1)
            continue

        mode, actions, option_texts = blackjack_agent.build_request(game)

        if mode == "done":
            if not result_recorded:
                _record_blackjack_result(game)
                result_recorded = True
            _publish()
            time.sleep(max(0.5, BLACKJACK_CONTROL["dwell"]))
            game.reset()
            result_recorded = False
            continue

        t0 = time.perf_counter()
        ex = PackedExample(context=blackjack_agent.CONTEXT_REQUEST, instructions=blackjack_agent.INSTRUCTIONS,
                            option_texts=option_texts, qtype="choice", answer_idx=0)
        with INFERENCE_LOCK:
            with torch.no_grad():
                batch = builder.build_batch([ex], device)
                with _autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits_per_depth = model(tok, batch, device, k=AGENT.k)
                probs = _F.softmax(logits_per_depth[-1][0].float(), dim=-1).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        chosen_idx = max(range(len(probs)), key=lambda i: probs[i])
        chosen = actions[chosen_idx]
        matched = blackjack_agent.strategy_match(game, chosen)  # read BEFORE step() -- it
        # needs the hand as it stood when the decision was made.

        with BLACKJACK_LOCK:
            BLACKJACK_STATE["strategy_decisions"] += 1
            BLACKJACK_STATE["strategy_matches"] += int(matched)

        game.step(chosen)
        _publish(chosen=chosen, probs_by_action=dict(zip(actions, probs)),
                 strategy_match=matched, latency_ms=latency_ms)

        time.sleep(max(0.3, BLACKJACK_CONTROL["dwell"]))


@app.route("/api/blackjack_state")
def api_blackjack_state():
    with BLACKJACK_LOCK:
        return jsonify(BLACKJACK_STATE)


@app.route("/api/blackjack_control", methods=["POST"])
def api_blackjack_control():
    body = request.get_json(force=True, silent=True) or {}
    if "paused" in body:
        BLACKJACK_CONTROL["paused"] = bool(body["paused"])
    if "dwell" in body:
        try:
            BLACKJACK_CONTROL["dwell"] = max(0.3, min(8.0, float(body["dwell"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "control": BLACKJACK_CONTROL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                     help="exp7 checkpoint path. Pass --ckpt '' to force the heuristic agent.")
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--fps", type=float, default=8.0, help="Target game ticks/sec (won't exceed real model speed).")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    CONTROL["fps"] = args.fps
    use_heuristic_only = (args.ckpt == "")

    t = threading.Thread(
        target=run_game_loop,
        args=(args.ckpt, args.width, args.height, use_heuristic_only, args.k, args.fps),
        daemon=True,
    )
    t.start()

    if not use_heuristic_only:
        # Quiz/Maze/Blackjack all need real model weights -- skip them in
        # --ckpt "" mode, same guard the Snake side uses for the heuristic
        # fallback. All three run regardless of which dashboard tab is
        # open (see static/index.html's tab-switch comment) -- switching
        # tabs only changes which one this page polls and renders.
        threading.Thread(target=run_quiz_loop, daemon=True).start()
        threading.Thread(target=run_maze_loop, daemon=True).start()
        threading.Thread(target=run_blackjack_loop, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.get("chrome").open(url) if _has_chrome() else webbrowser.open(url)).start()

    print(f"[server] serving on {url}", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


def _has_chrome():
    try:
        webbrowser.get("chrome")
        return True
    except webbrowser.Error:
        return False


if __name__ == "__main__":
    main()
