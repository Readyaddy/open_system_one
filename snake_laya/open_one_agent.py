"""Bridge between the exp7 hybrid-decision text model and the Snake game.

exp7 (experiments/exp7_hybrid_decision/model.py) is a `choice`-type decision
model: it takes a text context, text instructions, and a list of text
options, and returns a probability distribution over the options. It has no
notion of a grid or of movement -- so every game tick is framed as one
`choice` question:

    context      = SnakeGame.describe()  (see snake_env.py)
    instructions = sampled from exp7's own INSTRUCTION_BANK["choice"] (data.py)
    options      = the legal moves this turn (reverse excluded), each
                   rendered with exp7's own label_description template style
                   (render_direction_option, below) instead of plain English

and the model's argmax over the options is read back as a Direction.

If no checkpoint is available yet (exp7 hasn't finished training/downloading),
`OpenOneAgent` falls back to a simple greedy heuristic so the game is still
playable end-to-end -- swap in the checkpoint later with no other code
changes.
"""
import os
import random
import sys
import threading

import torch
import torch.nn.functional as F
from torch.amp import autocast

from snake_env import Direction, DIRECTION_ORDER

AUTOCAST_DTYPE = torch.bfloat16

# Keyed by (checkpoint path, device). Weights are read-only at inference, so
# sharing one instance between every OpenOneAgent in the process is safe.
_MODEL_CACHE = {}

# Sharing the model/tokenizer across threads is NOT automatically safe just
# because the weights are read-only: HuggingFace's fast (Rust-backed)
# tokenizer panics with "RuntimeError: Already borrowed" if two threads call
# it at the same instant (its internal Rust RefCell isn't reentrant across
# threads the way a plain Python object would be). Once snake_laya grew to
# four concurrent loops (Snake, Quiz, Maze, Blackjack) all sharing one
# OpenOneAgent, this fired for real and silently killed whichever thread's
# call lost the race -- Python doesn't restart a crashed thread, so that
# game just froze on its last published state. Every caller that runs a
# forward pass through the shared model/tokenizer -- OpenOneAgent.decide()
# below, and server.py's quiz/maze/blackjack loops -- must hold this lock
# for the tokenize+forward span. It serializes inference across all four
# games, which costs latency under contention but is correct; a broken game
# is a worse trade than a slower one.
INFERENCE_LOCK = threading.Lock()

EXP7_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "exp7_hybrid_decision")
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
sys.path.insert(0, os.path.abspath(EXP7_DIR))  # inserted last -> searched first, so exp7's
# own model.py wins over the unrelated src/model.py (JEPADecisionModel) that
# also happens to be importable as `model`.

import model as _exp7_model  # noqa: E402,F401 -- must be imported (and thus cached in
# sys.modules under the bare name "model") BEFORE `data` below. data.py does its own
# `sys.path.insert(0, .../src)` at module-import time, which re-shadows "model" with
# src/model.py's unrelated JEPADecisionModel; pre-importing here means Python reuses
# this already-cached module instead of re-resolving the (by-then-shadowed) path.

from data import INSTRUCTION_BANK  # noqa: E402 -- same paraphrase bank used for
# real "choice" training instructions.

# --------------------------------------------------------------------------
# Framing (see README "How Open-One actually plays"). The earlier version of this
# file handed the model raw board coordinates via SnakeGame.describe() and
# asked it to work out the move. Measured: the model's output was unchanged
# when fed a DIFFERENT board's description -- it never read the board at all,
# and on boards with a fatal move it died 53% of the time (random play dies
# 34%), because describe() named the deadly directions and a text matcher is
# pulled toward words it can see.
#
# So the question is reshaped into the one exp7 was actually trained on:
# intent routing. The game states the facts about each candidate move
# (SnakeGame.move_facts) and renders them as option text; the context is the
# REQUEST ("what the snake wants"); the model routes the request to the
# best-matching option. Same architecture, same checkpoint, no retraining --
# it is now being asked a question of the type it can answer.
# --------------------------------------------------------------------------

# Graded descriptors rather than a small set of fixed strings: the model has
# to weigh "narrow but closing on the food" against "wide open but drifting
# away", which is a real preference judgment over the option texts, not a
# lookup of one canned phrase per bucket.

def _space_phrase(facts) -> str:
    r = facts.room_ratio
    if r >= 6.0:
        return "a wide open region with room to spare"
    if r >= 3.0:
        return "an open route with comfortable room"
    if r >= 1.5:
        return "a workable corridor with some room"
    if r >= 1.0:
        return "a tight passage that only just fits"
    return "a cramped pocket far too small to escape from"


def _food_phrase(facts) -> str:
    if facts.food_delta < 0:
        return "closing in on the food"
    if facts.food_delta == 0:
        return "holding level with the food"
    return "drifting away from the food"


LIVE_OPTION = "{name} -- Category: {space}, {food}."


def render_direction_option(d, facts) -> str:
    """One option string per candidate move, built from that move's facts.

    Deliberately in the `label -- Category: <description>.` shape: that is
    data.py's `label_description` rendering mode with dataset_v7's "Category:
    {x}." template, i.e. a format this checkpoint saw throughout training.

    Only ever called for survivable moves -- see `build_request`.
    """
    assert not facts.fatal, "fatal moves are filtered out before rendering"
    return LIVE_OPTION.format(name=d.name.lower(), space=_space_phrase(facts),
                              food=_food_phrase(facts))


def build_request(game):
    """Decides what (if anything) to ask the model this tick.

    Returns `(mode, moves, option_texts)`:
      - "model"   -- >=2 survivable moves; a real choice, ask the model.
      - "forced"  -- exactly 1 survivable move; there is nothing to choose
                     between, so the model is not called at all. Saves a
                     ~60ms forward pass, and avoids a 1-option softmax whose
                     output is 1.0 by construction and means nothing.
      - "trapped" -- 0 survivable moves; every move dies this tick. Nothing
                     the model could say changes that.

    Fatal moves are never offered. Survival is a hard rule of the game, not a
    preference to be weighed, so the rules engine enforces it and the model
    is left to do what it is good at: choosing between viable routes. This is
    the same reasoning that already excludes the reverse move -- an option
    that can never be correct is not a real choice, and offering one only
    gives the softmax a way to lose.

    The cost of this is worth stating plainly: the model can no longer kill
    the snake directly, so the game no longer tests whether it avoids death.
    It tests navigation among survivable routes. Deaths that remain are
    "trapped" -- walked into a pocket several moves earlier -- and those are
    still genuinely the model's doing.
    """
    legal = game.legal_directions()
    facts = [game.move_facts(d) for d in legal]
    survivable = [(d, f) for d, f in zip(legal, facts) if not f.fatal]

    if not survivable:
        return "trapped", legal[:1], []
    if len(survivable) == 1:
        return "forced", [survivable[0][0]], []
    moves = [d for d, _ in survivable]
    return "model", moves, [render_direction_option(d, f) for d, f in survivable]


CONTEXT_REQUEST = (
    "The snake is hunting food on a grid and must stay alive. "
    "It wants to move through open space that it can still get out of, "
    "on a route that takes it toward the food."
)

# Fixed, not sampled. Training randomized instructions to stop the model
# overfitting one phrasing; at inference that randomness is pure variance --
# the same board would get different answers on different ticks. This is the
# bank entry whose phrasing matches what is being asked here.
INSTRUCTIONS = "Pick the option that best matches this request."
assert INSTRUCTIONS in INSTRUCTION_BANK["choice"]


class OpenOneAgent:
    """Wraps the exp7 HybridDecisionModel for one-decision-per-tick inference.

    Pass `ckpt_path=None` (or a path that doesn't exist yet) to run in
    fallback mode -- useful while the real checkpoint is still training or
    downloading.
    """

    def __init__(self, ckpt_path=None, device=None, k=None, seed=0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.available = False
        self.model = None
        self.tokenizer = None
        self.builder = None
        self.rng = random.Random(seed)
        self.epoch = None
        self.val_acc = None
        self.zero_shot_acc = None
        self.label = "HEURISTIC"

        if ckpt_path and os.path.exists(ckpt_path):
            self._load(ckpt_path)
        else:
            if ckpt_path:
                print(f"[open_one_agent] checkpoint not found at {ckpt_path!r} -- "
                      f"falling back to the heuristic agent.", flush=True)
            else:
                print("[open_one_agent] no checkpoint given -- running the heuristic agent.", flush=True)

    def _load(self, ckpt_path):
        from model import HybridDecisionModel, PackedSequenceBuilder, get_tokenizer, BACKBONE

        key = (os.path.abspath(ckpt_path), str(self.device))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            # A second OpenOneAgent in the same process (the web server builds
            # one for the game loop, and the /api/inspect playground wants
            # the same weights) must not pay for the load twice, or worse
            # hold two ~1.8GB copies of identical weights.
            self.model, self.tokenizer, self.builder, saved_args, meta = cached
            print(f"[open_one_agent] reusing the model already loaded from {ckpt_path}", flush=True)
        else:
            # mmap=True + map_location="cpu": the checkpoint on disk is ~2.6GB
            # but only `model_state` (~1.8GB) is ever wanted -- the rest is
            # optimizer state from training. Memory-mapping means the
            # optimizer tensors are never paged in at all, and the read drops
            # from seconds to ~0.15s. weights_only=True is both safer and
            # required for the mmap path to be worth anything.
            ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
            saved_args = ckpt.get("args", {})
            backbone = saved_args.get("backbone", BACKBONE)
            self.tokenizer = get_tokenizer(backbone)

            # Build the backbone from CONFIG, not from_pretrained(). The
            # default path inside HybridDecisionModel reads ModernBERT-large's
            # ~1.6GB of pretrained weights off disk and initializes them --
            # and then load_state_dict() below overwrites every single one
            # with the fine-tuned weights from the checkpoint. That is a
            # second full load of the model, for weights we discard
            # immediately. `backbone_override` already exists for exactly
            # this kind of substitution (smoke_test.py uses it).
            from transformers import AutoConfig, AutoModel
            skeleton = AutoModel.from_config(AutoConfig.from_pretrained(backbone))

            self.trained_k_max = saved_args.get("k_max", 6)
            self.model = HybridDecisionModel(
                backbone=backbone, mask_token_id=self.tokenizer.mask_token_id,
                n_context_codes=saved_args.get("n_context_codes", 16),
                k_max=self.trained_k_max, head_n_layers=saved_args.get("head_n_layers", 2),
                maxsim_dim=saved_args.get("maxsim_dim", 128), use_maxsim=saved_args.get("use_maxsim", False),
                gradient_checkpointing=False, backbone_override=skeleton,
            ).to(self.device)
            # `emb_scale` is a persistent buffer saved in model_state, so the
            # value computed from the randomly-initialized skeleton above is
            # replaced here along with everything else -- verified present in
            # the checkpoint rather than assumed.
            missing, unexpected = self.model.load_state_dict(ckpt["model_state"])
            assert not missing and not unexpected, (missing, unexpected)
            self.model.eval()
            self.builder = PackedSequenceBuilder(
                self.tokenizer, budget_total=saved_args.get("budget_total", 2048),
                l_context=saved_args.get("l_context", 768),
                l_instructions=saved_args.get("l_instructions", 96),
                l_max_per_option=saved_args.get("l_max_per_option", 64),
            )
            meta = {f: ckpt.get(f) for f in ("epoch", "val_acc", "zero_shot_acc")}
            _MODEL_CACHE[key] = (self.model, self.tokenizer, self.builder, saved_args, meta)
            del ckpt

        self.available = True
        self.epoch = meta["epoch"]
        self.val_acc = meta["val_acc"]
        self.zero_shot_acc = meta["zero_shot_acc"]
        self.label = f"Open-One (exp7, epoch {self.epoch})"

        # The recurrent decision block (model.py Sec 2.5) is trained
        # depth-invariant in principle, but only up to however many passes
        # this specific checkpoint's run actually looped during training.
        # Requesting more than that is off-distribution for a block that has
        # never seen its own state fed back that many times -- clamp rather
        # than silently degrade.
        self.trained_k_max = saved_args.get("k_max", 6)

        if self.k is not None and self.k > self.trained_k_max:
            print(f"[open_one_agent] requested k={self.k} exceeds this checkpoint's trained "
                  f"k_max={self.trained_k_max} -- clamping to {self.trained_k_max}. Extra recurrent "
                  f"passes are off-distribution for a block only ever trained that deep.", flush=True)
            self.k = self.trained_k_max

        print(f"[open_one_agent] loaded {ckpt_path} "
              f"(epoch {self.epoch}, val_acc {self.val_acc}, zero_shot_acc {self.zero_shot_acc}, "
              f"trained k_max={self.trained_k_max}, using k={self.k or self.trained_k_max})", flush=True)

    @torch.no_grad()
    def decide(self, game):
        """Returns (Direction, probs_dict) where probs_dict maps every
        Direction to the model's (or heuristic's) confidence, for the game
        to render as a HUD readout."""
        legal = game.legal_directions()

        if not self.available:
            return self._heuristic(game, legal)

        from model import PackedExample

        mode, moves, option_texts = build_request(game)
        if mode != "model":
            # Nothing to decide -- report a flat readout so the HUD shows
            # honestly that this tick was not the model's call.
            probs_by_dir = {d: 0.0 for d in DIRECTION_ORDER}
            probs_by_dir[moves[0]] = 1.0
            return moves[0], probs_by_dir

        ex = PackedExample(
            context=CONTEXT_REQUEST,
            instructions=INSTRUCTIONS,
            option_texts=option_texts,
            qtype="choice",
            answer_idx=0,  # unused at inference; PackedExample requires a value
        )
        with INFERENCE_LOCK:  # see module docstring on INFERENCE_LOCK -- the shared
            # tokenizer is not safe to call from two threads at once.
            batch = self.builder.build_batch([ex], self.device)
            with autocast(device_type=self.device.type, dtype=AUTOCAST_DTYPE, enabled=(self.device.type == "cuda")):
                logits_per_depth = self.model(self.tokenizer, batch, self.device, k=self.k)
        probs = F.softmax(logits_per_depth[-1][0].float(), dim=-1).tolist()

        probs_by_dir = {d: 0.0 for d in DIRECTION_ORDER}
        probs_by_dir.update({d: p for d, p in zip(moves, probs)})
        best = max(moves, key=lambda d: probs_by_dir[d])
        return best, probs_by_dir

    def _heuristic(self, game, legal):
        """Greedy fallback used only when no exp7 checkpoint is loaded:
        picks the legal move that reduces Manhattan distance to the food,
        preferring survival (no immediate death) above that."""
        head = game.snake[0]
        fx, fy = game.food if game.food else head

        def is_safe(d):
            dx, dy = d.value
            nx, ny = head[0] + dx, head[1] + dy
            if nx < 0 or nx >= game.width or ny < 0 or ny >= game.height:
                return False
            if (nx, ny) in game.snake:
                return False
            return True

        def dist_after(d):
            dx, dy = d.value
            nx, ny = head[0] + dx, head[1] + dy
            return abs(fx - nx) + abs(fy - ny)

        safe = [d for d in legal if is_safe(d)]
        pool = safe or legal
        best = min(pool, key=dist_after)

        # Fake a probability readout so the HUD has something to show.
        dists = {d: dist_after(d) for d in DIRECTION_ORDER}
        max_d = max(dists.values()) + 1
        scores = {d: (max_d - dists[d]) for d in DIRECTION_ORDER}
        total = sum(scores.values()) or 1
        probs_by_dir = {d: s / total for d, s in scores.items()}
        return best, probs_by_dir
