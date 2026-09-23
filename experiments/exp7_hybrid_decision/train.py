"""Experiment 7 training loop.

Unlike exp6's dual-encoder model (separate encode_context/encode_outcome/
compatibility_* methods needing a different code path per task), the exp7
architecture answers every task through the SAME packed-sequence mechanism
(model.HybridDecisionModel.forward over model.PackedBatch). So training here
is genuinely unified: every step draws a mixed batch across task types
(intent / mcq / qqp-as-bool / bool_corpus / score_corpus / diversity_corpus),
builds one PackedBatch, and computes one depth-summed proper-scoring-rule
loss (NOTES.md Sec 3). There is no separate qqp/mcq forward pass to weight
against each other the way exp6 needed -- task mixing happens at the
example level, not the loss level.

Missing corpora (bool/score/diversity, built by scripts/build_exp7_data.py)
are handled gracefully: if a corpus isn't on disk yet, its task weight is
zeroed and the remaining weights renormalized, with a loud warning -- so
smoke tests and early partial builds still run.

Staging (NOTES.md Sec 6) is all one script, controlled by flags:
  exp7a: --k_max 1 --use_maxsim false   (fixed depth, MaxSim off)
  exp7b: --k_max 6 --use_maxsim false   (depth-conditioned)
  exp7d: --k_max 6 --use_maxsim true    (MaxSim on, only if 7a's sweep justifies it)
exp7c (learned halting) is a separate follow-on script, not this one -- it
freezes this model and trains two small heads on its depth trajectories.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from model import HybridDecisionModel, PackedSequenceBuilder, PackedExample, QTYPE_IDX, get_tokenizer, BACKBONE
import data as D

AUTOCAST_DTYPE = torch.bfloat16  # matches exp6's own choice -- bf16 has the same exponent range
# as fp32 (unlike fp16), so it needs no loss-scaling machinery at all, hence the _NoOpScaler
# below (kept for symmetry with exp6's code, not because scaling is actually needed here).

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# --------------------------------------------------------------------------
# Loss (NOTES.md Sec 3) -- proper scoring rules, minimized directly (no RL;
# see Sec 3.1 for why that's a deliberate departure from Laya's RLCD/GRPO).
# --------------------------------------------------------------------------

SCORE_QTYPE_IDX = QTYPE_IDX["score"]


def compute_answer_loss(logits: torch.Tensor, answer_idx: torch.Tensor, valid_mask: torch.Tensor,
                         qtype_idx: torch.Tensor, spherical_weight: float = 0.5, rps_weight: float = 1.0):
    """logits: (B, N) with -inf at padding. Returns a scalar loss and a
    dict of the three component means, for logging."""
    log_score = F.cross_entropy(logits, answer_idx)

    p = F.softmax(logits, dim=-1)
    t = F.one_hot(answer_idx, num_classes=logits.size(-1)).to(p.dtype)
    spherical = (t * p).sum(-1) / p.norm(dim=-1).clamp(min=1e-8)
    spherical_loss = (1.0 - spherical).mean()

    is_score = (qtype_idx == SCORE_QTYPE_IDX)
    if is_score.any():
        # Options are laid out in ascending ordinal order for score-type
        # examples (data.build_score_example never shuffles them), and
        # real options are always contiguous at the front (the packed-
        # sequence builder pads invalid slots at the tail) -- both are
        # required for cumsum to mean "ordinal position", not an arbitrary
        # index. cumsum over padding (p~0, t=0 there) is a correct no-op.
        cp = torch.cumsum(p, dim=-1)
        ct = torch.cumsum(t, dim=-1)
        rps_per_example = ((cp - ct) ** 2).sum(-1)
        rps_loss = rps_per_example[is_score].mean()
    else:
        rps_loss = torch.zeros((), device=logits.device)

    total = log_score + spherical_weight * spherical_loss + rps_weight * rps_loss
    return total, {"log_score": log_score.item(), "spherical_loss": spherical_loss.item(),
                    "rps_loss": rps_loss.item() if is_score.any() else 0.0}


def compute_depth_loss(logits_per_depth, answer_idx, valid_mask, qtype_idx):
    """Sec 3.2: uniform average of the answer loss at EVERY depth. Uniform,
    not depth-weighted -- weighting deep passes more would undertrain
    shallow ones and make exp7c's learned halting meaningless (it needs
    every depth to already be a competent standalone answer)."""
    total = 0.0
    comps = {"log_score": 0.0, "spherical_loss": 0.0, "rps_loss": 0.0}
    for logits in logits_per_depth:
        loss, c = compute_answer_loss(logits, answer_idx, valid_mask, qtype_idx)
        total = total + loss
        for k in comps:
            comps[k] += c[k]
    n = len(logits_per_depth)
    for k in comps:
        comps[k] /= n
    return total / n, comps


# --------------------------------------------------------------------------
# Unified task mixer
# --------------------------------------------------------------------------

class TaskMixer:
    """Draws one PackedExample per slot, task chosen by weighted random
    choice each time (not fixed per-batch counts) -- this is what makes
    every step a genuinely mixed batch, matching the unified architecture's
    single training mechanism (see this file's module docstring)."""

    def __init__(self, seed: int = 12345):
        self.rng = random.Random(seed)
        self.pools = {}     # task_name -> (items, builder_fn)
        self.weights = {}   # task_name -> float

    def register(self, name: str, items: list, weight: float, builder):
        if not items or weight <= 0:
            return
        self.pools[name] = (items, builder)
        self.weights[name] = weight

    def finalize(self):
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("TaskMixer has no active tasks with positive weight -- "
                              "check that at least one corpus loaded and its weight > 0.")
        self.names = list(self.weights.keys())
        self.probs = [self.weights[n] / total for n in self.names]
        print(f"TaskMixer active tasks (normalized weights): "
              + ", ".join(f"{n}={p:.3f}" for n, p in zip(self.names, self.probs)), flush=True)

    def sample_batch(self, batch_size: int, extra_rng: random.Random = None) -> list:
        rng = extra_rng or self.rng
        examples = []
        for _ in range(batch_size):
            name = rng.choices(self.names, weights=self.probs, k=1)[0]
            items, builder = self.pools[name]
            item = rng.choice(items)
            examples.append(builder(item, rng))
        return examples


def _diversity_builder(diversity_pool_all_labels):
    def build(row, rng):
        return D.build_diversity_example(row, diversity_pool_all_labels, rng)
    return build


def _mcq_builder(all_option_texts):
    def build(ex, rng):
        return D.build_mcq_example(ex, all_option_texts, rng)
    return build


def build_mixer(args, intent_train, mcq_train, qqp_train, bool_train, score_train, diversity_train):
    mixer = TaskMixer(seed=args.mixer_seed)

    mixer.register("intent", intent_train, args.w_intent,
                    lambda item, rng: D.build_intent_example(item[0], item[1], _INTENT_SEEN_LABELS[0], rng))

    if mcq_train:
        all_mcq_options = [opt for ex in mcq_train for opt in ex["options"]]
        mixer.register("mcq", mcq_train, args.w_mcq, _mcq_builder(all_mcq_options))

    qqp_bool_items = [("qqp", t1, t2, bool(lab)) for t1, t2, lab in qqp_train] if qqp_train else []
    bool_items = [("bool", r) for r in bool_train] if bool_train else []
    if qqp_bool_items or bool_items:
        combined = qqp_bool_items + bool_items

        def build_bool_mixed(item, rng):
            if item[0] == "qqp":
                _, t1, t2, lab = item
                return D.build_qqp_example(t1, t2, lab, rng)
            return D.build_bool_example(item[1], rng)

        mixer.register("bool", combined, args.w_bool, build_bool_mixed)
    else:
        print("WARNING: no bool-type data available (qqp_paraphrase_pairs missing AND "
              "exp7_bool_corpus not built) -- bool-question training signal is INACTIVE this run.",
              flush=True)

    if score_train:
        mixer.register("score", score_train, args.w_score, lambda r, rng: D.build_score_example(r, rng))
    else:
        print("WARNING: exp7_score_corpus not built -- score-question (RPS) training signal "
              "is INACTIVE this run. Build it with scripts/build_exp7_data.py first.", flush=True)

    if diversity_train:
        all_diversity_labels = list({l for row in diversity_train for l in row["labels"]})
        mixer.register("diversity", diversity_train, args.w_diversity, _diversity_builder(all_diversity_labels))
    else:
        print("WARNING: exp7_diversity_corpus not built -- label-vocabulary-diversity training "
              "signal is INACTIVE this run (this is the single biggest lever for zero-shot "
              "generalization per NOTES.md Sec 5.2 -- build it before trusting a full run).",
              flush=True)

    mixer.finalize()
    return mixer


# Set once in main() -- a small hack to keep build_mixer's lambda simple
# without threading seen_labels through every call site.
_INTENT_SEEN_LABELS = [None]


# --------------------------------------------------------------------------
# Param groups: layer-wise LR decay over the (single, weight-tied) backbone,
# a separate higher LR for the head -- same convention as exp5/exp6's
# build_param_groups, adapted to exp7's single-backbone architecture
# (NOTES.md Sec 4.2 -- forgetting control via decay, not freezing; exp4's
# frozen-backbone result is why freezing is off the table here).
# --------------------------------------------------------------------------

def build_param_groups(model: HybridDecisionModel, base_lr: float, head_lr_mult: float = 5.0,
                        decay: float = 0.9, gate_lr_mult: float = 0.0):
    """Layer-wise-decayed backbone groups + one head group, same convention
    as exp5/exp6. Also correct with a fully frozen backbone (--freeze_backbone,
    the recursion/head-only continuation phase): trainable_layers/
    other_backbone_trainable both come out empty, requires_grad already
    filters them everywhere, and the optimizer only ever sees non-empty
    groups (some optimizers, and bitsandbytes' 8-bit AdamW in particular,
    are worth not trusting with a zero-param group).

    gate_lr_mult > 0 carves inject_gate/maxsim_gate out of head_params into
    their OWN group at base_lr * gate_lr_mult. Why this exists: exp7a's run
    showed inject_gate climbing monotonically but by only ~0.0012/epoch
    (0.0096 -> 0.0215 over 13 epochs) at the head's ordinary 5x LR -- at
    that rate it needs 100+ epochs to reach a magnitude where the injected
    vector is competitive with a normal token embedding, which the
    cardinality-sweep ablation confirmed it never did (vector_only scored
    at chance at every N). A single scalar can tolerate a much higher LR
    than a weight matrix without the usual stability risk (no fan-in/fan-out
    to destabilize, no risk of blowing up a whole layer's activations) --
    this exists specifically to find out whether the vector path was
    unlearnable or just moving too slowly to observe within 13 epochs.
    """
    groups = []
    trainable_layers = [layer for layer in model.backbone.layers
                         if any(p.requires_grad for p in layer.parameters())]
    n = len(trainable_layers)
    for i, layer in enumerate(trainable_layers):
        lr = base_lr * (decay ** (n - 1 - i))
        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr})

    # ALL backbone params, not just .layers -- a real bug, found while adding
    # --freeze_backbone: the token embedding table and final norm live on
    # model.backbone but outside model.backbone.layers, so the previous
    # version of this id-set silently left them out of the exclusion set and
    # they fell into head_params below, getting the head's 5x LR instead of
    # a backbone-appropriate one on every full-fine-tune run to date
    # (exp7a included). Given the base (bottom-layer-equivalent) LR here,
    # not the head's -- they're backbone components, not head components.
    backbone_params = set(id(p) for p in model.backbone.parameters())
    layer_prefixes = tuple(f"layers.{i}." for i in range(len(model.backbone.layers)))
    other_backbone_trainable = [p for name, p in model.backbone.named_parameters()
                                 if p.requires_grad and not name.startswith(layer_prefixes)]
    if other_backbone_trainable:
        groups.append({"params": other_backbone_trainable, "lr": base_lr})

    gate_ids = {id(model.inject_gate), id(model.maxsim_gate)} if gate_lr_mult > 0 else set()
    gate_params = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
    head_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in backbone_params and id(p) not in gate_ids]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * head_lr_mult})
    if gate_params:
        groups.append({"params": gate_params, "lr": base_lr * gate_lr_mult})
    return groups


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------

def run_eval(model, builder: PackedSequenceBuilder, tokenizer, device, examples: list,
             batch_size: int = 8, k: int = None, option_chunk_size: int = 128):
    """Returns a list of per-depth accuracies (Sec 7's depth curve is
    exactly this, aggregated across epochs)."""
    model.eval()
    k = k or model.k_max
    correct = [0] * k
    total = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i:i + batch_size]
            batch = builder.build_batch(batch_ex, device)
            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                logits_per_depth = model(tokenizer, batch, device, k=k, option_chunk_size=option_chunk_size)
            for d, logits in enumerate(logits_per_depth):
                preds = logits.argmax(dim=-1)
                correct[d] += (preds == batch.answer_idx).sum().item()
            total += len(batch_ex)
    model.train()
    return [c / total for c in correct]


def eval_gate_values(model):
    return {"inject_gate": model.inject_gate.item(), "maxsim_gate": model.maxsim_gate.item()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # architecture / staging
    ap.add_argument("--backbone", type=str, default=BACKBONE)
    ap.add_argument("--k_max", type=int, default=6, help="exp7a=1, exp7b/7d=6.")
    ap.add_argument("--freeze_backbone", action="store_true",
                     help="Recursion/head-only continuation phase: freeze the backbone AFTER "
                          "loading --resume's weights (must be combined with --resume -- freezing "
                          "a randomly-initialized or merely-pretrained backbone is the exact "
                          "frozen-from-the-start regime that failed badly in Experiment 4, see "
                          "PROJECT_HISTORY.md; this is only sound once the backbone has already "
                          "been fully fine-tuned by a prior exp7a run). Trains only the entry "
                          "layer, recurrent block, scorer, context codes, and injection/MaxSim "
                          "components -- cheap enough to run --k_max 6 depth-conditioned training "
                          "without paying for backbone backward passes at all.")
    ap.add_argument("--reset_training_state", action="store_true",
                     help="With --resume, load ONLY the model weights -- ignore the checkpoint's "
                          "optimizer state / epoch counter / best-metric tracking and start those "
                          "fresh. Correct the first time you branch a new training phase off an "
                          "existing checkpoint (its optimizer state belongs to a different set of "
                          "param groups). Do NOT pass this on an ordinary relaunch-after-disconnect "
                          "of a run already in progress -- that should keep resuming its own state; "
                          "only --freeze_backbone needs re-applying every process start, since "
                          "requires_grad isn't part of a saved state_dict.")
    ap.add_argument("--use_maxsim", action="store_true", help="exp7d only -- see NOTES.md Sec 6/7.")
    ap.add_argument("--allow_arch_mismatch", action="store_true",
                     help="--resume loads with strict=False and prints missing/unexpected keys "
                          "instead of crashing. For exp7c: recurrent_block switched from "
                          "TransformerEncoderLayer to TransformerDecoderLayer (adds cross-attention "
                          "over the full backbone output, to fix exp7b's flat depth curve) and a "
                          "new depth_embed was added -- neither has any matching keys in an exp7a/7b "
                          "checkpoint, so those parts load fresh while backbone/entry_layer/scorer/ "
                          "context_codes/option_projector/inject_gate resume as before. Leave off "
                          "for an ordinary same-architecture resume -- it should stay strict so a "
                          "real bug (a typo'd shape, a genuinely missing weight) still crashes "
                          "loudly instead of silently training from a partial init.")
    ap.add_argument("--n_context_codes", type=int, default=16)
    ap.add_argument("--head_n_layers", type=int, default=2)
    ap.add_argument("--maxsim_dim", type=int, default=128)
    ap.add_argument("--no_grad_checkpoint", action="store_true")

    # packed-sequence budget (NOTES.md Sec 2.2)
    ap.add_argument("--budget_total", type=int, default=2048)
    ap.add_argument("--l_context", type=int, default=768)
    ap.add_argument("--l_instructions", type=int, default=96)
    ap.add_argument("--l_max_per_option", type=int, default=64)
    ap.add_argument("--option_chunk_size", type=int, default=128)
    ap.add_argument("--maxsim_chunk_size", type=int, default=32)

    # task weights (renormalized over whatever corpora actually loaded)
    ap.add_argument("--w_intent", type=float, default=0.30)
    ap.add_argument("--w_mcq", type=float, default=0.30)
    ap.add_argument("--w_bool", type=float, default=0.10)
    ap.add_argument("--w_score", type=float, default=0.10)
    ap.add_argument("--w_diversity", type=float, default=0.20)
    ap.add_argument("--mixer_seed", type=int, default=12345)

    # optimization
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=500,
                     help="Mixed-task training has no natural 'one epoch through the data' "
                          "boundary (every task pool is resampled every step) -- an epoch here "
                          "is just this many optimizer steps, a checkpoint/eval cadence, not a "
                          "claim about data coverage.")
    ap.add_argument("--batch_size", type=int, default=8,
                     help="Start LOW and raise once actual GPU memory usage is observed (see "
                          "colab_exp7_watch.sh) -- packed sequences here are up to ~2048 tokens "
                          "(vs. exp6's 32-384 token sequences), so batch size has to be "
                          "recalibrated from scratch, the same way exp6 staged 48->160->384.")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--base_lr", type=float, default=1e-5)
    ap.add_argument("--layer_decay", type=float, default=0.9)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--gate_lr_mult", type=float, default=0.0,
                     help="If > 0, inject_gate/maxsim_gate get their OWN param group at "
                          "base_lr * gate_lr_mult instead of the ordinary head group. exp7a's "
                          "gate rose only 0.0096->0.0215 over 13 epochs at the head's usual LR -- "
                          "too slow to reach a magnitude where the sweep's vector_only ablation "
                          "(which scored at chance every N) could tell 'unlearnable' apart from "
                          "'still ramping'. Try something like 20-50 for the frozen-recursion phase.")
    ap.add_argument("--gate_init", type=float, default=None,
                     help="With --resume, overwrites the checkpoint's inject_gate value right "
                          "after loading (does NOT touch maxsim_gate or anything else) -- a "
                          "one-time head start on top of --gate_lr_mult's faster ongoing climb, "
                          "rather than resuming from exp7a's 0.0215 and waiting for the higher LR "
                          "to catch it up. Try something like 0.05-0.1 (still far below a typical "
                          "token embedding's norm, so this isn't force-feeding the vector path --"
                          "it just starts the race further down the track).")
    ap.add_argument("--patience", type=int, default=6,
                     help="Early stopping is keyed to ZERO-SHOT accuracy, not val_acc -- see "
                          "NOTES.md Sec 4.2 and exp6's own open thread on this.")
    ap.add_argument("--warmup_frac", type=float, default=0.05)

    # eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--zero_shot_n", type=int, default=300)
    ap.add_argument("--banking77_n", type=int, default=400)
    ap.add_argument("--eval_batch_size", type=int, default=8)

    # infra
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--ckpt_prefix", type=str, default=None,
                     help="Checkpoint/metrics filename prefix (default 'exp7', or 'exp7_frozen' "
                          "automatically under --freeze_backbone) -- distinct prefixes let "
                          "different stages (exp7a / a frozen-backbone continuation / exp7b) "
                          "share one --ckpt_dir without overwriting each other's files.")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_run_id", type=str, default="exp7")
    ap.add_argument("--max_train_intent", type=int, default=None, help="Smoke-test cap.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bitsandbytes available: {HAS_BNB}", flush=True)

    use_wandb = args.wandb_project is not None and HAS_WANDB
    if args.wandb_project and not HAS_WANDB:
        print("--wandb_project given but wandb isn't installed; skipping.", flush=True)
    if use_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow", config=vars(args))

    # ---- data ----
    intent = D.load_intent_corpus_minus_banking77()
    _INTENT_SEEN_LABELS[0] = intent.seen_labels
    intent_train = intent.train[:args.max_train_intent] if args.max_train_intent else intent.train
    print(f"intent_corpus (Banking77 held out): train {len(intent_train)}  val {len(intent.val)}  "
          f"seen_labels {len(intent.seen_labels)}  zero_shot_labels {len(intent.zero_shot_labels)}  "
          f"banking77_holdout {len(intent.banking77_holdout)} ({len(intent.banking77_labels)} labels)",
          flush=True)

    mcq = D.load_mcq_corpus_reweighted()
    print(f"mcq_corpus (HellaSwag downweighted to ~{D.HELLASWAG_TARGET_FRACTION:.0%}): "
          f"train {len(mcq['train'])}  val {len(mcq['val'])}", flush=True)

    try:
        qqp_train, qqp_val = D.load_qqp_pairs()
    except FileNotFoundError:
        qqp_train, qqp_val = [], []
        print("WARNING: qqp_paraphrase_pairs not found on disk.", flush=True)

    def _try_load(fn, name):
        try:
            d = fn()
            print(f"{name}: train {len(d['train'])}  val {len(d['val'])}  test {len(d['test'])}", flush=True)
            return d
        except FileNotFoundError:
            print(f"{name} not found on disk -- run scripts/build_exp7_data.py to build it.", flush=True)
            return None

    bool_corpus = _try_load(D.load_bool_corpus, "exp7_bool_corpus")
    score_corpus = _try_load(D.load_score_corpus, "exp7_score_corpus")
    diversity_corpus = _try_load(D.load_diversity_corpus, "exp7_diversity_corpus")

    mixer = build_mixer(
        args, intent_train, mcq["train"], qqp_train,
        bool_corpus["train"] if bool_corpus else [],
        score_corpus["train"] if score_corpus else [],
        diversity_corpus["train"] if diversity_corpus else [],
    )

    # ---- model ----
    tokenizer = get_tokenizer(args.backbone)
    model = HybridDecisionModel(
        backbone=args.backbone, mask_token_id=tokenizer.mask_token_id,
        n_context_codes=args.n_context_codes, k_max=args.k_max, head_n_layers=args.head_n_layers,
        maxsim_dim=args.maxsim_dim, use_maxsim=args.use_maxsim,
        # NOT gated on --freeze_backbone: the packed-sequence pass still
        # needs a full backward graph through the (frozen) backbone,
        # because its input embeddings carry gradient from the trainable
        # injection path -- see HybridDecisionModel.freeze_backbone()'s
        # docstring for the correction. Only --no_grad_checkpoint controls
        # this now.
        gradient_checkpointing=not args.no_grad_checkpoint,
    ).to(device)
    builder = PackedSequenceBuilder(tokenizer, budget_total=args.budget_total, l_context=args.l_context,
                                     l_instructions=args.l_instructions, l_max_per_option=args.l_max_per_option)

    if args.freeze_backbone and not args.resume:
        raise ValueError("--freeze_backbone requires --resume pointing at an already fully "
                          "fine-tuned checkpoint (e.g. exp7a's) -- freezing an un-adapted backbone "
                          "is the frozen-from-the-start regime Experiment 4 already showed fails "
                          "badly. See train.py's --freeze_backbone help.")

    resume_ckpt = None
    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device)
        if args.allow_arch_mismatch:
            missing, unexpected = model.load_state_dict(resume_ckpt["model_state"], strict=False)
            print(f"--allow_arch_mismatch: {len(missing)} missing keys (fresh init), "
                  f"{len(unexpected)} unexpected keys (dropped) -- ", flush=True)
            if missing:
                print(f"  missing: {sorted(set(k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k for k in missing))}", flush=True)
            if unexpected:
                print(f"  unexpected: {sorted(set(k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k for k in unexpected))}", flush=True)
        else:
            model.load_state_dict(resume_ckpt["model_state"])
        print(f"Loaded weights from {args.resume} (epoch {resume_ckpt['epoch']}, "
              f"val_acc {resume_ckpt['val_acc']:.4f})", flush=True)

        # freeze_backbone() is applied unconditionally whenever --freeze_backbone
        # is passed, regardless of whether the rest of this checkpoint's training
        # state gets restored below -- requires_grad is NOT part of a saved
        # state_dict, so every process start has to re-apply it itself, including
        # a relaunch-after-disconnect that's continuing THIS SAME frozen phase
        # (not just the first launch that branches off an exp7a checkpoint). An
        # earlier version of this code only froze on the branch-off launch and
        # silently let the backbone become trainable again on any later relaunch
        # of the same frozen run -- caught before ever being deployed.
        if args.freeze_backbone:
            model.freeze_backbone()
            print("Backbone frozen -- training recursion/head components only.", flush=True)

        if args.gate_init is not None:
            old_gate = model.inject_gate.item()
            with torch.no_grad():
                model.inject_gate.fill_(args.gate_init)
            print(f"--gate_init: overwrote inject_gate {old_gate:.4f} -> {args.gate_init:.4f} "
                  f"(one-time head start; maxsim_gate untouched)", flush=True)

        if args.reset_training_state:
            print("--reset_training_state: treating this as a FRESH run (fresh epoch counter, "
                  "fresh optimizer, fresh best-metric tracking) despite --resume -- correct when "
                  "branching a new phase off a checkpoint whose optimizer state doesn't correspond "
                  "to this run's param groups at all (e.g. the first --freeze_backbone launch off "
                  "an exp7a checkpoint). Do NOT pass this on a plain relaunch-after-disconnect of "
                  "an already-in-progress run -- that should resume its own epoch/optimizer state.",
                  flush=True)
            resume_ckpt = None  # weights are loaded; nothing else below should be resumed from it

    # Printed here, not right after construction -- freeze_backbone() (if
    # applied above) changes requires_grad on ~395M params, and the count
    # is only meaningful once that's already happened.
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,} "
          f"({100 * model.num_trainable_params() / model.num_params():.1f}%)", flush=True)

    param_groups = build_param_groups(model, args.base_lr, args.head_lr_mult, args.layer_decay,
                                       args.gate_lr_mult)
    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)
        print("Using bitsandbytes 8-bit AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_opt_steps = math.ceil(args.steps_per_epoch / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_opt_steps * args.warmup_frac))

    def lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    resume_opt_step = 0
    start_epoch = 1
    best_val = -1.0
    best_zero_shot = -1.0
    no_improve = 0
    if resume_ckpt is not None:
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            resume_opt_step = resume_ckpt.get("opt_step", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt.get("val_acc", -1.0)
        best_zero_shot = resume_ckpt.get("zero_shot_acc", -1.0)
        no_improve = resume_ckpt.get("no_improve", 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale, last_epoch=resume_opt_step - 1)
    scaler = _NoOpScaler()

    ckpt_dir = args.ckpt_dir or os.path.dirname(__file__)
    ckpt_prefix = args.ckpt_prefix or ("exp7_frozen" if args.freeze_backbone else "exp7")
    os.makedirs(ckpt_dir, exist_ok=True)

    intent_val_subset = intent.val[:args.val_n]
    zero_shot_val_subset = intent.test_zero_shot[:args.zero_shot_n]
    banking77_subset = intent.banking77_holdout[:args.banking77_n]

    def eval_intent_like(examples, label_pool, seed_base):
        rng = random.Random(seed_base)
        packed = [D.build_intent_example(t, l, label_pool, rng, n_target=50) for t, l in examples]
        return run_eval(model, builder, tokenizer, device, packed, batch_size=args.eval_batch_size,
                         option_chunk_size=args.option_chunk_size)

    print(f"\n=== Training from epoch {start_epoch}, best_val={best_val:.4f}, "
          f"best_zero_shot={best_zero_shot:.4f} ===", flush=True)

    opt_step = resume_opt_step
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        running_comps = {"log_score": 0.0, "spherical_loss": 0.0, "rps_loss": 0.0}

        for step in range(args.steps_per_epoch):
            examples = mixer.sample_batch(args.batch_size)
            batch = builder.build_batch(examples, device)
            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                logits_per_depth = model(tokenizer, batch, device, k=args.k_max,
                                          option_chunk_size=args.option_chunk_size,
                                          maxsim_chunk_size=args.maxsim_chunk_size)
            # Loss deliberately computed OUTSIDE the autocast block: softmax/
            # log_softmax/cross_entropy are numerically sensitive and
            # autocast already keeps them in fp32 internally when invoked
            # from within its context, but the logits themselves come out as
            # bf16 tensors from the block above -- upcasting explicitly here
            # (rather than relying on autocast's op-level casting alone)
            # keeps the RPS cumsum-of-squared-differences term, which has no
            # autocast-registered special case, numerically stable.
            logits_per_depth = [l.float() for l in logits_per_depth]
            loss, comps = compute_depth_loss(logits_per_depth, batch.answer_idx, batch.valid_mask,
                                              batch.qtype_idx)
            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            for k_ in comps:
                running_comps[k_] += comps[k_]

            is_last = (step + 1) == args.steps_per_epoch
            if (step + 1) % args.grad_accum == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if opt_step < total_opt_steps:
                    scheduler.step()
                opt_step += 1

            elapsed = time.time() - t0
            done = step + 1
            avg = elapsed / done
            eta = avg * (args.steps_per_epoch - done)
            mem_str = ""
            if device.type == "cuda":
                mem_str = f"  gpu_mem {torch.cuda.memory_allocated() / (1024 ** 3):.1f}GiB"
            print(f"  epoch {epoch}/{args.epochs}  step {done}/{args.steps_per_epoch} "
                  f"({100 * done / args.steps_per_epoch:.0f}%)  loss {loss.item():.4f}  "
                  f"step_time {avg:.2f}s  ETA {eta:.0f}s{mem_str}", flush=True)

        dt = time.time() - t0
        avg_loss = running_loss / args.steps_per_epoch
        for k_ in running_comps:
            running_comps[k_] /= args.steps_per_epoch

        do_eval = (epoch % args.eval_every == 0)
        if do_eval:
            val_curve = eval_intent_like(intent_val_subset, intent.seen_labels, seed_base=9000 + epoch)
            zs_curve = eval_intent_like(zero_shot_val_subset, intent.all_labels, seed_base=9500 + epoch)
            bank_curve = eval_intent_like(banking77_subset, intent.banking77_labels, seed_base=9900 + epoch)
            val_acc, zero_shot_acc, banking77_acc = val_curve[-1], zs_curve[-1], bank_curve[-1]
            gates = eval_gate_values(model)
            cur_lr = scheduler.get_last_lr()[-1]

            print(f"epoch {epoch}/{args.epochs}  loss {avg_loss:.4f} "
                  f"(log {running_comps['log_score']:.4f} sph {running_comps['spherical_loss']:.4f} "
                  f"rps {running_comps['rps_loss']:.4f})  "
                  f"val_acc@final {val_acc:.4f}  zero_shot_acc@final {zero_shot_acc:.4f}  "
                  f"banking77_holdout_acc@final {banking77_acc:.4f}  "
                  f"inject_gate {gates['inject_gate']:.4f}  maxsim_gate {gates['maxsim_gate']:.4f}  "
                  f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)
            print(f"  depth curve  val: {['%.4f' % a for a in val_curve]}  "
                  f"zero_shot: {['%.4f' % a for a in zs_curve]}  "
                  f"banking77: {['%.4f' % a for a in bank_curve]}", flush=True)

            with open(os.path.join(ckpt_dir, f"{ckpt_prefix}_metrics.jsonl"), "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "zero_shot_acc": zero_shot_acc,
                    "banking77_holdout_acc": banking77_acc, "val_curve": val_curve, "zero_shot_curve": zs_curve,
                    "banking77_curve": bank_curve, "inject_gate": gates["inject_gate"],
                    "maxsim_gate": gates["maxsim_gate"], "lr": cur_lr, "epoch_seconds": dt,
                    "timestamp": time.time(),
                }) + "\n")
            if use_wandb:
                wandb.log({"loss": avg_loss, "val_acc": val_acc, "zero_shot_acc": zero_shot_acc,
                           "banking77_holdout_acc": banking77_acc, **gates, "lr": cur_lr,
                           "epoch_seconds": dt, **{f"loss/{k_}": v for k_, v in running_comps.items()}},
                          step=epoch)

            def make_ckpt():
                return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "opt_step": opt_step, "epoch": epoch, "val_acc": val_acc,
                        "zero_shot_acc": zero_shot_acc, "banking77_holdout_acc": banking77_acc,
                        "no_improve": no_improve, "args": vars(args)}

            torch.save(make_ckpt(), os.path.join(ckpt_dir, f"{ckpt_prefix}_latest.pt"))

            if zero_shot_acc > best_zero_shot:
                best_zero_shot = zero_shot_acc
                torch.save(make_ckpt(), os.path.join(ckpt_dir, f"{ckpt_prefix}_best_zeroshot.pt"))
                print(f"  -> new best zero_shot_acc {zero_shot_acc:.4f}", flush=True)

            # Early stopping keyed to ZERO-SHOT, not val_acc (NOTES.md Sec 4.2).
            if val_acc > best_val:
                best_val = val_acc
                torch.save(make_ckpt(), os.path.join(ckpt_dir, f"{ckpt_prefix}_best_val.pt"))
            if zero_shot_acc >= best_zero_shot - 1e-9:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping: zero_shot_acc hasn't improved for {args.patience} evals.",
                          flush=True)
                    break
        else:
            print(f"epoch {epoch}/{args.epochs}  loss {avg_loss:.4f}  (eval skipped this epoch)  ({dt:.1f}s)",
                  flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
