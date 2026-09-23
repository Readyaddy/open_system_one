"""Experiment 6 training: Qwen2.5-0.5B, fully unfrozen (validated in
experiment 5 to fit and run at reasonable speed), on the diverse 5-source
local data corpus, with a QQP paraphrase-pair auxiliary objective mixed in
every step.

Two things changed vs. experiment 5, both targeting the same diagnosed
weakness (narrow data -> narrow/template-bound paraphrase invariance):
  1. Data: data/intent_corpus/ (5 independent sources, 300 intents,
     44.4k train examples, deduplicated and label-collision-fixed -- see
     data/README.md) instead of the old 3-source, 234-intent corpus.
  2. QQP auxiliary loss: every training step also samples a batch of real
     human-written QQP paraphrase pairs and computes a binary
     compatibility loss on them (model.compatibility_pairwise), through
     the SAME encoder weights the main task uses -- this is meant to
     teach genuine paraphrase invariance directly, since the main task's
     only other source of "same meaning, different words" pressure is 8
     fixed templates (dataset_v7.sample_outcome_description).

All data loading is local (src/local_data.py reads data/*.jsonl) -- no
network calls at train time except the one-time backbone checkpoint
download.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # local exp6 folder takes priority (e.g. its own model.py)

import argparse
import math
import time
import random
import torch
import torch.nn as nn
from torch.amp import autocast

from local_data import (load_intent_corpus, load_qqp_pairs, load_mcq_corpus,
                         sample_outcome_description, base_description, raw_intent_name)
from model import JEPAPolyEncoderV4, get_tokenizer

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

AUTOCAST_DTYPE = torch.bfloat16


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def tokenize(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


OUTCOME_CHUNK_SIZE = 64  # overwritten in main() from --outcome_chunk_size


def encode_outcome_bank(model, tokenizer, texts, device, chunk_size=None):
    """Chunked outcome-bank encoding -- see experiment 4's NOTES.md for
    why chunking exists at all (a single-shot forward pass over ~250
    candidates was the actual OOM cause there, not batch size). On a GPU
    with memory to spare, fewer/bigger chunks means fewer sequential
    round-trips (each chunk is a separate kernel-launch-heavy step) --
    chunk_size is tunable via --outcome_chunk_size rather than fixed,
    since the right value depends on how much headroom the GPU has."""
    if chunk_size is None:
        chunk_size = OUTCOME_CHUNK_SIZE
    embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        otok, omask = tokenize(tokenizer, chunk, device)
        with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
            embeddings.append(model.encode_outcome(otok, omask))
    return torch.cat(embeddings, dim=0)


MCQ_CONTEXT_MAX_LENGTH = 384  # real passages (RACE/SciQ), not the short 32-token
# utterances used everywhere else -- truncates the longest RACE passages (a
# known limitation: no passage-retrieval/sentence-selection step exists to
# shorten a passage down to just its relevant span before truncation).


def mcq_context_text(ex):
    """context + question when context is non-empty (real passage,
    RACE/SciQ), else just the bare question (the context-free sources)."""
    return f"{ex['context']}\n\nQuestion: {ex['question']}" if ex["context"] else ex["question"]


def routing_question_text(utterance):
    """Same context+question shape as mcq_context_text: the customer
    utterance IS the context, paired with an explicit routing question --
    instead of scoring the bare utterance directly against candidate
    descriptions, this matches the exact input shape the MCQ task uses
    everywhere else, so the intent task isn't secretly a different kind of
    input just because it's phrased as classification historically."""
    return f"{utterance}\n\nQuestion: Which category best describes what this customer wants, or where should this request be routed?"


def encode_mcq_batch(model, tokenizer, batch, device):
    """Each MCQ example carries its OWN candidate set (unlike the intent
    task's one shared bank), so batch encoding needs padding: encode every
    real option once (flat, no wasted padding tokens), then scatter into a
    (B, max_k, D) tensor with a (B, max_k) validity mask for
    model.compatibility_grouped to mask out padding slots."""
    option_lists = [ex["options"] for ex in batch]
    max_k = max(len(o) for o in option_lists)
    flat_texts = [opt for opts in option_lists for opt in opts]
    otok, omask = tokenize(tokenizer, flat_texts, device)
    with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
        flat_emb = model.encode_outcome(otok, omask)
    d = flat_emb.size(-1)
    B = len(batch)
    outcome_emb = torch.zeros(B, max_k, d, device=device, dtype=flat_emb.dtype)
    valid_mask = torch.zeros(B, max_k, dtype=torch.bool, device=device)
    pos = 0
    for i, opts in enumerate(option_lists):
        k = len(opts)
        outcome_emb[i, :k] = flat_emb[pos:pos + k]
        valid_mask[i, :k] = True
        pos += k
    answer_idx = torch.tensor([ex["answer_idx"] for ex in batch], device=device)
    return outcome_emb, valid_mask, answer_idx


def build_param_groups(model: JEPAPolyEncoderV4, base_lr: float, decay: float = 0.9):
    groups = []
    head_params = []
    for encoder in [model.context_encoder, model.outcome_encoder]:
        trainable_layers = [layer for layer in encoder.backbone.layers if
                             any(p.requires_grad for p in layer.parameters())]
        n = len(trainable_layers)
        for i, layer in enumerate(trainable_layers):
            lr = base_lr * (decay ** (n - 1 - i))
            groups.append({"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr})
        head_params += [p for n_, p in encoder.named_parameters()
                         if p.requires_grad and "backbone.layers" not in n_]
    head_params.append(model.log_temperature)
    groups.append({"params": head_params, "lr": base_lr * 5})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--qqp_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--freeze_layers", type=int, default=0)  # fully unfrozen, validated in exp5
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--qqp_weight", type=float, default=0.3)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--no_grad_checkpoint", action="store_true",
                         help="Disable gradient checkpointing -- it trades compute for memory "
                              "(recomputes activations during backward instead of storing them). "
                              "Worth disabling on a GPU with memory to spare (e.g. A100 40GB), "
                              "since it's pure overhead once you're not memory-constrained.")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from (model + optimizer + schedule position).")
    parser.add_argument("--outcome_chunk_size", type=int, default=64,
                         help="Chunk size for outcome-bank encoding. Bigger = fewer sequential "
                              "round-trips, at the cost of more peak memory per chunk.")
    parser.add_argument("--qqp_every_n_steps", type=int, default=1,
                         help="Only run the QQP auxiliary step every N micro-steps instead of "
                              "every step. >1 trades a less-dense auxiliary signal for speed.")
    parser.add_argument("--mcq_batch_size", type=int, default=12)
    parser.add_argument("--mcq_weight", type=float, default=0.5,
                         help="MCQ (answer-selection-from-options) is the actual target "
                              "capability, not just a regularizer like QQP -- weighted higher "
                              "than qqp_weight accordingly.")
    parser.add_argument("--mcq_every_n_steps", type=int, default=1)
    parser.add_argument("--mcq_eval_n", type=int, default=800,
                         help="Size of the mcq_val subset used for per-epoch eval (speed vs. "
                              "precision tradeoff, same idea as the zero-shot eval subset).")
    parser.add_argument("--intent_options", type=int, default=50,
                         help="Total options (1 correct + N-1 distractors) per intent example, "
                              "for BOTH training and eval -- must be the same task shape in both "
                              "or the eval number doesn't measure what training optimizes. "
                              "Bumped from an initial 10 (too easy/narrow, plausibly why zero-shot "
                              "kept declining even with this dynamic-candidate mechanism) toward "
                              "the old full-255-bank's difficulty without fully reverting to it.")
    parser.add_argument("--wandb_project", type=str, default=None,
                         help="If set (and wandb is installed), logs live metrics to this W&B "
                              "project -- a real-time web dashboard that survives session drops, "
                              "instead of only a local/Drive log file that needs manual polling.")
    parser.add_argument("--wandb_run_id", type=str, default="exp6",
                         help="Fixed W&B run id so relaunching after a disconnect RESUMES the same "
                              "run's chart (same id every time) instead of starting a new one each "
                              "time this script restarts.")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                         help="Where to save checkpoints. Defaults to this script's own directory "
                              "-- on an ephemeral cloud VM, point this at a mounted persistent "
                              "location (e.g. Google Drive) instead, so checkpoints survive if the "
                              "session disconnects.")
    parser.add_argument("--compile", action="store_true",
                         help="Wrap the model in torch.compile(). Can meaningfully cut Python/"
                              "framework overhead, but may fail or recompile repeatedly on "
                              "variable-length inputs -- verify with a smoke test first.")
    args = parser.parse_args()

    global OUTCOME_CHUNK_SIZE
    OUTCOME_CHUNK_SIZE = args.outcome_chunk_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bitsandbytes available: {HAS_BNB}", flush=True)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        if not HAS_WANDB:
            print("--wandb_project given but wandb isn't installed; skipping W&B logging.", flush=True)
            use_wandb = False
        else:
            # Fixed id + resume="allow": relaunching this script after a
            # disconnect continues the SAME run's chart instead of starting
            # a new one each restart -- since restarts have been frequent
            # today, a fresh run per restart would fragment the history
            # across many unrelated tiny charts.
            wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow",
                       config=vars(args))
            print(f"W&B logging enabled: project={args.wandb_project} run_id={args.wandb_run_id}",
                  flush=True)

    data = load_intent_corpus()
    train_ex, val_ex = data["train"], data["val"]
    test_ex, test_oos_ex, test_zs_ex = data["test"], data["test_oos"], data["test_zero_shot"]
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])
    if args.max_train:
        train_ex = train_ex[:args.max_train]
    if args.max_val:
        val_ex = val_ex[:args.max_val]
    seen_idx = {l: i for i, l in enumerate(seen_labels)}
    all_idx = {l: i for i, l in enumerate(all_labels)}

    qqp_train, qqp_val = load_qqp_pairs()
    mcq_data = load_mcq_corpus()
    mcq_train, mcq_val, mcq_test = mcq_data["train"], mcq_data["val"], mcq_data["test"]
    print(f"Train/val/test (seen intents): {len(train_ex)}/{len(val_ex)}/{len(test_ex)}  "
          f"| OOS: {len(test_oos_ex)}  | zero-shot: {len(test_zs_ex)}  "
          f"| seen intents: {len(seen_labels)}  | total: {len(all_labels)}", flush=True)
    print(f"QQP auxiliary pairs: train {len(qqp_train)}  val {len(qqp_val)}", flush=True)
    print(f"MCQ answer-selection examples: train {len(mcq_train)}  val {len(mcq_val)}  test {len(mcq_test)}",
          flush=True)

    tokenizer = get_tokenizer()
    model = JEPAPolyEncoderV4(freeze_layers=args.freeze_layers).to(device)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        print(f"Resumed model weights from {args.resume} (epoch {resume_ckpt['epoch']}, "
              f"val_acc {resume_ckpt['val_acc']:.4f})", flush=True)

    backbone_has_trainable = any(p.requires_grad for p in model.context_encoder.backbone.parameters())
    if backbone_has_trainable and not args.no_grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.", flush=True)
    elif not backbone_has_trainable:
        print("Backbone fully frozen: skipping gradient checkpointing.", flush=True)
    else:
        print("Gradient checkpointing disabled via --no_grad_checkpoint (trading memory for speed).",
              flush=True)
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,} "
          f"({100 * model.num_trainable_params() / model.num_params():.1f}%)", flush=True)

    if args.compile:
        # Compile the two encoder sub-modules individually rather than the
        # top-level model -- our model class has four separate entry-point
        # methods (encode_context, encode_outcome, compatibility,
        # compatibility_pairwise), not one unified forward(), which is a
        # poor fit for compiling the whole object. context_encoder and
        # outcome_encoder each have a single clear forward(input_ids,
        # attention_mask), the actual hot path called every step -- a much
        # more standard torch.compile target. Attribute access (e.g.
        # build_param_groups reading .backbone.layers) still works because
        # torch.compile's OptimizedModule wrapper proxies attribute access
        # to the underlying module.
        model.context_encoder = torch.compile(model.context_encoder)
        model.outcome_encoder = torch.compile(model.outcome_encoder)
        print("torch.compile enabled on context_encoder and outcome_encoder "
              "(first few steps will be slower -- compilation warmup).", flush=True)

    param_groups = build_param_groups(model, args.base_lr)
    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)
        print("Using bitsandbytes 8-bit AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    bce_qqp = nn.BCEWithLogitsLoss()
    # No label_smoothing here: compatibility_grouped pads unused option
    # slots to -inf, and CrossEntropyLoss's label-smoothing term averages
    # -log_softmax over EVERY class including padding -- smoothing_eps/K *
    # (-inf) is inf, so smoothing would blow up the loss on any padded row.
    ce_mcq = nn.CrossEntropyLoss()

    batch_size = args.batch_size
    epochs = args.epochs
    n = len(train_ex)
    micro_steps_per_epoch = math.ceil(n / batch_size)
    opt_steps_per_epoch = math.ceil(micro_steps_per_epoch / args.grad_accum)
    total_opt_steps = opt_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_opt_steps * 0.05))

    def lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # Resuming: restore optimizer momentum/variance state and fast-forward
    # the LR schedule to where it left off (last_epoch here means "last
    # scheduler step", torch's naming, not our training epoch) -- without
    # this the schedule would restart at the warmup LR and Adam would lose
    # its accumulated statistics, both of which would make the first few
    # resumed steps behave like a fresh run, not a continuation.
    resume_opt_step = 0
    start_epoch = 1
    best_val = -1.0
    best_zero_shot = -1.0
    no_improve = 0
    if resume_ckpt is not None:
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            resume_opt_step = resume_ckpt.get("opt_step", 0)
            print(f"Resumed optimizer state, opt_step={resume_opt_step}", flush=True)
        else:
            print("Checkpoint has no optimizer_state (older format) -- optimizer starts fresh.", flush=True)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt.get("val_acc", -1.0)
        best_zero_shot = resume_ckpt.get("zero_shot_acc", -1.0)
        no_improve = resume_ckpt.get("no_improve", 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale, last_epoch=resume_opt_step - 1)
    scaler = _NoOpScaler()

    seed_counter = [0]

    INTENT_TOTAL_OPTIONS = args.intent_options  # 1 correct + (N-1) distractors per
    # example, freshly resampled every step -- unlike the old fixed-255-bank
    # scoring (every example always discriminated against the exact same
    # closed set), this makes the intent task structurally identical to
    # mcq_corpus's per-example variable candidate set. A SHARED constant
    # with run_eval below matters: eval must pose the exact same size task
    # training solves, not an easier-or-harder one, or the accuracy number
    # isn't measuring the skill actually being trained. Tunable via
    # --intent_options since the right size is an open empirical question --
    # too few (e.g. 10) risks an easy local shortcut that doesn't need
    # broad embedding structure; too many re-approaches the old closed-set
    # memorization problem.
    intent_option_rng = random.Random(4000)

    def sample_intent_option_set(true_label):
        others_pool = [l for l in seen_labels if l != true_label]
        distractors = intent_option_rng.sample(others_pool, INTENT_TOTAL_OPTIONS - 1)
        options = distractors + [true_label]
        intent_option_rng.shuffle(options)
        return options, options.index(true_label)

    qqp_rng = random.Random(555)

    def sample_qqp_batch():
        batch = qqp_rng.sample(qqp_train, args.qqp_batch_size)
        t1 = [p[0] for p in batch]
        t2 = [p[1] for p in batch]
        labels = torch.tensor([float(p[2]) for p in batch], device=device)
        return t1, t2, labels

    mcq_rng = random.Random(999)

    def sample_mcq_batch():
        return mcq_rng.sample(mcq_train, args.mcq_batch_size)

    base_seen_texts = [base_description(raw_intent_name(l)) for l in seen_labels]
    base_all_texts = [base_description(raw_intent_name(l)) for l in all_labels]

    def sample_eval_option_set(true_label, label_pool, seed):
        """Same shape as sample_intent_option_set, but with a FIXED
        per-example seed -- eval must be reproducible run-to-run (so
        epoch-to-epoch accuracy changes reflect the model improving, not
        which random distractors happened to get drawn this time)."""
        rng = random.Random(seed)
        others_pool = [l for l in label_pool if l != true_label]
        distractors = rng.sample(others_pool, min(INTENT_TOTAL_OPTIONS - 1, len(others_pool)))
        options = distractors + [true_label]
        rng.shuffle(options)
        return options, options.index(true_label)

    def run_eval(examples, label_pool, eval_bs=16):
        """Mirrors the training task EXACTLY: same routing-question framing
        (routing_question_text), same small per-example candidate set size
        (INTENT_TOTAL_OPTIONS), same compatibility_grouped scoring -- not a
        different, easier-or-harder full-bank task. label_pool is what
        distractors get drawn from (seen_labels for val_acc, all_labels for
        zero_shot_acc, so a held-out zero-shot example can be confused with
        either other zero-shot intents or ordinary seen ones, same as a
        real deployment would face)."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [routing_question_text(t) for t, _ in batch]
                ctok, cmask = tokenize(tokenizer, texts, device)

                option_sets, answer_idxs = zip(*(
                    sample_eval_option_set(l, label_pool, seed=9000 + i + j)
                    for j, (_, l) in enumerate(batch)))
                lab = torch.tensor(answer_idxs, device=device)
                flat_option_texts = [base_description(raw_intent_name(l))
                                      for opts in option_sets for l in opts]
                otok, omask = tokenize(tokenizer, flat_option_texts, device)

                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    option_emb = model.encode_outcome(otok, omask).view(len(batch), INTENT_TOTAL_OPTIONS, -1)
                    valid_mask = torch.ones(len(batch), INTENT_TOTAL_OPTIONS, dtype=torch.bool, device=device)
                    ctx_codes = model.encode_context(ctok, cmask)
                    logits = model.compatibility_grouped(ctx_codes, option_emb, valid_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_qqp_eval(pairs, eval_bs=32):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(pairs), eval_bs):
                batch = pairs[i:i + eval_bs]
                t1 = [p[0] for p in batch]
                t2 = [p[1] for p in batch]
                lab = torch.tensor([float(p[2]) for p in batch], device=device)
                tok1, mask1 = tokenize(tokenizer, t1, device)
                tok2, mask2 = tokenize(tokenizer, t2, device)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    ctx_codes = model.encode_context(tok1, mask1)
                    outcome_emb = model.encode_outcome(tok2, mask2)
                    logits = model.compatibility_pairwise(ctx_codes, outcome_emb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == lab).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_mcq_eval(examples, eval_bs=16):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [mcq_context_text(ex) for ex in batch]
                ctok, cmask = tokenize(tokenizer, texts, device, max_length=MCQ_CONTEXT_MAX_LENGTH)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    ctx_codes = model.encode_context(ctok, cmask)
                outcome_emb, valid_mask, answer_idx = encode_mcq_batch(model, tokenizer, batch, device)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    logits = model.compatibility_grouped(ctx_codes, outcome_emb, valid_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == answer_idx).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_oos_check(outcome_texts_eval, n_sample=1000, eval_bs=24):
        model.eval()
        with torch.no_grad():
            out_emb = encode_outcome_bank(model, tokenizer, outcome_texts_eval, device)

        def max_sim(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [t for t, _ in batch]
                    ctok, cmask = tokenize(tokenizer, texts, device)
                    with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                        ctx_codes = model.encode_context(ctok, cmask)
                        logits = model.compatibility(ctx_codes, out_emb)
                    sims.append(logits.max(dim=-1).values)
            return torch.cat(sims)

        in_scope = max_sim(test_ex[:n_sample])
        oos = max_sim(test_oos_ex)
        model.train()
        return in_scope.mean().item(), oos.mean().item()

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.dirname(__file__)
    os.makedirs(ckpt_dir, exist_ok=True)
    zero_shot_eval_subset = test_zs_ex[:400]
    mcq_val_subset = mcq_val[:args.mcq_eval_n]

    avg_options = sum(len(ex["options"]) for ex in mcq_val_subset) / len(mcq_val_subset)
    print(f"\n=== Baseline metrics BEFORE this run's training (current weights, "
          f"pre-training -- so we can tell whether THIS run actually improves anything, "
          f"not just compare against whatever the last run happened to end at) ===", flush=True)
    baseline_val_acc = run_eval(val_ex, seen_labels)
    baseline_zero_shot_acc = run_eval(zero_shot_eval_subset, all_labels)
    baseline_qqp_val_acc = run_qqp_eval(qqp_val)
    baseline_mcq_acc = run_mcq_eval(mcq_val_subset)
    print(f"baseline val_acc:         {baseline_val_acc:.4f}", flush=True)
    print(f"baseline zero_shot_acc:   {baseline_zero_shot_acc:.4f}", flush=True)
    print(f"baseline qqp_val_acc:     {baseline_qqp_val_acc:.4f}", flush=True)
    print(f"baseline mcq_val_acc:     {baseline_mcq_acc:.4f}  "
          f"(chance ~{100 / avg_options:.1f}% at avg {avg_options:.2f} options/question)", flush=True)
    import json as _json_baseline
    with open(os.path.join(ckpt_dir, "exp6_mcq_baseline.json"), "w") as _bf:
        _json_baseline.dump({"baseline_val_acc": baseline_val_acc, "baseline_zero_shot_acc": baseline_zero_shot_acc,
                              "baseline_qqp_val_acc": baseline_qqp_val_acc, "baseline_mcq_val_acc": baseline_mcq_acc,
                              "chance_level": 1 / avg_options, "avg_options_per_question": avg_options,
                              "n_eval": len(mcq_val_subset), "resumed_from": args.resume}, _bf, indent=2)
    if use_wandb:
        wandb.log({"baseline/val_acc": baseline_val_acc, "baseline/zero_shot_acc": baseline_zero_shot_acc,
                   "baseline/qqp_val_acc": baseline_qqp_val_acc, "baseline/mcq_val_acc": baseline_mcq_acc},
                   step=start_epoch - 1)

    print(f"\n=== Training (starting at epoch {start_epoch}, best_val={best_val:.4f}, "
          f"best_zero_shot={best_zero_shot:.4f}) ===", flush=True)
    opt_step = resume_opt_step
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n).tolist()
        total_intent_loss = 0.0
        total_qqp_loss = 0.0
        qqp_examples_this_epoch = 0
        qqp_steps_this_epoch = 0
        total_mcq_loss = 0.0
        mcq_examples_this_epoch = 0
        mcq_steps_this_epoch = 0
        optimizer.zero_grad()

        for micro_i, i in enumerate(range(0, n, batch_size)):
            idx = perm[i:i + batch_size]
            batch = [train_ex[j] for j in idx]
            texts = [routing_question_text(t) for t, _ in batch]

            option_sets, answer_idxs = zip(*(sample_intent_option_set(l) for _, l in batch))
            lab = torch.tensor(answer_idxs, device=device)
            seed_counter[0] += 1
            desc_rng = random.Random(2000 + seed_counter[0])
            flat_option_texts = [sample_outcome_description(raw_intent_name(l), desc_rng)
                                  for opts in option_sets for l in opts]

            ctok, cmask = tokenize(tokenizer, texts, device)
            # Chunked (not one giant forward pass): batch_size * INTENT_TOTAL_OPTIONS
            # texts here (e.g. 32*50=1600) is exactly the kind of single-shot
            # OOM risk encode_outcome_bank's chunking already exists to avoid.
            flat_option_emb = encode_outcome_bank(model, tokenizer, flat_option_texts, device)
            option_emb = flat_option_emb.view(len(batch), INTENT_TOTAL_OPTIONS, -1)
            valid_mask = torch.ones(len(batch), INTENT_TOTAL_OPTIONS, dtype=torch.bool, device=device)

            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                ctx_codes = model.encode_context(ctok, cmask)
                logits = model.compatibility_grouped(ctx_codes, option_emb, valid_mask)
                intent_loss = ce(logits, lab)

            # QQP auxiliary step -- same encoders, real human paraphrases.
            # Only every --qqp_every_n_steps micro-steps: a direct speed/
            # signal-density tradeoff, not a free win -- skipping it on
            # some steps cuts real compute (two fewer full encoder passes)
            # at the cost of a less densely-applied auxiliary signal.
            run_qqp_this_step = (micro_i % args.qqp_every_n_steps == 0)
            if run_qqp_this_step:
                qt1, qt2, qlab = sample_qqp_batch()
                qtok1, qmask1 = tokenize(tokenizer, qt1, device)
                qtok2, qmask2 = tokenize(tokenizer, qt2, device)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    q_ctx_codes = model.encode_context(qtok1, qmask1)
                    q_outcome_emb = model.encode_outcome(qtok2, qmask2)
                    q_logits = model.compatibility_pairwise(q_ctx_codes, q_outcome_emb)
                    qqp_loss = bce_qqp(q_logits, qlab)
                total_qqp_loss += qqp_loss.item() * len(idx)
                qqp_examples_this_epoch += len(idx)
                qqp_steps_this_epoch += 1

            run_mcq_this_step = (micro_i % args.mcq_every_n_steps == 0)
            if run_mcq_this_step:
                mcq_batch = sample_mcq_batch()
                mq_texts = [mcq_context_text(ex) for ex in mcq_batch]
                mqtok, mqmask = tokenize(tokenizer, mq_texts, device, max_length=MCQ_CONTEXT_MAX_LENGTH)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    mq_ctx_codes = model.encode_context(mqtok, mqmask)
                mq_outcome_emb, mq_valid_mask, mq_answer_idx = encode_mcq_batch(model, tokenizer, mcq_batch, device)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    mq_logits = model.compatibility_grouped(mq_ctx_codes, mq_outcome_emb, mq_valid_mask)
                    mcq_loss = ce_mcq(mq_logits, mq_answer_idx)
                total_mcq_loss += mcq_loss.item() * len(mcq_batch)
                mcq_examples_this_epoch += len(mcq_batch)
                mcq_steps_this_epoch += 1

            loss = intent_loss
            if run_qqp_this_step:
                loss = loss + args.qqp_weight * qqp_loss
            if run_mcq_this_step:
                loss = loss + args.mcq_weight * mcq_loss
            loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            total_intent_loss += intent_loss.item() * len(idx)

            is_last_micro = (micro_i + 1) == micro_steps_per_epoch
            if (micro_i + 1) % args.grad_accum == 0 or is_last_micro:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if opt_step < total_opt_steps:
                    scheduler.step()
                opt_step += 1

            # Per-step progress line -- the per-epoch summary print (below)
            # only fires once an entire epoch finishes, which at these batch
            # sizes can be 30-40+ minutes of total silence in the log file.
            # Printed every step (not just every N) since at batch_size=384
            # there are only ~116 steps/epoch total -- not excessive output,
            # and the alternative (silence) is exactly what made it
            # impossible to tell "still working" from "stuck" earlier today.
            step_elapsed = time.time() - t0
            steps_done = micro_i + 1
            avg_step_time = step_elapsed / steps_done
            eta_seconds = avg_step_time * (micro_steps_per_epoch - steps_done)
            if device.type == "cuda":
                mem_used_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_str = f"  gpu_mem {mem_used_gib:.1f}GiB"
            else:
                mem_str = ""
            print(f"  epoch {epoch}/{epochs}  step {steps_done}/{micro_steps_per_epoch} "
                  f"({100 * steps_done / micro_steps_per_epoch:.0f}%)  "
                  f"step_time {step_elapsed / steps_done:.1f}s avg  "
                  f"elapsed {step_elapsed:.0f}s  ETA {eta_seconds:.0f}s{mem_str}", flush=True)

        val_acc = run_eval(val_ex, seen_labels)
        zero_shot_acc = run_eval(zero_shot_eval_subset, all_labels)
        qqp_val_acc = run_qqp_eval(qqp_val)
        mcq_val_acc = run_mcq_eval(mcq_val_subset)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[-1]
        avg_qqp_loss = total_qqp_loss / qqp_examples_this_epoch if qqp_examples_this_epoch else float("nan")
        avg_mcq_loss = total_mcq_loss / mcq_examples_this_epoch if mcq_examples_this_epoch else float("nan")
        print(f"epoch {epoch}/{epochs}  intent_loss {total_intent_loss / n:.4f}  "
              f"qqp_loss {avg_qqp_loss:.4f} (ran {qqp_steps_this_epoch}/{micro_steps_per_epoch} steps)  "
              f"mcq_loss {avg_mcq_loss:.4f} (ran {mcq_steps_this_epoch}/{micro_steps_per_epoch} steps)  "
              f"val_acc {val_acc:.4f}  "
              f"zero_shot_acc {zero_shot_acc:.4f}  qqp_val_acc {qqp_val_acc:.4f}  mcq_val_acc {mcq_val_acc:.4f}  "
              f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        # Lightweight metrics history, written to the SAME persistent
        # ckpt_dir as the checkpoints (e.g. mounted Drive) -- a few KB, so
        # progress can always be checked instantly without loading a
        # multi-GB checkpoint just to read three numbers. This is exactly
        # what the training log (which only ever lived on the ephemeral
        # VM disk) couldn't give us after a session disconnect.
        import json as _json
        with open(os.path.join(ckpt_dir, "exp6_metrics.jsonl"), "a") as _mf:
            _mf.write(_json.dumps({
                "epoch": epoch, "val_acc": val_acc, "zero_shot_acc": zero_shot_acc,
                "qqp_val_acc": qqp_val_acc, "mcq_val_acc": mcq_val_acc, "intent_loss": total_intent_loss / n,
                "qqp_loss": avg_qqp_loss, "mcq_loss": avg_mcq_loss, "lr": cur_lr, "epoch_seconds": dt,
                "timestamp": time.time(),
            }) + "\n")
        if use_wandb:
            wandb.log({"val_acc": val_acc, "zero_shot_acc": zero_shot_acc, "qqp_val_acc": qqp_val_acc,
                       "mcq_val_acc": mcq_val_acc, "intent_loss": total_intent_loss / n,
                       "qqp_loss": avg_qqp_loss, "mcq_loss": avg_mcq_loss, "lr": cur_lr,
                       "epoch_seconds": dt}, step=epoch)

        def make_ckpt():
            return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "opt_step": opt_step, "epoch": epoch, "val_acc": val_acc,
                    "zero_shot_acc": zero_shot_acc, "qqp_val_acc": qqp_val_acc,
                    "mcq_val_acc": mcq_val_acc, "no_improve": no_improve}

        torch.save(make_ckpt(), os.path.join(ckpt_dir, "exp6_latest.pt"))

        if zero_shot_acc > best_zero_shot:
            best_zero_shot = zero_shot_acc
            torch.save(make_ckpt(), os.path.join(ckpt_dir, "exp6_best_zeroshot.pt"))
            print(f"  -> new best zero_shot_acc {zero_shot_acc:.4f}, saved exp6_best_zeroshot.pt", flush=True)

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save(make_ckpt(), os.path.join(ckpt_dir, "exp6_best.pt"))
            print(f"  -> new best val_acc {val_acc:.4f}, saved exp6_best.pt", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "exp6_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_acc {best_ckpt['val_acc']:.4f})", flush=True)

    test_acc = run_eval(test_ex, seen_labels)
    print(f"test_acc (seen intents, base descriptions): {test_acc:.4f}", flush=True)

    zs_acc = run_eval(test_zs_ex, all_labels)
    seen_mixed_acc = run_eval(test_ex[:1500], all_labels)
    chance = 1.0 / len(all_labels)
    print(f"\nZERO-SHOT test_acc ({len(zero_shot_labels)} never-trained intents, "
          f"pool={len(all_labels)}): {zs_acc:.4f}  (n={len(test_zs_ex)})", flush=True)
    print(f"  (reference) seen-intent acc in same mixed pool: {seen_mixed_acc:.4f}  | chance: {chance:.4f}",
          flush=True)

    in_scope_logit, oos_logit = run_oos_check(base_seen_texts)
    print(f"\nOOS separation: mean max-compat-logit  in-scope={in_scope_logit:.4f}  oos={oos_logit:.4f}  "
          f"gap={in_scope_logit - oos_logit:.4f}", flush=True)

    qqp_test_acc = run_qqp_eval(qqp_val)
    print(f"\nQQP val accuracy (final): {qqp_test_acc:.4f}", flush=True)

    mcq_test_acc = run_mcq_eval(mcq_test)
    print(f"\nMCQ test accuracy (final, {len(mcq_test)} held-out questions): {mcq_test_acc:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
