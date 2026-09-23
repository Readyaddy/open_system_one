"""Experiment 7 post-hoc evaluation -- run against a saved checkpoint,
never during training (the per-epoch loop in train.py only tracks val_acc /
zero_shot_acc / banking77_holdout_acc for speed). This is where the actual
deliverables from NOTES.md Sec 7 get produced:

  --mode sweep      The headline figure: cardinality x ablation-mode table
                     (text-only ~ Laya, vector-only ~ bi-encoder, both ~ ours).
  --mode external    AG News / DAIR Emotion (zero-shot, vs. Laya 95.0/59.5,
                     Jev 91.0/48.0) + Banking77 holdout (TRUE zero-shot here,
                     vs. Laya 42.5, Jev 87.0) + typed-decisions (target domain).
  --mode order_inv   Shuffle option order, measure prediction flip rate.
  --mode ece         Expected calibration error, per cardinality bucket.
  --mode latency     Wall-clock forward-pass time vs. option count.
  --mode all         Everything above.

Usage: python eval.py --ckpt path/to/exp7_best_zeroshot.pt --mode all
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.amp import autocast

AUTOCAST_DTYPE = torch.bfloat16


def _ac(device):
    """Shared autocast context -- same dtype/enable-condition as train.py,
    so eval numbers reflect the same precision training and (eventual)
    production inference actually run in, and so latency measurements here
    are comparable to Laya/Jev's own reported numbers rather than an
    artificially slow fp32 baseline."""
    return autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda"))

from model import HybridDecisionModel, PackedSequenceBuilder, PackedExample, get_tokenizer, BACKBONE
import data as D

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_model(ckpt_path, device, backbone=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    backbone = backbone or saved_args.get("backbone", BACKBONE)
    tokenizer = get_tokenizer(backbone)
    model = HybridDecisionModel(
        backbone=backbone, mask_token_id=tokenizer.mask_token_id,
        n_context_codes=saved_args.get("n_context_codes", 16),
        k_max=saved_args.get("k_max", 6), head_n_layers=saved_args.get("head_n_layers", 2),
        maxsim_dim=saved_args.get("maxsim_dim", 128), use_maxsim=saved_args.get("use_maxsim", False),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    builder = PackedSequenceBuilder(
        tokenizer, budget_total=saved_args.get("budget_total", 2048),
        l_context=saved_args.get("l_context", 768), l_instructions=saved_args.get("l_instructions", 96),
        l_max_per_option=saved_args.get("l_max_per_option", 64),
    )
    print(f"Loaded {ckpt_path}: epoch {ckpt.get('epoch')}  val_acc {ckpt.get('val_acc'):.4f}  "
          f"zero_shot_acc {ckpt.get('zero_shot_acc'):.4f}  "
          f"banking77_holdout_acc {ckpt.get('banking77_holdout_acc', float('nan')):.4f}", flush=True)
    return model, tokenizer, builder, saved_args


@torch.no_grad()
def run_batch_accuracy(model, tokenizer, builder, device, examples, k=None, option_chunk_size=64,
                        eval_batch_size=8):
    correct, total = 0, 0
    for i in range(0, len(examples), eval_batch_size):
        chunk = examples[i:i + eval_batch_size]
        batch = builder.build_batch(chunk, device)
        with _ac(device):
            logits_per_depth = model(tokenizer, batch, device, k=k, option_chunk_size=option_chunk_size)
        preds = logits_per_depth[-1].argmax(dim=-1)
        correct += (preds == batch.answer_idx).sum().item()
        total += len(chunk)
    return correct / total if total else float("nan")


# --------------------------------------------------------------------------
# 1. Cardinality sweep x ablation mode -- the headline figure (Sec 7)
# --------------------------------------------------------------------------

CARDINALITIES = [2, 4, 8, 20, 77, 255]
ABLATION_MODES = {
    "text_only": (True, False),     # ~ Laya (joint sequence, no injected vector)
    "vector_only": (False, True),   # ~ bi-encoder (no option text at all)
    "both": (True, True),           # exp7 as designed
}


def run_cardinality_sweep(model, tokenizer, builder, device, intent, n_per_cell=150,
                           option_chunk_size=64, seed=31337):
    rng = random.Random(seed)
    results = {}
    pool = intent.val
    for mode_name, (use_text, use_vector) in ABLATION_MODES.items():
        results[mode_name] = {}
        for n in CARDINALITIES:
            n_target = min(n, len(intent.seen_labels))
            examples = []
            sample_pool = rng.sample(pool, min(n_per_cell, len(pool)))
            for text, label in sample_pool:
                ex = D.build_intent_example(text, label, intent.seen_labels, rng, n_target=n_target)
                ex.use_text, ex.use_vector = use_text, use_vector
                examples.append(ex)
            acc = run_batch_accuracy(model, tokenizer, builder, device, examples,
                                      option_chunk_size=option_chunk_size)
            results[mode_name][n] = acc
            print(f"  sweep  mode={mode_name:12s}  N={n:4d}  acc={acc:.4f}", flush=True)
    return results


# --------------------------------------------------------------------------
# 2. External benchmarks
# --------------------------------------------------------------------------

AG_NEWS_OPTIONS = ["World news", "Sports news", "Business news", "Science and technology news"]
EMOTION_OPTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def run_fixed_bank_eval(model, tokenizer, builder, device, rows, instructions, options,
                         option_chunk_size=64):
    examples = [PackedExample(context=text, instructions=instructions, option_texts=options,
                               qtype="choice", answer_idx=label, use_text=True, use_vector=True)
                for text, label in rows]
    return run_batch_accuracy(model, tokenizer, builder, device, examples, option_chunk_size=option_chunk_size)


def run_banking77_fixed_question(model, tokenizer, builder, device, intent, option_chunk_size=64):
    """Same holdout set, same 71-way option bank, one variable changed: a
    single, consistently task-appropriate routing question for every
    example, and consistent bare-label option rendering (matching
    AG_NEWS_OPTIONS/EMOTION_OPTIONS' plain style above) -- instead of
    build_intent_example's per-example RANDOM draw from a 6-item generic
    instruction bank (which includes "" and non-routing phrasings like
    "Pick the option that best matches this request") and random choice
    among 4 option-rendering styles. Answers directly: does asking the
    "right" question, at inference time, change what an already-trained
    checkpoint does -- no retraining involved, this only changes the input.
    """
    option_texts_fixed = [D.render_intent_option(D.raw_intent_name(l), "bare", random.Random(0))
                           for l in intent.banking77_labels]
    examples = []
    for text, label in intent.banking77_holdout:
        answer_idx = intent.banking77_labels.index(label)
        examples.append(PackedExample(
            context=text, instructions="Which category should this be routed to?",
            option_texts=option_texts_fixed, qtype="choice", answer_idx=answer_idx,
            use_text=True, use_vector=True,
        ))
    return run_batch_accuracy(model, tokenizer, builder, device, examples, option_chunk_size=option_chunk_size)


def run_external_benchmarks(model, tokenizer, builder, device, intent, n=1000):
    out = {}
    if HAS_DATASETS:
        ag = load_dataset("ag_news", split="test").select(range(min(n, 1000)))
        ag_rows = [(ex["text"], ex["label"]) for ex in ag]
        out["ag_news"] = run_fixed_bank_eval(model, tokenizer, builder, device, ag_rows,
                                              "Which category best describes this news article?",
                                              AG_NEWS_OPTIONS)
        em = load_dataset("dair-ai/emotion", split="test").select(range(min(n, 1000)))
        em_rows = [(ex["text"], ex["label"]) for ex in em]
        out["dair_emotion"] = run_fixed_bank_eval(model, tokenizer, builder, device, em_rows,
                                                   "What emotion does this text express?", EMOTION_OPTIONS)
        print(f"  AG News (zero-shot):     {out['ag_news']:.4f}  (Laya 0.950, Jev 0.910, chance 0.25)",
              flush=True)
        print(f"  DAIR Emotion (zero-shot):{out['dair_emotion']:.4f}  (Laya 0.595, Jev 0.480, chance 0.167)",
              flush=True)
    else:
        print("  `datasets` not installed -- skipping AG News / DAIR Emotion.", flush=True)

    if intent.banking77_holdout:
        rng = random.Random(2024)
        examples = [D.build_intent_example(t, l, intent.banking77_labels, rng, n_target=len(intent.banking77_labels))
                    for t, l in intent.banking77_holdout]
        out["banking77_holdout"] = run_batch_accuracy(model, tokenizer, builder, device, examples)
        print(f"  Banking77 holdout (TRUE zero-shot, {len(intent.banking77_labels)}-way, "
              f"randomized generic instructions): "
              f"{out['banking77_holdout']:.4f}  (Laya 0.425, Jev 0.870, chance ~0.013)", flush=True)

        # Same holdout, same checkpoint, same option set -- ONE thing changed:
        # a fixed, consistently routing-appropriate question ("Which category
        # should this be routed to?") instead of build_intent_example's
        # randomized generic bank (which includes non-routing phrasings and
        # an empty-string option). This is testable against the already-
        # trained model directly -- unlike the diversity_corpus instruction-
        # bank fix (a training-data change that needs a new checkpoint to
        # show up anywhere), this changes only what's fed in at inference.
        out["banking77_holdout_routing_question"] = run_banking77_fixed_question(
            model, tokenizer, builder, device, intent)
        print(f"  Banking77 holdout (fixed routing question): "
              f"{out['banking77_holdout_routing_question']:.4f}", flush=True)

    try:
        td_rows = D.load_typed_decisions()
        examples = [D.typed_decision_to_packed(r) for r in td_rows]
        out["typed_decisions"] = run_batch_accuracy(model, tokenizer, builder, device, examples)
        print(f"  typed-decisions (target domain): {out['typed_decisions']:.4f}  (n={len(td_rows)})",
              flush=True)
    except FileNotFoundError:
        print("  exp7_typed_decisions not built -- skipping.", flush=True)

    return out


# --------------------------------------------------------------------------
# 3. Order invariance
# --------------------------------------------------------------------------

def run_order_invariance(model, tokenizer, builder, device, intent, n=200, shuffles=3, seed=17):
    rng = random.Random(seed)
    sample = rng.sample(intent.val, min(n, len(intent.val)))
    flips, total = 0, 0
    for text, label in sample:
        base_ex = D.build_intent_example(text, label, intent.seen_labels, rng, n_target=20)
        base_batch = builder.build_batch([base_ex], device)
        with torch.no_grad():
            with _ac(device):
                base_pred = model(tokenizer, base_batch, device)[-1].argmax(dim=-1).item()
        base_options = base_ex.option_texts
        for _ in range(shuffles):
            perm_ex = copy.deepcopy(base_ex)
            order = list(range(len(base_options)))
            rng.shuffle(order)
            perm_ex.option_texts = [base_options[i] for i in order]
            perm_ex.answer_idx = order.index(base_ex.answer_idx)
            perm_batch = builder.build_batch([perm_ex], device)
            with torch.no_grad():
                with _ac(device):
                    perm_logits = model(tokenizer, perm_batch, device)[-1]
            # Map the permuted prediction back to the ORIGINAL option identity
            # to compare against base_pred fairly.
            perm_pred_in_perm_space = perm_logits.argmax(dim=-1).item()
            perm_pred_original_identity = order[perm_pred_in_perm_space]
            total += 1
            if perm_pred_original_identity != base_pred:
                flips += 1
    rate = flips / total if total else float("nan")
    print(f"  Order-invariance flip rate: {rate:.4f}  ({flips}/{total})", flush=True)
    return rate


# --------------------------------------------------------------------------
# 4. ECE per cardinality bucket
# --------------------------------------------------------------------------

def expected_calibration_error(confidences, corrects, n_bins=10):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mask = [(lo <= c < hi) or (i == n_bins - 1 and c == hi) for c in confidences]
        if not any(mask):
            continue
        bin_conf = [c for c, m in zip(confidences, mask) if m]
        bin_corr = [x for x, m in zip(corrects, mask) if m]
        ece += (len(bin_conf) / n) * abs(sum(bin_conf) / len(bin_conf) - sum(bin_corr) / len(bin_corr))
    return ece


def run_ece(model, tokenizer, builder, device, intent, buckets=((2, 5), (6, 10), (11, 30), (31, 255)),
            n_per_bucket=150, seed=99):
    rng = random.Random(seed)
    out = {}
    for lo, hi in buckets:
        n_target = rng.randint(lo, min(hi, len(intent.seen_labels)))
        sample = rng.sample(intent.val, min(n_per_bucket, len(intent.val)))
        confidences, corrects = [], []
        for text, label in sample:
            ex = D.build_intent_example(text, label, intent.seen_labels, rng, n_target=n_target)
            batch = builder.build_batch([ex], device)
            with torch.no_grad():
                with _ac(device):
                    logits = model(tokenizer, batch, device)[-1]
            p = F.softmax(logits, dim=-1)[0]
            conf, pred = p.max(dim=-1)
            confidences.append(conf.item())
            corrects.append(int(pred.item() == batch.answer_idx.item()))
        ece = expected_calibration_error(confidences, corrects)
        out[f"{lo}-{hi}"] = ece
        print(f"  ECE bucket {lo}-{hi}: {ece:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------
# 5. Latency
# --------------------------------------------------------------------------

def run_latency(model, tokenizer, builder, device, intent, cardinalities=CARDINALITIES, n_trials=10,
                 seed=7):
    rng = random.Random(seed)
    out = {}
    text, label = intent.val[0]
    for n in cardinalities:
        n_target = min(n, len(intent.seen_labels))
        ex = D.build_intent_example(text, label, intent.seen_labels, rng, n_target=n_target)
        batch = builder.build_batch([ex], device)
        # warmup
        with torch.no_grad(), _ac(device):
            model(tokenizer, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), _ac(device):
            for _ in range(n_trials):
                model(tokenizer, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_trials * 1000
        out[n] = dt
        print(f"  latency  N={n:4d}  {dt:.1f}ms  (Laya 33-40ms, Jev 236-276ms)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--mode", type=str, default="all",
                     choices=["sweep", "external", "order_inv", "ece", "latency", "all"])
    ap.add_argument("--backbone", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="Optional path to dump results as JSON.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, builder, saved_args = load_model(args.ckpt, device, backbone=args.backbone)
    intent = D.load_intent_corpus_minus_banking77()

    results = {}
    if args.mode in ("sweep", "all"):
        print("\n=== Cardinality sweep x ablation mode (the headline figure) ===", flush=True)
        results["sweep"] = run_cardinality_sweep(model, tokenizer, builder, device, intent)
    if args.mode in ("external", "all"):
        print("\n=== External benchmarks ===", flush=True)
        results["external"] = run_external_benchmarks(model, tokenizer, builder, device, intent)
    if args.mode in ("order_inv", "all"):
        print("\n=== Order invariance ===", flush=True)
        results["order_invariance"] = run_order_invariance(model, tokenizer, builder, device, intent)
    if args.mode in ("ece", "all"):
        print("\n=== ECE per cardinality bucket ===", flush=True)
        results["ece"] = run_ece(model, tokenizer, builder, device, intent)
    if args.mode in ("latency", "all"):
        print("\n=== Latency vs. cardinality ===", flush=True)
        results["latency"] = run_latency(model, tokenizer, builder, device, intent)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
