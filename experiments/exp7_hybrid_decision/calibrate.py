"""Post-hoc per-(question type, cardinality bucket) temperature fitting
(NOTES.md Sec 2.7). Deliberately NOT learned during training -- Laya's own
published fitted values (choice:2 -> 1.91 soften, choice:11+ -> 0.10 sharpen)
are the warning this exists to check for: raw logits are expected to drift
badly miscalibrated as option count grows, and the right fix is fitting a
scalar per bucket on held-out data after training, not hoping the training
objective handles it on its own.

Grid search over a temperature range, minimizing NLL per bucket -- no
gradient-based fitting needed since we're only rescaling ALREADY-COMPUTED
logits (collect once, evaluate the grid cheaply on CPU).

Output: a JSON table {qtype: {bucket: temperature}}, meant to be applied at
INFERENCE time (divide logits by the matching temperature before softmax)
-- this file does not modify the checkpoint or the model.

Usage: python calibrate.py --ckpt path/to/exp7_best_zeroshot.pt --out calibration.json
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from eval import load_model, _ac
import data as D

BUCKETS = [(2, 2), (3, 5), (6, 10), (11, 30), (31, 255)]
TEMPERATURE_GRID = [round(math.exp(x), 4) for x in
                    [-3.0 + 0.1 * i for i in range(61)]]  # ~0.05 .. ~1.0 .. ~20, log-spaced


def bucket_for(n: int):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return f"{BUCKETS[-1][0]}-{BUCKETS[-1][1]}"


@torch.no_grad()
def collect_logits_choice(model, tokenizer, builder, device, examples, label_pool, n_per_bucket=200,
                           seed=555):
    """Collects (logits, answer_idx) pairs for the CHOICE task, spanning all
    cardinality buckets, by explicitly sampling one option-set size per
    bucket rather than relying on whatever sizes training happened to draw."""
    rng = random.Random(seed)
    by_bucket = {}
    for lo, hi in BUCKETS:
        n_target = min(rng.randint(lo, hi), len(label_pool))
        sample = rng.sample(examples, min(n_per_bucket, len(examples)))
        logits_list, ans_list = [], []
        for text, label in sample:
            ex = D.build_intent_example(text, label, label_pool, rng, n_target=n_target)
            batch = builder.build_batch([ex], device)
            with _ac(device):
                logits = model(tokenizer, batch, device)[-1][0].float().cpu()
            logits_list.append(logits)
            ans_list.append(batch.answer_idx.item())
        by_bucket[f"{lo}-{hi}"] = (logits_list, ans_list)
    return by_bucket


def nll_at_temperature(logits_list, ans_list, temp):
    total = 0.0
    for logits, ans in zip(logits_list, ans_list):
        scaled = logits / temp
        log_probs = F.log_softmax(scaled, dim=-1)
        total += -log_probs[ans].item()
    return total / len(logits_list)


def fit_best_temperature(logits_list, ans_list):
    best_t, best_nll = 1.0, float("inf")
    for t in TEMPERATURE_GRID:
        nll = nll_at_temperature(logits_list, ans_list, t)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t, best_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--backbone", type=str, default=None)
    ap.add_argument("--n_per_bucket", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, builder, _ = load_model(args.ckpt, device, backbone=args.backbone)
    intent = D.load_intent_corpus_minus_banking77()

    print("Fitting choice-type temperatures per cardinality bucket...", flush=True)
    by_bucket = collect_logits_choice(model, tokenizer, builder, device, intent.val, intent.seen_labels,
                                       n_per_bucket=args.n_per_bucket)

    table = {"choice": {}}
    for bucket, (logits_list, ans_list) in by_bucket.items():
        t, nll = fit_best_temperature(logits_list, ans_list)
        raw_nll = nll_at_temperature(logits_list, ans_list, 1.0)
        table["choice"][bucket] = t
        print(f"  choice:{bucket:8s}  temperature={t:.3f}  NLL(raw)={raw_nll:.4f}  "
              f"NLL(calibrated)={nll:.4f}", flush=True)

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "exp7_calibration.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nWrote calibration table to {out_path}", flush=True)
    print("Apply at inference: divide logits by table[qtype][bucket] before softmax.", flush=True)


if __name__ == "__main__":
    main()
