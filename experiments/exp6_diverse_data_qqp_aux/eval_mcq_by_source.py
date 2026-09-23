"""Breaks mcq_val_acc down PER SOURCE DATASET, instead of one blended
number -- useful because the sources have very different intrinsic
difficulty/scale-sensitivity (HellaSwag in particular is known to stay
near chance for sub-1B models regardless of training, while SciQ/ARC-Easy
are much more tractable at this scale), so a single blended accuracy
hides which sources the model is actually doing well or badly on.

Usage: python eval_mcq_by_source.py --ckpt path/to/checkpoint.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from collections import defaultdict
import torch
from torch.amp import autocast

from local_data import load_mcq_corpus
from model import JEPAPolyEncoderV4, get_tokenizer
from train import tokenize, encode_mcq_batch, mcq_context_text, MCQ_CONTEXT_MAX_LENGTH

AUTOCAST_DTYPE = torch.bfloat16


def run_mcq_eval_by_source(model, tokenizer, device, examples, eval_bs=16):
    model.eval()
    correct_by_source = defaultdict(int)
    total_by_source = defaultdict(int)
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
            correct = (preds == answer_idx)
            for j, ex in enumerate(batch):
                total_by_source[ex["source"]] += 1
                if correct[j].item():
                    correct_by_source[ex["source"]] += 1
    return correct_by_source, total_by_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"],
                     help="test is larger and never used for per-epoch eval during training, "
                          "so it's the more honest number to report per-source.")
    ap.add_argument("--n", type=int, default=None, help="Optional cap per split for speed.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    print(f"Loaded {args.ckpt}: epoch {ckpt['epoch']}, mcq_val_acc (blended, training-time) "
          f"{ckpt.get('mcq_val_acc'):.4f}", flush=True)

    model = JEPAPolyEncoderV4(freeze_layers=0).to(device)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = get_tokenizer()

    mcq_data = load_mcq_corpus()
    examples = mcq_data[args.split]
    if args.n:
        examples = examples[:args.n]
    print(f"Evaluating on mcq_corpus['{args.split}'], n={len(examples)}", flush=True)

    source_counts = defaultdict(int)
    for ex in examples:
        source_counts[ex["source"]] += 1
    print("Source distribution in this eval set:", dict(source_counts), flush=True)

    correct_by_source, total_by_source = run_mcq_eval_by_source(model, tokenizer, device, examples)

    overall_correct = sum(correct_by_source.values())
    overall_total = sum(total_by_source.values())
    print(f"\n=== Per-source MCQ accuracy ({args.split} split) ===", flush=True)
    for source in sorted(total_by_source, key=lambda s: -total_by_source[s]):
        acc = correct_by_source[source] / total_by_source[source]
        print(f"  {source:20s}  {acc:.4f}  ({correct_by_source[source]}/{total_by_source[source]})", flush=True)
    print(f"\n  {'OVERALL (blended)':20s}  {overall_correct/overall_total:.4f}  "
          f"({overall_correct}/{overall_total})", flush=True)


if __name__ == "__main__":
    main()
