"""Evaluates our exp6 checkpoint on the same public benchmarks Laya/Jev
were compared on (per https://huggingface.co/convaiinnovations/laya),
for a real apples-to-apples reference point instead of just our own
internal numbers.

AG News and DAIR Emotion are run genuinely ZERO-SHOT (the model never
trained on either dataset) -- a fair comparison to Laya/Jev's presumably
zero-shot numbers.

Banking77 is NOT zero-shot for us: it's literally one of intent_corpus's
five source datasets, so our model trained directly on it. That number is
reported separately and clearly labeled as in-distribution / not
comparable to Laya's 0.425 or Jev's 0.870 zero-shot numbers -- it answers
a different question ("how well did fine-tuning on this data work") not
"how well does this generalize to an unseen high-cardinality task."

Usage: python eval_external_benchmarks.py --ckpt path/to/checkpoint.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import torch
from torch.amp import autocast
from datasets import load_dataset

from local_data import load_intent_corpus, base_description, raw_intent_name
from model import JEPAPolyEncoderV4, get_tokenizer
from train import tokenize, routing_question_text

AUTOCAST_DTYPE = torch.bfloat16

AG_NEWS_OPTIONS = ["World news", "Sports news", "Business news", "Science and technology news"]
EMOTION_OPTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def run_fixed_bank_eval(model, tokenizer, device, examples, question_prefix, option_texts, eval_bs=32):
    """Every example shares the SAME small option bank (AG News: 4 labels,
    Emotion: 6 labels) -- simpler than the per-example dynamic sampling
    used for intent training, since here the label set genuinely is fixed
    and small for the whole task, matching how Laya/Jev would see it too."""
    model.eval()
    with torch.no_grad():
        otok, omask = tokenize(tokenizer, option_texts, device)
        with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
            option_emb = model.encode_outcome(otok, omask)  # (K, D)
        correct, total = 0, 0
        for i in range(0, len(examples), eval_bs):
            batch = examples[i:i + eval_bs]
            texts = [f"{t}\n\nQuestion: {question_prefix}" for t, _ in batch]
            lab = torch.tensor([l for _, l in batch], device=device)
            ctok, cmask = tokenize(tokenizer, texts, device, max_length=128)
            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                ctx_codes = model.encode_context(ctok, cmask)
                logits = model.compatibility(ctx_codes, option_emb)
            preds = logits.argmax(dim=-1)
            correct += (preds == lab).sum().item()
            total += len(batch)
    return correct / total, total


def run_banking77_eval(model, tokenizer, device, eval_bs=16):
    """NOT zero-shot -- banking77 is one of intent_corpus's training
    sources. Uses our test split, filtered to banking:: labels, scored
    against the full 77-way banking-intent bank (matching Laya/Jev's
    Banking77 comparison, which also used all 77 as the candidate set)."""
    data = load_intent_corpus()
    banking_labels = sorted(l for l in data["seen_labels"] if l.startswith("banking::"))
    label_idx = {l: i for i, l in enumerate(banking_labels)}
    examples = [(t, label_idx[l]) for t, l in data["test"] if l in label_idx]
    option_texts = [base_description(raw_intent_name(l)) for l in banking_labels]

    model.eval()
    with torch.no_grad():
        otok, omask = tokenize(tokenizer, option_texts, device)
        with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
            option_emb = model.encode_outcome(otok, omask)
        correct, total = 0, 0
        for i in range(0, len(examples), eval_bs):
            batch = examples[i:i + eval_bs]
            texts = [routing_question_text(t) for t, _ in batch]
            lab = torch.tensor([l for _, l in batch], device=device)
            ctok, cmask = tokenize(tokenizer, texts, device)
            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                ctx_codes = model.encode_context(ctok, cmask)
                logits = model.compatibility(ctx_codes, option_emb)
            preds = logits.argmax(dim=-1)
            correct += (preds == lab).sum().item()
            total += len(batch)
    return correct / total, total, len(banking_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ag_news_n", type=int, default=2000)
    ap.add_argument("--emotion_n", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    print(f"Loaded {args.ckpt}: epoch {ckpt['epoch']}", flush=True)

    model = JEPAPolyEncoderV4(freeze_layers=0).to(device)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = get_tokenizer()

    results = {}

    print("\n=== AG News (4-way, ZERO-SHOT) ===", flush=True)
    ag_test = load_dataset("ag_news", split="test")
    ag_examples = [(ex["text"], ex["label"]) for ex in ag_test.select(range(min(args.ag_news_n, len(ag_test))))]
    acc, n = run_fixed_bank_eval(model, tokenizer, device, ag_examples,
                                  "Which category best describes this news article?", AG_NEWS_OPTIONS)
    print(f"AG News accuracy: {acc:.4f} (n={n}, chance=0.25)", flush=True)
    results["ag_news"] = {"accuracy": acc, "n": n, "chance": 0.25, "zero_shot": True}

    print("\n=== DAIR Emotion (6-way, ZERO-SHOT) ===", flush=True)
    em_test = load_dataset("dair-ai/emotion", split="test")
    em_examples = [(ex["text"], ex["label"]) for ex in em_test.select(range(min(args.emotion_n, len(em_test))))]
    acc, n = run_fixed_bank_eval(model, tokenizer, device, em_examples,
                                  "What emotion does this text express?", EMOTION_OPTIONS)
    print(f"DAIR Emotion accuracy: {acc:.4f} (n={n}, chance=0.1667)", flush=True)
    results["dair_emotion"] = {"accuracy": acc, "n": n, "chance": 1 / 6, "zero_shot": True}

    print("\n=== Banking77 (77-way, NOT zero-shot -- trained on this source) ===", flush=True)
    acc, n, n_labels = run_banking77_eval(model, tokenizer, device)
    print(f"Banking77 accuracy: {acc:.4f} (n={n}, {n_labels}-way, chance={1/n_labels:.4f})", flush=True)
    results["banking77"] = {"accuracy": acc, "n": n, "n_labels": n_labels,
                             "chance": 1 / n_labels, "zero_shot": False}

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)) or ".", "external_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({"source_ckpt": args.ckpt, "source_epoch": ckpt["epoch"], "results": results}, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
