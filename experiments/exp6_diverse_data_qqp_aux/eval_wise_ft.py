"""WiSE-FT weight-space interpolation test (Wortsman et al. 2022):
theta_final = rho * theta_pretrained + (1 - rho) * theta_finetuned.

Only applies to the two backbone sub-modules (context_encoder.backbone,
outcome_encoder.backbone) -- these are the only parameters that also
exist in the original, never-fine-tuned Qwen2.5-0.5B checkpoint. The
poly-encoder heads (codes, code_attn, proj, OutcomeEncoder.head,
log_temperature) were randomly initialized and trained from scratch, so
there is no "pretrained" version of them to interpolate toward -- they
are always kept at their fine-tuned values regardless of rho.

Matches the CURRENT (post-session) eval mechanism exactly: routing-
question-framed intent eval against a per-example INTENT_TOTAL_OPTIONS-
sized candidate set (not the old full-bank compatibility()), plus MCQ
eval with real context passages -- so these numbers are directly
comparable to what train.py itself reports.

Usage: python eval_wise_ft.py --ckpt exp6_latest.pt --rhos 0.0,0.25,0.5,0.75,1.0
Writes results to wise_ft_results.json next to the checkpoint.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import random
import time
import torch
from torch.amp import autocast
from transformers import AutoModel

from local_data import load_intent_corpus, load_qqp_pairs, load_mcq_corpus, base_description, raw_intent_name
from model import JEPAPolyEncoderV4, get_tokenizer, BACKBONE
from train import (tokenize, encode_outcome_bank, encode_mcq_batch, routing_question_text,
                    mcq_context_text, MCQ_CONTEXT_MAX_LENGTH)

AUTOCAST_DTYPE = torch.bfloat16
INTENT_TOTAL_OPTIONS = 50  # must match the run being evaluated


def sample_eval_option_set(true_label, label_pool, seed):
    rng = random.Random(seed)
    others_pool = [l for l in label_pool if l != true_label]
    distractors = rng.sample(others_pool, min(INTENT_TOTAL_OPTIONS - 1, len(others_pool)))
    options = distractors + [true_label]
    rng.shuffle(options)
    return options, options.index(true_label)


def run_eval(model, tokenizer, device, examples, label_pool, eval_bs=16):
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
    return correct / total


def run_qqp_eval(model, tokenizer, device, pairs, eval_bs=32):
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
    return correct / total


def run_mcq_eval(model, tokenizer, device, examples, eval_bs=16):
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
    return correct / total


def interpolate_backbone(finetuned_state, pretrained_backbone_state, prefix, rho):
    """Returns a new dict of keys under `prefix` (e.g. 'context_encoder.backbone.')
    interpolated between pretrained and fine-tuned. rho=0 -> pure fine-tuned,
    rho=1 -> pure pretrained."""
    out = {}
    for k, v in finetuned_state.items():
        if not k.startswith(prefix):
            continue
        bare_key = k[len(prefix):]
        pre = pretrained_backbone_state[bare_key].to(v.dtype)
        out[k] = rho * pre + (1 - rho) * v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="exp6_latest.pt")
    ap.add_argument("--rhos", type=str, default="0.0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--val_n", type=int, default=400, help="val_acc eval subset size (speed).")
    ap.add_argument("--zs_n", type=int, default=400, help="Zero-shot eval subset size (speed).")
    ap.add_argument("--qqp_n", type=int, default=300, help="QQP val subset size (speed).")
    ap.add_argument("--mcq_n", type=int, default=400, help="MCQ val subset size (speed).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt)) or "."
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    finetuned_state = ckpt["model_state"]
    print(f"Loaded {args.ckpt}: epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}, "
          f"zero_shot_acc {ckpt.get('zero_shot_acc'):.4f}, qqp_val_acc {ckpt.get('qqp_val_acc'):.4f}, "
          f"mcq_val_acc {ckpt.get('mcq_val_acc'):.4f}", flush=True)

    print(f"Loading original pretrained {BACKBONE} for interpolation reference...", flush=True)
    pretrained_backbone = AutoModel.from_pretrained(BACKBONE)
    pretrained_backbone_state = {k: v.clone() for k, v in pretrained_backbone.state_dict().items()}
    del pretrained_backbone

    data = load_intent_corpus()
    val_ex = data["val"][:args.val_n]
    test_zs_ex = data["test_zero_shot"][:args.zs_n]
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    _, qqp_val = load_qqp_pairs()
    qqp_val = qqp_val[:args.qqp_n]
    mcq_data = load_mcq_corpus()
    mcq_val = mcq_data["val"][:args.mcq_n]

    tokenizer = get_tokenizer()

    results = []
    rhos = [float(x) for x in args.rhos.split(",")]
    for rho in rhos:
        t0 = time.time()
        model = JEPAPolyEncoderV4(freeze_layers=0).to(device)
        state = dict(finetuned_state)
        state.update(interpolate_backbone(finetuned_state, pretrained_backbone_state,
                                           "context_encoder.backbone.", rho))
        state.update(interpolate_backbone(finetuned_state, pretrained_backbone_state,
                                           "outcome_encoder.backbone.", rho))
        model.load_state_dict(state)
        model.to(device)

        val_acc = run_eval(model, tokenizer, device, val_ex, seen_labels)
        zs_acc = run_eval(model, tokenizer, device, test_zs_ex, all_labels)
        qqp_acc = run_qqp_eval(model, tokenizer, device, qqp_val)
        mcq_acc = run_mcq_eval(model, tokenizer, device, mcq_val)
        dt = time.time() - t0
        print(f"rho={rho:.2f}  val_acc={val_acc:.4f}  zero_shot_acc={zs_acc:.4f}  "
              f"qqp_val_acc={qqp_acc:.4f}  mcq_val_acc={mcq_acc:.4f}  ({dt:.1f}s)", flush=True)
        results.append({"rho": rho, "val_acc": val_acc, "zero_shot_acc": zs_acc,
                         "qqp_val_acc": qqp_acc, "mcq_val_acc": mcq_acc})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = os.path.join(ckpt_dir, "wise_ft_results.json")
    with open(out_path, "w") as f:
        json.dump({"source_ckpt": args.ckpt, "source_epoch": ckpt["epoch"], "results": results}, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
