"""Standalone full-eval runner for a saved experiment-4 checkpoint.
Training was stopped intentionally (user request) at epoch 9 to inspect
checkpoint quality directly rather than waiting for early-stop. This
loads a given checkpoint and runs the same final-evaluation suite
train.py's "Final evaluation" section would have run: seen-intent test
accuracy, paraphrase generalization, TRUE zero-shot accuracy, and OOS
separation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
from torch.amp import autocast

from dataset_v4 import build_combined_dataset, base_description, raw_intent_name
from paraphrases_v4 import get_paraphrases
from model import JEPAPolyEncoderV4, get_tokenizer
from train import tokenize_context, encode_outcome_bank, AUTOCAST_DTYPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--freeze_layers", type=int, default=28)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    data = build_combined_dataset()
    test_ex, test_oos_ex, test_zs_ex = data["test"], data["test_oos"], data["test_zero_shot"]
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])
    seen_idx = {l: i for i, l in enumerate(seen_labels)}
    all_idx = {l: i for i, l in enumerate(all_labels)}

    tokenizer = get_tokenizer()
    model = JEPAPolyEncoderV4(freeze_layers=args.freeze_layers).to(device)

    ckpt_path = os.path.join(os.path.dirname(__file__), args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc {ckpt['val_acc']:.4f})", flush=True)

    base_seen_texts = [base_description(raw_intent_name(l)) for l in seen_labels]
    base_all_texts = [base_description(raw_intent_name(l)) for l in all_labels]

    def run_eval(examples, outcome_texts_eval, idx_map, eval_bs=24):
        with torch.no_grad():
            out_emb = encode_outcome_bank(model, tokenizer, outcome_texts_eval, device)
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [t for t, _ in batch]
                lab = torch.tensor([idx_map[l] for _, l in batch], device=device)
                ctok, cmask = tokenize_context(tokenizer, texts, device)
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    ctx_codes = model.encode_context(ctok, cmask)
                    logits = model.compatibility(ctx_codes, out_emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += len(batch)
        return correct / total

    def run_oos_check(outcome_texts_eval, n_sample=1000, eval_bs=24):
        with torch.no_grad():
            out_emb = encode_outcome_bank(model, tokenizer, outcome_texts_eval, device)

        def max_logit(examples):
            vals = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [t for t, _ in batch]
                    ctok, cmask = tokenize_context(tokenizer, texts, device)
                    with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                        ctx_codes = model.encode_context(ctok, cmask)
                        logits = model.compatibility(ctx_codes, out_emb)
                    vals.append(logits.max(dim=-1).values)
            return torch.cat(vals)

        in_scope = max_logit(test_ex[:n_sample])
        oos = max_logit(test_oos_ex)
        return in_scope.mean().item(), oos.mean().item()

    print("\n=== Evaluation ===", flush=True)
    test_acc = run_eval(test_ex, base_seen_texts, seen_idx)
    print(f"test_acc (seen intents, base descriptions): {test_acc:.4f}", flush=True)

    paraphrases = get_paraphrases()
    paraphrases = {k: v for k, v in paraphrases.items() if k in seen_idx}
    para_texts = [paraphrases.get(l, base_description(raw_intent_name(l))) for l in seen_labels]
    test_para_subset = [(t, l) for t, l in test_ex if l in paraphrases]
    para_acc = run_eval(test_para_subset, para_texts, seen_idx)
    base_acc_subset = run_eval(test_para_subset, base_seen_texts, seen_idx)
    print(f"test_acc, {len(paraphrases)}-intent subset, UNSEEN paraphrases: {para_acc:.4f}  "
          f"(n={len(test_para_subset)})", flush=True)
    print(f"  same subset, base descriptions: {base_acc_subset:.4f}", flush=True)
    print(f"  paraphrase gap: {base_acc_subset - para_acc:.4f}", flush=True)

    zs_acc = run_eval(test_zs_ex, base_all_texts, all_idx)
    seen_mixed_acc = run_eval(test_ex[:1500], base_all_texts, all_idx)
    chance = 1.0 / len(all_labels)
    print(f"\nZERO-SHOT test_acc ({len(zero_shot_labels)} never-trained intents, "
          f"candidate pool = all {len(all_labels)} intents): {zs_acc:.4f}  (n={len(test_zs_ex)})", flush=True)
    print(f"  (reference) seen-intent acc in same mixed pool: {seen_mixed_acc:.4f}  | chance: {chance:.4f}",
          flush=True)

    in_scope_logit, oos_logit = run_oos_check(base_seen_texts)
    print(f"\nOOS separation: mean max-compat-logit  in-scope={in_scope_logit:.4f}  oos={oos_logit:.4f}  "
          f"gap={in_scope_logit - oos_logit:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
