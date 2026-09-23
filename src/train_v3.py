"""Iteration 3 training: bigger backbone (mpnet-base, 110M/encoder), bigger
combined dataset (CLINC150 + Banking77, 227 intents, ~25k examples), longer
schedule (mixed precision, cosine LR, many epochs), AND outcome-description
augmentation -- a different random template+synonym phrasing sampled every
training step, so the outcome encoder cannot solve the task by memorizing
one fixed string per class (the diagnosed cause of iteration 2's flat
paraphrase generalization).

Evaluation, same three axes as before, at bigger scale:
  1. In-scope test accuracy (227-way), using the plain non-augmented base
     description.
  2. Paraphrase generalization on ~50 hand-written, never-trained-on
     descriptions spanning both datasets.
  3. OOS separation using CLINC150's out-of-scope test examples.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from dataset_v3 import build_combined_dataset, sample_outcome_description, base_description, raw_intent_name
from paraphrases_v3 import get_paraphrases
from model_v3 import JEPADecisionModelV3, get_tokenizer


def tokenize(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_train", type=int, default=None, help="Smoke-test: cap train set size")
    parser.add_argument("--max_val", type=int, default=None, help="Smoke-test: cap val set size")
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = build_combined_dataset()
    train_ex, val_ex, test_ex, test_oos_ex = data["train"], data["val"], data["test"], data["test_oos"]
    if args.max_train:
        train_ex = train_ex[:args.max_train]
    if args.max_val:
        val_ex = val_ex[:args.max_val]
    labels = data["labels"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    print(f"Train/val/test (in-scope): {len(train_ex)}/{len(val_ex)}/{len(test_ex)}  "
          f"| OOS test: {len(test_oos_ex)}  | total intents: {len(labels)}", flush=True)

    tokenizer = get_tokenizer()
    model = JEPADecisionModelV3().to(device)
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,}", flush=True)

    batch_size = args.batch_size
    epochs = args.epochs
    warmup_frac = 0.05
    base_lr = 2e-5
    n = len(train_ex)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_frac)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    ce = nn.CrossEntropyLoss()

    rng_seed_counter = [0]

    def make_augmented_outcome_bank():
        import random
        rng = random.Random(1000 + rng_seed_counter[0])
        rng_seed_counter[0] += 1
        texts = [sample_outcome_description(raw_intent_name(l), rng) for l in labels]
        return texts

    base_outcome_texts = [base_description(raw_intent_name(l)) for l in labels]

    def run_eval(examples, outcome_texts_eval):
        model.eval()
        tok, mask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(tok, mask)
        correct, total = 0, 0
        eval_bs = 128
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [t for t, _ in batch]
                lab = torch.tensor([label_to_idx[l] for _, l in batch], device=device)
                ctok, cmask = tokenize(tokenizer, texts, device)
                ctx_emb = model.encode_context(ctok, cmask)
                logits = model.compatibility(ctx_emb, out_emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_oos_check(outcome_texts_eval, n_in_scope_sample=1500):
        model.eval()
        tok, mask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(tok, mask)
        eval_bs = 128

        def max_sim(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [t for t, _ in batch]
                    ctok, cmask = tokenize(tokenizer, texts, device)
                    ctx_emb = model.encode_context(ctok, cmask)
                    sim = ctx_emb @ out_emb.t()
                    sims.append(sim.max(dim=-1).values)
            return torch.cat(sims)

        in_scope_sim = max_sim(test_ex[:n_in_scope_sample])
        oos_sim = max_sim(test_oos_ex)
        model.train()
        return in_scope_sim.mean().item(), oos_sim.mean().item()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = -1.0
    no_improve_epochs = 0
    patience = args.patience

    print("\n=== Training ===", flush=True)
    global_step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n).tolist()
        total_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = [train_ex[j] for j in idx]
            texts = [t for t, _ in batch]
            lab = torch.tensor([label_to_idx[l] for _, l in batch], device=device)

            ctok, cmask = tokenize(tokenizer, texts, device)
            outcome_texts = make_augmented_outcome_bank()
            otok, omask = tokenize(tokenizer, outcome_texts, device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                ctx_emb = model.encode_context(ctok, cmask)
                out_emb = model.encode_outcome(otok, omask)
                logits = model.compatibility(ctx_emb, out_emb)
                loss = ce(logits, lab)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            total_loss += loss.item() * len(idx)

        val_acc = run_eval(val_ex, base_outcome_texts)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  val_acc {val_acc:.4f}  "
              f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                   os.path.join(ckpt_dir, "jepa_v3_latest.pt"))

        if val_acc > best_val:
            best_val = val_acc
            no_improve_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       os.path.join(ckpt_dir, "jepa_v3_best.pt"))
            print(f"  -> new best val_acc {val_acc:.4f}, saved jepa_v3_best.pt", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "jepa_v3_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_acc {best_ckpt['val_acc']:.4f})", flush=True)

    test_acc = run_eval(test_ex, base_outcome_texts)
    print(f"test_acc (227-way, base descriptions): {test_acc:.4f}", flush=True)

    paraphrases = get_paraphrases()
    para_labels = [l for l in labels if l in paraphrases]
    para_outcome_texts = [paraphrases.get(l, base_description(raw_intent_name(l))) for l in labels]
    test_subset = [(t, l) for t, l in test_ex if l in paraphrases]

    para_acc = run_eval(test_subset, para_outcome_texts)
    base_acc_subset = run_eval(test_subset, base_outcome_texts)
    print(f"test_acc on {len(para_labels)}-intent subset, UNSEEN paraphrased descriptions: "
          f"{para_acc:.4f}  (n={len(test_subset)})", flush=True)
    print(f"  (same subset, base descriptions, for comparison): {base_acc_subset:.4f}", flush=True)

    in_scope_sim, oos_sim = run_oos_check(base_outcome_texts)
    print(f"\nOOS separation: mean max-similarity in-scope={in_scope_sim:.4f}  oos={oos_sim:.4f}  "
          f"gap={in_scope_sim - oos_sim:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
