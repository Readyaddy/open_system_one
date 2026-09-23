"""Experiment 2 training: poly-encoder (identical architecture and
fine-tuning stack to iteration 4), trained on small per-example candidate
sets instead of one shared 199-way bank. See NOTES.md for the full
rationale and what this isolates vs. iteration 4 and experiment 1.

Mechanical difference from iteration 4's training loop: outcome encoding
now produces a DIFFERENT small candidate bank per example in the batch
(not one bank shared across the whole batch), so scoring uses
`compatibility_per_example` (added to model.py) instead of `compatibility`.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import math
import time
import random
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from dataset import build_request_dataset
from paraphrases_v4 import get_paraphrases
from model import JEPAPolyEncoderV4, get_tokenizer

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def tokenize(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def encode_batch_options(model, tokenizer, requests, device, n_options):
    """Flattens every request's option texts into one big tokenize+encode
    call (efficient), then reshapes back to (B, n_options, D)."""
    flat_texts = [t for r in requests for t in r["option_texts"]]
    otok, omask = tokenize(tokenizer, flat_texts, device)
    emb = model.encode_outcome(otok, omask)  # (B*n_options, D)
    return emb.view(len(requests), n_options, -1)


def build_param_groups(model: JEPAPolyEncoderV4, base_lr: float, decay: float = 0.9):
    groups = []
    head_params = []
    for encoder in [model.context_encoder, model.outcome_encoder]:
        trainable_layers = [layer for layer in encoder.backbone.encoder.layer if
                             any(p.requires_grad for p in layer.parameters())]
        n = len(trainable_layers)
        for i, layer in enumerate(trainable_layers):
            lr = base_lr * (decay ** (n - 1 - i))
            groups.append({"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr})
        head_params += [p for n_, p in encoder.named_parameters()
                         if p.requires_grad and "encoder.layer" not in n_]
    head_params.append(model.log_temperature)
    groups.append({"params": head_params, "lr": base_lr * 5})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--freeze_layers", type=int, default=16)
    parser.add_argument("--n_options", type=int, default=6)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bitsandbytes available: {HAS_BNB}", flush=True)

    data = build_request_dataset(n_options=args.n_options)
    train_req, val_req = data["train"], data["val"]
    test_req, oos_req, zs_req = data["test"], data["test_oos"], data["test_zero_shot"]
    if args.max_train:
        train_req = train_req[:args.max_train]
    if args.max_val:
        val_req = val_req[:args.max_val]

    print(f"Train/val/test requests: {len(train_req)}/{len(val_req)}/{len(test_req)}  "
          f"| OOS requests: {len(oos_req)}  | zero-shot requests: {len(zs_req)}  "
          f"| options/request: {args.n_options}", flush=True)

    tokenizer = get_tokenizer()
    model = JEPAPolyEncoderV4(freeze_layers=args.freeze_layers).to(device)
    model.enable_gradient_checkpointing()
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,}", flush=True)

    param_groups = build_param_groups(model, args.base_lr)
    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)
        print("Using bitsandbytes 8-bit AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    batch_size = args.batch_size
    epochs = args.epochs
    n = len(train_req)
    micro_steps_per_epoch = math.ceil(n / batch_size)
    opt_steps_per_epoch = math.ceil(micro_steps_per_epoch / args.grad_accum)
    total_opt_steps = opt_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_opt_steps * 0.05))

    def lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    def run_eval(requests, eval_bs=48):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(requests), eval_bs):
                batch = requests[i:i + eval_bs]
                texts = [r["text"] for r in batch]
                lab = torch.tensor([r["correct_idx"] for r in batch], device=device)
                ctok, cmask = tokenize(tokenizer, texts, device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    ctx_codes = model.encode_context(ctok, cmask)
                    out_emb = encode_batch_options(model, tokenizer, batch, device, args.n_options)
                    logits = model.compatibility_per_example(ctx_codes, out_emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_oos_check(requests, in_scope_requests, eval_bs=48):
        model.eval()

        def top_logit(reqs):
            vals = []
            with torch.no_grad():
                for i in range(0, len(reqs), eval_bs):
                    batch = reqs[i:i + eval_bs]
                    texts = [r["text"] for r in batch]
                    ctok, cmask = tokenize(tokenizer, texts, device)
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        ctx_codes = model.encode_context(ctok, cmask)
                        out_emb = encode_batch_options(model, tokenizer, batch, device, args.n_options)
                        logits = model.compatibility_per_example(ctx_codes, out_emb)
                    vals.append(logits.max(dim=-1).values)
            return torch.cat(vals)

        in_scope = top_logit(in_scope_requests[:1000])
        oos = top_logit(requests)
        model.train()
        return in_scope.mean().item(), oos.mean().item()

    ckpt_dir = os.path.dirname(__file__)
    best_val = -1.0
    no_improve = 0

    print("\n=== Training ===", flush=True)
    opt_step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n).tolist()
        total_loss = 0.0
        optimizer.zero_grad()

        for micro_i, i in enumerate(range(0, n, batch_size)):
            idx = perm[i:i + batch_size]
            batch = [train_req[j] for j in idx]
            texts = [r["text"] for r in batch]
            lab = torch.tensor([r["correct_idx"] for r in batch], device=device)

            ctok, cmask = tokenize(tokenizer, texts, device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                ctx_codes = model.encode_context(ctok, cmask)
                out_emb = encode_batch_options(model, tokenizer, batch, device, args.n_options)
                logits = model.compatibility_per_example(ctx_codes, out_emb)
                loss = ce(logits, lab) / args.grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum * len(idx)

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

        val_acc = run_eval(val_req)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[-1]
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  val_acc {val_acc:.4f}  "
              f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                   os.path.join(ckpt_dir, "exp2_latest.pt"))
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       os.path.join(ckpt_dir, "exp2_best.pt"))
            print(f"  -> new best val_acc {val_acc:.4f}, saved exp2_best.pt", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "exp2_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_acc {best_ckpt['val_acc']:.4f})", flush=True)

    test_acc = run_eval(test_req)
    print(f"test_acc (small option sets, base descriptions): {test_acc:.4f}", flush=True)

    paraphrases = get_paraphrases()
    para_subset = []
    for r in test_req:
        if r["correct_label"] in paraphrases:
            r2 = dict(r)
            r2["option_texts"] = list(r["option_texts"])
            r2["option_texts"][r["correct_idx"]] = paraphrases[r["correct_label"]]
            para_subset.append(r2)
    if para_subset:
        para_acc = run_eval(para_subset)
        base_subset_acc = run_eval([r for r in test_req if r["correct_label"] in paraphrases])
        print(f"test_acc, {len(para_subset)} requests, UNSEEN paraphrase swapped into correct slot: "
              f"{para_acc:.4f}", flush=True)
        print(f"  same subset, base descriptions: {base_subset_acc:.4f}", flush=True)

    zs_acc = run_eval(zs_req)
    chance = 1.0 / args.n_options
    print(f"\nZERO-SHOT test_acc ({len(zs_req)} requests, never-trained correct intent "
          f"among {args.n_options} sampled options): {zs_acc:.4f}  | chance: {chance:.4f}", flush=True)

    in_scope_logit, oos_logit = run_oos_check(oos_req, test_req)
    print(f"\nOOS separation: mean top-candidate logit  in-scope={in_scope_logit:.4f}  "
          f"oos={oos_logit:.4f}  gap={in_scope_logit - oos_logit:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
