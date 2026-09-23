"""Experiment 3 training: joint-sequence, marker-token readout model on
ModernBERT-large. See NOTES.md for the full architectural rationale.

Note this model has ONE shared backbone (not two separate encoders like
the poly-encoder family), since state and all questions/options are
packed into a single sequence and processed together -- so the
fine-tuning stack (layer-wise LR decay etc.) is simpler here: one set of
transformer layers, not two.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import math
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from dataset import build_request_dataset
from paraphrases_v4 import get_paraphrases
from model import (JointSequenceModelV6, build_tokenizer, build_sequence_text,
                    score_expected_value, MARKER_TOKENS)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def tokenize_requests(tokenizer, requests, device, max_length=384):
    texts = [build_sequence_text(r) for r in requests]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def build_param_groups(model: JointSequenceModelV6, base_lr: float, decay: float = 0.9):
    groups = []
    trainable_layers = [layer for layer in model.backbone.layers if
                         any(p.requires_grad for p in layer.parameters())]
    n = len(trainable_layers)
    for i, layer in enumerate(trainable_layers):
        lr = base_lr * (decay ** (n - 1 - i))
        groups.append({"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr})

    head_params = [p for n_, p in model.named_parameters()
                    if p.requires_grad and "backbone.layers" not in n_]
    groups.append({"params": head_params, "lr": base_lr * 5})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--freeze_layers", type=int, default=20)
    parser.add_argument("--n_choice_options", type=int, default=6)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--choice_weight", type=float, default=1.0)
    parser.add_argument("--noul_weight", type=float, default=0.5)
    parser.add_argument("--score_weight", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bitsandbytes available: {HAS_BNB}", flush=True)

    data = build_request_dataset(n_choice_options=args.n_choice_options)
    train_req, val_req, test_req, zs_req = data["train"], data["val"], data["test"], data["test_zero_shot"]
    if args.max_train:
        train_req = train_req[:args.max_train]
    if args.max_val:
        val_req = val_req[:args.max_val]

    nh_pos_rate = sum(r["noul_label"] for r in train_req) / len(train_req)
    print(f"Train/val/test/zero-shot requests: {len(train_req)}/{len(val_req)}/{len(test_req)}/{len(zs_req)}",
          flush=True)
    print(f"needs_human positive rate: {nh_pos_rate:.3f} "
          f"-> majority baseline {max(nh_pos_rate, 1 - nh_pos_rate):.3f}", flush=True)

    tokenizer = build_tokenizer()
    model = JointSequenceModelV6(tokenizer, freeze_layers=args.freeze_layers,
                                  n_choice_options=args.n_choice_options).to(device)
    model.enable_gradient_checkpointing()
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,}", flush=True)

    param_groups = build_param_groups(model, args.base_lr)
    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)
        print("Using bitsandbytes 8-bit AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    ce_choice = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_score = nn.CrossEntropyLoss(label_smoothing=0.1)
    nh_pos_weight = torch.tensor((1 - nh_pos_rate) / max(nh_pos_rate, 1e-6), device=device)
    bce_noul = nn.BCEWithLogitsLoss(pos_weight=nh_pos_weight)

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

    def run_eval(requests, eval_bs=32):
        model.eval()
        choice_correct = nh_correct = score_correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(requests), eval_bs):
                batch = requests[i:i + eval_bs]
                input_ids, attn = tokenize_requests(tokenizer, batch, device)
                choice_lab = torch.tensor([r["choice_correct_idx"] for r in batch], device=device)
                nh_lab = torch.tensor([r["noul_label"] for r in batch], device=device)
                score_lab = torch.tensor([r["score_correct_idx"] for r in batch], device=device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    hidden = model(input_ids, attn)
                    choice_logits = model.read_choice(hidden, input_ids)
                    score_logits = model.read_score(hidden, input_ids)
                    noul_logits = model.read_noul(hidden, input_ids)

                choice_correct += (choice_logits.argmax(-1) == choice_lab).sum().item()
                nh_correct += ((torch.sigmoid(noul_logits) > 0.5).float() == nh_lab).sum().item()
                score_correct += (score_logits.argmax(-1) == score_lab).sum().item()
                total += len(batch)
        model.train()
        return {"choice_acc": choice_correct / total, "noul_acc": nh_correct / total,
                "score_acc": score_correct / total}

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
            input_ids, attn = tokenize_requests(tokenizer, batch, device)
            choice_lab = torch.tensor([r["choice_correct_idx"] for r in batch], device=device)
            nh_lab = torch.tensor([r["noul_label"] for r in batch], device=device)
            score_lab = torch.tensor([r["score_correct_idx"] for r in batch], device=device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                hidden = model(input_ids, attn)
                choice_logits = model.read_choice(hidden, input_ids)
                score_logits = model.read_score(hidden, input_ids)
                noul_logits = model.read_noul(hidden, input_ids)

                loss = (args.choice_weight * ce_choice(choice_logits, choice_lab) +
                        args.score_weight * ce_score(score_logits, score_lab) +
                        args.noul_weight * bce_noul(noul_logits, nh_lab)) / args.grad_accum

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

        metrics = run_eval(val_req)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[-1]
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  "
              f"val_choice_acc {metrics['choice_acc']:.4f}  val_noul_acc {metrics['noul_acc']:.4f}  "
              f"val_score_acc {metrics['score_acc']:.4f}  lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        val_score = metrics["choice_acc"]
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_score": val_score},
                   os.path.join(ckpt_dir, "exp3_latest.pt"))
        if val_score > best_val:
            best_val = val_score
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_score": val_score},
                       os.path.join(ckpt_dir, "exp3_best.pt"))
            print(f"  -> new best val_choice_acc {val_score:.4f}, saved exp3_best.pt", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "exp3_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_choice_acc "
          f"{best_ckpt['val_score']:.4f})", flush=True)

    test_metrics = run_eval(test_req)
    print(f"test: choice_acc={test_metrics['choice_acc']:.4f}  noul_acc={test_metrics['noul_acc']:.4f}  "
          f"score_acc={test_metrics['score_acc']:.4f}", flush=True)

    paraphrases = get_paraphrases()
    para_subset = []
    for r in test_req:
        if r["choice_correct_label"] in paraphrases:
            r2 = dict(r)
            r2["choice_options"] = list(r["choice_options"])
            r2["choice_options"][r["choice_correct_idx"]] = paraphrases[r["choice_correct_label"]]
            para_subset.append(r2)
    if para_subset:
        para_metrics = run_eval(para_subset)
        base_subset_metrics = run_eval([r for r in test_req if r["choice_correct_label"] in paraphrases])
        print(f"\ntest, {len(para_subset)} requests, UNSEEN paraphrase in correct slot: "
              f"choice_acc={para_metrics['choice_acc']:.4f}", flush=True)
        print(f"  same subset, base descriptions: choice_acc={base_subset_metrics['choice_acc']:.4f}", flush=True)

    zs_metrics = run_eval(zs_req)
    chance = 1.0 / args.n_choice_options
    print(f"\nZERO-SHOT test ({len(zs_req)} requests, never-trained correct intent among "
          f"{args.n_choice_options} options): choice_acc={zs_metrics['choice_acc']:.4f}  "
          f"| chance: {chance:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
