"""Iteration 5 training: Multi-Question Poly-Encoder.

Trains three heterogeneous question types jointly from ONE context forward
pass each step: `intent` (Choice, same task as v4), `needs_human` (Bool,
heuristic label), `urgency` (Score, 5 ordinal bins, heuristic label). See
dataset_v5.py for the explicit caveat on the two heuristic labels.

Reuses the v4 fine-tuning stack (partial layer freezing, layer-wise LR
decay, 8-bit AdamW, gradient checkpointing, mixed precision, cosine
schedule with warmup, label smoothing, early stopping) since that stack
was validated in iteration 4.

New evaluation added specifically for this iteration: a timing benchmark
that directly checks the claimed benefit -- answering all 3 questions
about a batch of contexts in ONE forward pass should be meaningfully
faster than answering them with 3 separate forward passes over the same
contexts. This turns "it answers multiple questions in parallel" from an
architectural claim into a measured number.
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

from dataset_v5 import build_multi_question_dataset
from dataset_v4 import sample_outcome_description, base_description, raw_intent_name
from paraphrases_v4 import get_paraphrases
from model_v5 import MultiQuestionPolyEncoderV5, get_tokenizer, QUESTION_TYPES, QTYPE_TO_ID

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def tokenize(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def build_param_groups(model: MultiQuestionPolyEncoderV5, base_lr: float, decay: float = 0.9):
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

    head_params += list(model.needs_human_head.parameters())
    head_params += list(model.urgency_head.parameters())
    head_params.append(model.log_temperature)
    groups.append({"params": head_params, "lr": base_lr * 5})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--freeze_layers", type=int, default=16)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--intent_weight", type=float, default=1.0)
    parser.add_argument("--needs_human_weight", type=float, default=0.5)
    parser.add_argument("--urgency_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="Force 'cpu' or 'cuda' (for smoke testing)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bitsandbytes available: {HAS_BNB}", flush=True)

    data = build_multi_question_dataset()
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

    print(f"Train/val/test (seen intents): {len(train_ex)}/{len(val_ex)}/{len(test_ex)}  "
          f"| OOS test: {len(test_oos_ex)}  | zero-shot test: {len(test_zs_ex)}", flush=True)
    nh_pos_rate = sum(e["needs_human_label"] for e in train_ex) / len(train_ex)
    print(f"needs_human positive rate (train): {nh_pos_rate:.3f}  "
          f"-> majority-class baseline accuracy: {max(nh_pos_rate, 1 - nh_pos_rate):.3f}", flush=True)
    urg_counts = {i: sum(1 for e in train_ex if e["urgency_label"] == i) for i in range(1, 6)}
    urg_majority_bin = max(urg_counts, key=urg_counts.get)
    urg_majority_acc = urg_counts[urg_majority_bin] / len(train_ex)
    print(f"urgency bin counts (train): {urg_counts}  "
          f"-> majority-class baseline accuracy: {urg_majority_acc:.3f} (always predict bin {urg_majority_bin})",
          flush=True)

    # Both auxiliary tasks are heavily imbalanced -- without class weighting the
    # model could just always predict the majority class and look accurate
    # without learning anything. Inverse-frequency weighting keeps the loss
    # honest about the minority classes.
    nh_pos_weight = torch.tensor((1 - nh_pos_rate) / max(nh_pos_rate, 1e-6), device=device)
    urg_class_counts = torch.tensor([urg_counts[i] for i in range(1, 6)], dtype=torch.float, device=device)
    urg_class_weights = (urg_class_counts.sum() / (5 * urg_class_counts)).clamp(max=10.0)

    tokenizer = get_tokenizer()
    model = MultiQuestionPolyEncoderV5(freeze_layers=args.freeze_layers).to(device)
    model.enable_gradient_checkpointing()
    print(f"Model params: {model.num_params():,}  trainable: {model.num_trainable_params():,} "
          f"({100 * model.num_trainable_params() / model.num_params():.1f}%)", flush=True)

    param_groups = build_param_groups(model, args.base_lr)
    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)
        print("Using bitsandbytes 8-bit AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        print("Using standard fp32 AdamW", flush=True)

    ce_intent = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_urgency = nn.CrossEntropyLoss(weight=urg_class_weights, label_smoothing=0.1)
    bce_needs_human = nn.BCEWithLogitsLoss(pos_weight=nh_pos_weight)

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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    seed_counter = [0]

    def make_augmented_bank(label_list):
        rng = random.Random(3000 + seed_counter[0])
        seed_counter[0] += 1
        return [sample_outcome_description(raw_intent_name(l), rng) for l in label_list]

    base_seen_texts = [base_description(raw_intent_name(l)) for l in seen_labels]
    base_all_texts = [base_description(raw_intent_name(l)) for l in all_labels]

    INTENT_QID = QTYPE_TO_ID["intent"]
    NH_QID = QTYPE_TO_ID["needs_human"]
    URG_QID = QTYPE_TO_ID["urgency"]

    def run_eval(examples, label_list, outcome_texts_eval, idx_map, eval_bs=64):
        model.eval()
        otok, omask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)

        intent_correct = 0
        nh_correct = 0
        urg_correct = 0
        urg_abs_err = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [e["text"] for e in batch]
                intent_lab = torch.tensor([idx_map[e["intent_label"]] for e in batch], device=device)
                nh_lab = torch.tensor([float(e["needs_human_label"]) for e in batch], device=device)
                urg_lab = torch.tensor([e["urgency_label"] - 1 for e in batch], device=device)

                ctok, cmask = tokenize(tokenizer, texts, device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    shared = model.encode_context_all(ctok, cmask)  # (B, 3, hidden_dim), ONE forward pass
                    intent_logits = model.intent_logits(shared[:, INTENT_QID], out_emb)
                    nh_logits = model.needs_human_logits(shared[:, NH_QID])
                    urg_logits = model.urgency_logits(shared[:, URG_QID])

                intent_preds = intent_logits.argmax(dim=-1)
                nh_preds = (torch.sigmoid(nh_logits) > 0.5).float()
                urg_preds = urg_logits.argmax(dim=-1)

                intent_correct += (intent_preds == intent_lab).sum().item()
                nh_correct += (nh_preds == nh_lab).sum().item()
                urg_correct += (urg_preds == urg_lab).sum().item()
                urg_abs_err += (urg_preds - urg_lab).abs().sum().item()
                total += len(batch)

        model.train()
        return {
            "intent_acc": intent_correct / total,
            "needs_human_acc": nh_correct / total,
            "urgency_acc": urg_correct / total,
            "urgency_mae": urg_abs_err / total,
        }

    def run_oos_check(outcome_texts_eval, n_sample=1000, eval_bs=64):
        model.eval()
        otok, omask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)

        def max_logit(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [e["text"] for e in batch]
                    ctok, cmask = tokenize(tokenizer, texts, device)
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        shared = model.encode_context_all(ctok, cmask)
                        logits = model.intent_logits(shared[:, INTENT_QID], out_emb)
                    sims.append(logits.max(dim=-1).values)
            return torch.cat(sims)

        in_scope = max_logit(test_ex[:n_sample])
        oos = max_logit(test_oos_ex)
        model.train()
        return in_scope.mean().item(), oos.mean().item()

    def run_timing_benchmark(n_examples=256, eval_bs=32):
        """Directly measures the claimed benefit: answering all 3 question
        types in ONE forward pass vs. 3 SEPARATE forward passes over the
        same contexts (same total compute over the backbone otherwise)."""
        model.eval()
        examples = test_ex[:n_examples]
        texts_batches = [[e["text"] for e in examples[i:i + eval_bs]] for i in range(0, len(examples), eval_bs)]
        tok_batches = [tokenize(tokenizer, tb, device) for tb in texts_batches]

        # warmup
        with torch.no_grad():
            for ctok, cmask in tok_batches[:2]:
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    model.encode_context_all(ctok, cmask)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for ctok, cmask in tok_batches:
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    model.context_encoder(ctok, cmask, torch.arange(3, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_joint = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            for ctok, cmask in tok_batches:
                for qid in range(3):
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        model.context_encoder(ctok, cmask, torch.tensor([qid], device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_separate = time.time() - t0

        model.train()
        return t_joint, t_separate

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
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
            batch = [train_ex[j] for j in idx]
            texts = [e["text"] for e in batch]
            intent_lab = torch.tensor([seen_idx[e["intent_label"]] for e in batch], device=device)
            nh_lab = torch.tensor([float(e["needs_human_label"]) for e in batch], device=device)
            urg_lab = torch.tensor([e["urgency_label"] - 1 for e in batch], device=device)

            ctok, cmask = tokenize(tokenizer, texts, device)
            outcome_texts = make_augmented_bank(seen_labels)
            otok, omask = tokenize(tokenizer, outcome_texts, device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                shared = model.encode_context_all(ctok, cmask)   # ONE pass -> all 3 question read-outs
                out_emb = model.encode_outcome(otok, omask)

                intent_logits = model.intent_logits(shared[:, INTENT_QID], out_emb)
                nh_logits = model.needs_human_logits(shared[:, NH_QID])
                urg_logits = model.urgency_logits(shared[:, URG_QID])

                loss_intent = ce_intent(intent_logits, intent_lab)
                loss_nh = bce_needs_human(nh_logits, nh_lab)
                loss_urg = ce_urgency(urg_logits, urg_lab)
                loss = (args.intent_weight * loss_intent +
                        args.needs_human_weight * loss_nh +
                        args.urgency_weight * loss_urg) / args.grad_accum

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

        metrics = run_eval(val_ex, seen_labels, base_seen_texts, seen_idx)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[-1]
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  "
              f"val_intent_acc {metrics['intent_acc']:.4f}  val_needs_human_acc {metrics['needs_human_acc']:.4f}  "
              f"val_urgency_acc {metrics['urgency_acc']:.4f}  val_urgency_mae {metrics['urgency_mae']:.3f}  "
              f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        val_score = metrics["intent_acc"]  # primary task drives early stopping/checkpoint selection
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_score": val_score},
                   os.path.join(ckpt_dir, "jepa_v5_latest.pt"))

        if val_score > best_val:
            best_val = val_score
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_score": val_score},
                       os.path.join(ckpt_dir, "jepa_v5_best.pt"))
            print(f"  -> new best val_intent_acc {val_score:.4f}, saved jepa_v5_best.pt", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "jepa_v5_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_intent_acc "
          f"{best_ckpt['val_score']:.4f})", flush=True)

    test_metrics = run_eval(test_ex, seen_labels, base_seen_texts, seen_idx)
    test_nh_pos_rate = sum(e["needs_human_label"] for e in test_ex) / len(test_ex)
    test_nh_majority = max(test_nh_pos_rate, 1 - test_nh_pos_rate)
    test_urg_counts = {i: sum(1 for e in test_ex if e["urgency_label"] == i) for i in range(1, 6)}
    test_urg_majority = max(test_urg_counts.values()) / len(test_ex)
    print(f"test (seen intents, base descriptions): intent_acc={test_metrics['intent_acc']:.4f}  "
          f"needs_human_acc={test_metrics['needs_human_acc']:.4f} (majority-baseline={test_nh_majority:.4f})  "
          f"urgency_acc={test_metrics['urgency_acc']:.4f} (majority-baseline={test_urg_majority:.4f})  "
          f"urgency_mae={test_metrics['urgency_mae']:.3f}",
          flush=True)
    print("NOTE: needs_human / urgency are heuristic labels (see dataset_v5.py), not gold data -- "
          "these numbers show whether the model learned the RULE, not true urgency.", flush=True)

    paraphrases = get_paraphrases()
    paraphrases = {k: v for k, v in paraphrases.items() if k in seen_idx}
    para_texts = [paraphrases.get(l, base_description(raw_intent_name(l))) for l in seen_labels]
    test_para_subset = [e for e in test_ex if e["intent_label"] in paraphrases]
    para_metrics = run_eval(test_para_subset, seen_labels, para_texts, seen_idx)
    base_metrics_subset = run_eval(test_para_subset, seen_labels, base_seen_texts, seen_idx)
    print(f"\ntest, {len(paraphrases)}-intent subset, UNSEEN paraphrases: "
          f"intent_acc={para_metrics['intent_acc']:.4f}  (n={len(test_para_subset)})", flush=True)
    print(f"  same subset, base descriptions: intent_acc={base_metrics_subset['intent_acc']:.4f}", flush=True)

    zs_metrics = run_eval(test_zs_ex, all_labels, base_all_texts, all_idx)
    seen_mixed_metrics = run_eval(test_ex[:1500], all_labels, base_all_texts, all_idx)
    chance = 1.0 / len(all_labels)
    print(f"\nZERO-SHOT test ({len(zero_shot_labels)} never-trained intents, "
          f"candidate pool = all {len(all_labels)} intents): "
          f"intent_acc={zs_metrics['intent_acc']:.4f}  (n={len(test_zs_ex)})", flush=True)
    print(f"  (reference) seen-intent acc in same mixed pool: {seen_mixed_metrics['intent_acc']:.4f}  "
          f"| chance: {chance:.4f}", flush=True)

    in_scope_sim, oos_sim = run_oos_check(base_seen_texts)
    print(f"\nOOS separation: mean max-compat-logit  in-scope={in_scope_sim:.4f}  oos={oos_sim:.4f}  "
          f"gap={in_scope_sim - oos_sim:.4f}", flush=True)

    t_joint, t_separate = run_timing_benchmark()
    print(f"\nTiming (256 test contexts, {len(QUESTION_TYPES)} question types): "
          f"joint single-pass={t_joint:.3f}s  separate {len(QUESTION_TYPES)}x-pass={t_separate:.3f}s  "
          f"speedup={t_separate / t_joint:.2f}x", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
