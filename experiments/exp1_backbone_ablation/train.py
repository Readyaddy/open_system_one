"""Iteration 4 training: Poly-Encoder on all-roberta-large-v1 (355M/encoder),
combined CLINC150 + Banking77 + SNIPS (234 intents, ~38k examples before
the zero-shot holdout), with a real zero-shot generalization test.

DL techniques applied, each for a concrete reason:
  - Poly-encoder scoring (model_v4.py) instead of pure bi-encoder mean
    pooling -- more accurate compatibility scoring while staying mostly
    parallel/cacheable (Humeau et al., 2019).
  - Partial layer freezing (bottom 16/24 layers frozen) -- standard
    large-model fine-tuning practice; lower transformer layers hold
    generic language structure, only the top layers + poly head need to
    adapt to this task, and freezing the rest saves a large chunk of
    optimizer memory.
  - Layer-wise learning-rate decay on the trainable top layers -- layers
    closer to the frozen block get a smaller LR than layers closer to the
    (randomly initialized) task head, which typically stabilizes
    fine-tuning of deep transformers.
  - 8-bit AdamW (bitsandbytes) -- cuts optimizer state memory roughly 4x
    vs. fp32 AdamW, which is what makes a 355M-param backbone with an
    unfrozen top third fit in 12GB alongside activations.
  - Mixed precision (fp16 autocast + GradScaler) + gradient checkpointing
    on the backbone -- memory/speed, lets us afford a larger effective
    batch via gradient accumulation.
  - Label smoothing (0.1) on the compatibility softmax -- standard
    regularizer against over-confident wrong answers, useful here because
    many candidate outcomes are semantically very close (Banking77).
  - Outcome-description augmentation (carried over from iteration 3, the
    fix that actually closed the paraphrase gap) -- resampled every step
    from dataset_v4.sample_outcome_description.

Evaluation, four axes:
  1. In-scope test accuracy (seen intents, base descriptions).
  2. Paraphrase generalization (seen intents, unseen description wording).
  3. TRUE zero-shot generalization: intents with ZERO training examples
     and ZERO description exposure during training, tested against a
     candidate pool mixing seen + unseen intents.
  4. OOS separation (CLINC150's explicit out-of-scope examples).
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

from dataset_v4 import build_combined_dataset, sample_outcome_description, base_description, raw_intent_name
from paraphrases_v4 import get_paraphrases
from model import JEPAPolyEncoderV4, get_tokenizer  # EXPERIMENT 1: local model.py (e5-large-v2 backbone)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def tokenize(tokenizer, texts, device, max_length=32, prefix=""):
    # e5-large-v2 was contrastively trained with "query: " / "passage: "
    # prefixes distinguishing the two sides of a similarity pair -- since
    # our context is query-like (the thing being classified) and our
    # outcome descriptions are passage-like (the candidate being matched
    # against), this maps directly onto that convention and materially
    # affects e5's embedding quality per its model card.
    if prefix:
        texts = [prefix + t for t in texts]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def tokenize_context(tokenizer, texts, device, max_length=32):
    return tokenize(tokenizer, texts, device, max_length, prefix="query: ")


def tokenize_outcome(tokenizer, texts, device, max_length=32):
    return tokenize(tokenizer, texts, device, max_length, prefix="passage: ")


def build_param_groups(model: JEPAPolyEncoderV4, base_lr: float, decay: float = 0.9):
    """Layer-wise LR decay: trainable backbone layers closer to the frozen
    block get progressively smaller LR than layers near the output; the
    randomly-initialized heads (poly-attention, projections, temperature)
    get a higher LR since they start from scratch."""
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--freeze_layers", type=int, default=16)
    parser.add_argument("--n_codes", type=int, default=16)
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

    data = build_combined_dataset()
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
    print(f"Seen intents: {len(seen_labels)}  | zero-shot (held out) intents: {len(zero_shot_labels)}  "
          f"| total: {len(all_labels)}", flush=True)

    tokenizer = get_tokenizer()
    model = JEPAPolyEncoderV4(freeze_layers=args.freeze_layers, n_codes=args.n_codes).to(device)
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

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

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
        rng = random.Random(2000 + seed_counter[0])
        seed_counter[0] += 1
        return [sample_outcome_description(raw_intent_name(l), rng) for l in label_list]

    base_seen_texts = [base_description(raw_intent_name(l)) for l in seen_labels]
    base_all_texts = [base_description(raw_intent_name(l)) for l in all_labels]

    def run_eval(examples, label_list, outcome_texts_eval, idx_map, eval_bs=64):
        model.eval()
        otok, omask = tokenize_outcome(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [t for t, _ in batch]
                lab = torch.tensor([idx_map[l] for _, l in batch], device=device)
                ctok, cmask = tokenize_context(tokenizer, texts, device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    ctx_codes = model.encode_context(ctok, cmask)
                    logits = model.compatibility(ctx_codes, out_emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_oos_check(outcome_texts_eval, n_sample=1000, eval_bs=64):
        model.eval()
        otok, omask = tokenize_outcome(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)

        def max_sim(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [t for t, _ in batch]
                    ctok, cmask = tokenize_context(tokenizer, texts, device)
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        ctx_codes = model.encode_context(ctok, cmask)
                        logits = model.compatibility(ctx_codes, out_emb)
                    sims.append(logits.max(dim=-1).values)
            return torch.cat(sims)

        in_scope_sim = max_sim(test_ex[:n_sample])
        oos_sim = max_sim(test_oos_ex)
        model.train()
        return in_scope_sim.mean().item(), oos_sim.mean().item()

    ckpt_dir = os.path.dirname(__file__)
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
            texts = [t for t, _ in batch]
            lab = torch.tensor([seen_idx[l] for _, l in batch], device=device)

            ctok, cmask = tokenize_context(tokenizer, texts, device)
            outcome_texts = make_augmented_bank(seen_labels)
            otok, omask = tokenize_outcome(tokenizer, outcome_texts, device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                ctx_codes = model.encode_context(ctok, cmask)
                out_emb = model.encode_outcome(otok, omask)
                logits = model.compatibility(ctx_codes, out_emb)
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

        val_acc = run_eval(val_ex, seen_labels, base_seen_texts, seen_idx)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[-1]
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  val_acc {val_acc:.4f}  "
              f"lr {cur_lr:.2e}  ({dt:.1f}s)", flush=True)

        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                   os.path.join(ckpt_dir, "exp1_latest.pt"))

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       os.path.join(ckpt_dir, "exp1_best.pt"))
            print(f"  -> new best val_acc {val_acc:.4f}, saved exp1_best.pt", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.", flush=True)
                break

    print("\n=== Final evaluation (best checkpoint) ===", flush=True)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "exp1_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_acc {best_ckpt['val_acc']:.4f})", flush=True)

    test_acc = run_eval(test_ex, seen_labels, base_seen_texts, seen_idx)
    print(f"test_acc (seen intents, base descriptions): {test_acc:.4f}", flush=True)

    paraphrases = get_paraphrases()
    paraphrases = {k: v for k, v in paraphrases.items() if k in seen_idx}  # only ones actually trained on
    para_texts = [paraphrases.get(l, base_description(raw_intent_name(l))) for l in seen_labels]
    test_para_subset = [(t, l) for t, l in test_ex if l in paraphrases]
    para_acc = run_eval(test_para_subset, seen_labels, para_texts, seen_idx)
    base_acc_subset = run_eval(test_para_subset, seen_labels, base_seen_texts, seen_idx)
    print(f"test_acc, {len(paraphrases)}-intent subset, UNSEEN paraphrases: {para_acc:.4f}  "
          f"(n={len(test_para_subset)})", flush=True)
    print(f"  same subset, base descriptions: {base_acc_subset:.4f}", flush=True)

    # True zero-shot: candidate pool = ALL intents (seen + never-trained-on),
    # test examples = ONLY the never-trained-on intents.
    zs_acc = run_eval(test_zs_ex, all_labels, base_all_texts, all_idx)
    # For reference: same candidate pool, but scored on seen-intent test examples too.
    seen_in_mixed_pool_acc = run_eval(test_ex[:1500], all_labels, base_all_texts, all_idx)
    print(f"\nZERO-SHOT test_acc ({len(zero_shot_labels)} never-trained intents, "
          f"candidate pool = all {len(all_labels)} intents): {zs_acc:.4f}  (n={len(test_zs_ex)})", flush=True)
    print(f"  (reference) seen-intent test_acc in the SAME mixed {len(all_labels)}-way pool: "
          f"{seen_in_mixed_pool_acc:.4f}", flush=True)
    chance = 1.0 / len(all_labels)
    print(f"  chance level in a {len(all_labels)}-way pool: {chance:.4f}", flush=True)

    in_scope_sim, oos_sim = run_oos_check(base_seen_texts)
    print(f"\nOOS separation: mean max-compat-logit  in-scope={in_scope_sim:.4f}  oos={oos_sim:.4f}  "
          f"gap={in_scope_sim - oos_sim:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
