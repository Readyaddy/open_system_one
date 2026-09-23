"""exp7c -- learned halting + escalation (NOTES.md Sec 6). A separate
follow-on script, not part of train.py's staging (exp7a/7b/7d): it freezes
an already depth-conditioned checkpoint entirely and trains two small MLPs
(halting.HaltingHeads) on the frozen per-depth trajectories. See halting.py
for the feature/target definitions.

Pre-registered stopping rule (NOTES.md Sec 6, under exp7b):
  "If the curve is flat, stop here. Looping is a dead end for this task and
  exp7c is cancelled."
This script checks that live (via run_eval's per-depth accuracy) before
training anything, and refuses to proceed on a flat curve unless --force is
given -- a halting head trained on a flat curve has no real signal to learn
(every depth is equally right/wrong, so "should I stop here" is arbitrary),
and would produce a plausible-looking but meaningless compute-vs-accuracy
curve.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.amp import autocast

from model import HybridDecisionModel, PackedSequenceBuilder, get_tokenizer, BACKBONE
from halting import (HaltingHeads, compute_depth_features, compute_targets,
                      compute_speedup_curve, calibrate_escalation_threshold, N_FEATURES)
# NOTE: import train's symbols BEFORE `import data` -- data.py's own
# sys.path.insert(0, .../src) (needed for ITS OWN imports) re-shadows this
# directory's train.py with src/train.py (an unrelated older script that
# happens to share the name) if data has already been imported first. Once
# `train` is bound below, later re-imports are no-ops regardless of path order.
from train import (TaskMixer, build_mixer, run_eval, AUTOCAST_DTYPE, _INTENT_SEEN_LABELS)
import data as D


def flatness_check(depth_curve, min_range: float):
    rng = max(depth_curve) - min(depth_curve)
    is_flat = rng < min_range
    return is_flat, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, required=True,
                     help="Checkpoint of an already depth-conditioned model (k_max>1) to freeze "
                          "and distill halting/escalation heads from. Architecture flags below "
                          "must match how it was trained.")
    ap.add_argument("--backbone", type=str, default=BACKBONE)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--n_context_codes", type=int, default=16)
    ap.add_argument("--head_n_layers", type=int, default=2)
    ap.add_argument("--maxsim_dim", type=int, default=128)
    ap.add_argument("--use_maxsim", action="store_true")

    ap.add_argument("--budget_total", type=int, default=2048)
    ap.add_argument("--l_context", type=int, default=768)
    ap.add_argument("--l_instructions", type=int, default=96)
    ap.add_argument("--l_max_per_option", type=int, default=64)
    ap.add_argument("--option_chunk_size", type=int, default=64)

    ap.add_argument("--w_intent", type=float, default=0.30)
    ap.add_argument("--w_mcq", type=float, default=0.30)
    ap.add_argument("--w_bool", type=float, default=0.10)
    ap.add_argument("--w_score", type=float, default=0.10)
    ap.add_argument("--w_diversity", type=float, default=0.20)
    ap.add_argument("--mixer_seed", type=int, default=12345)
    ap.add_argument("--max_train_intent", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--base_lr", type=float, default=1e-3,
                     help="HaltingHeads are two tiny 6->48->1 MLPs -- no backbone/main-head "
                          "gradient at all, so a much higher LR than train.py's is fine and "
                          "converges faster.")
    ap.add_argument("--hidden_mult", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--alpha", type=float, default=0.1,
                     help="Target miss rate for the escalation head's split-conformal "
                          "threshold: at most this fraction of truly-wrong halted answers "
                          "should slip through unescalated, on exchangeable future data.")
    ap.add_argument("--val_n", type=int, default=400, help="For the flatness check and the "
                     "compute-vs-accuracy sweep.")
    ap.add_argument("--calib_n", type=int, default=600, help="Held-out split for the "
                     "escalation head's conformal calibration -- must be disjoint from "
                     "training and from --val_n.")

    ap.add_argument("--min_depth_range", type=float, default=0.02,
                     help="NOTES.md Sec 6's stopping rule: if max(depth_curve)-min(depth_curve) "
                          "is below this (default 2 points), the run refuses to proceed -- "
                          "there's no real per-example depth signal to learn a halting policy "
                          "from. Pass --force to override.")
    ap.add_argument("--force", action="store_true", help="Proceed even if the flatness check fails.")

    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--ckpt_prefix", type=str, default="exp7c_halting")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- data (same corpora as train.py; only intent/mcq/etc. needed for
    # the frozen model's forward pass -- no gradient touches it at all) ----
    intent = D.load_intent_corpus_minus_banking77()
    _INTENT_SEEN_LABELS[0] = intent.seen_labels
    intent_train = intent.train[:args.max_train_intent] if args.max_train_intent else intent.train
    mcq = D.load_mcq_corpus_reweighted()
    try:
        qqp_train, qqp_val = D.load_qqp_pairs()
    except FileNotFoundError:
        qqp_train, qqp_val = [], []

    def _try_load(fn, name):
        try:
            return fn()
        except FileNotFoundError:
            print(f"{name} not found on disk -- skipping.", flush=True)
            return None

    bool_corpus = _try_load(D.load_bool_corpus, "exp7_bool_corpus")
    score_corpus = _try_load(D.load_score_corpus, "exp7_score_corpus")
    diversity_corpus = _try_load(D.load_diversity_corpus, "exp7_diversity_corpus")

    mixer = build_mixer(
        args, intent_train, mcq["train"], qqp_train,
        bool_corpus["train"] if bool_corpus else [],
        score_corpus["train"] if score_corpus else [],
        diversity_corpus["train"] if diversity_corpus else [],
    )

    # ---- frozen model ----
    tokenizer = get_tokenizer(args.backbone)
    model = HybridDecisionModel(
        backbone=args.backbone, mask_token_id=tokenizer.mask_token_id,
        n_context_codes=args.n_context_codes, k_max=args.k_max, head_n_layers=args.head_n_layers,
        maxsim_dim=args.maxsim_dim, use_maxsim=args.use_maxsim, gradient_checkpointing=False,
    ).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"Loaded and FULLY froze {args.resume} (epoch {ckpt['epoch']}) -- "
          f"only HaltingHeads (halt_head + escalate_head) will train.", flush=True)

    builder = PackedSequenceBuilder(tokenizer, budget_total=args.budget_total, l_context=args.l_context,
                                     l_instructions=args.l_instructions, l_max_per_option=args.l_max_per_option)

    # ---- flatness check (NOTES.md Sec 6's own stopping rule) ----
    val_subset = intent.val[:args.val_n]
    rng = random.Random(999)
    val_packed = [D.build_intent_example(t, l, intent.seen_labels, rng, n_target=50) for t, l in val_subset]
    depth_curve = run_eval(model, builder, tokenizer, device, val_packed, batch_size=8,
                            k=args.k_max, option_chunk_size=args.option_chunk_size)
    is_flat, curve_range = flatness_check(depth_curve, args.min_depth_range)
    print(f"Depth curve (val, k=1..{args.k_max}): {[f'{v:.4f}' for v in depth_curve]}  "
          f"range={curve_range:.4f}", flush=True)
    if is_flat and not args.force:
        print(f"\nSTOPPING: depth-curve range {curve_range:.4f} < --min_depth_range "
              f"{args.min_depth_range:.4f}. NOTES.md Sec 6: 'If the curve is flat, stop here. "
              f"Looping is a dead end for this task and exp7c is cancelled.' A halting head "
              f"trained on this would have no real per-example signal -- every depth is about "
              f"equally right, so 'should I stop here' is arbitrary, and the compute-vs-accuracy "
              f"curve it produces would look plausible without meaning anything. Re-run with "
              f"--force to proceed anyway.", flush=True)
        return
    elif is_flat:
        print("--force given: proceeding despite a flat depth curve. Treat the compute-vs-"
              "accuracy curve this run produces as diagnostic only, not a real deliverable.",
              flush=True)

    # ---- halting heads ----
    heads = HaltingHeads(hidden_mult=args.hidden_mult, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.base_lr)
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"HaltingHeads params: {n_params} (halt_head + escalate_head, {N_FEATURES}-dim features)",
          flush=True)

    ckpt_dir = args.ckpt_dir or os.path.dirname(__file__)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n=== Training halting/escalation heads, {args.epochs} epochs x "
          f"{args.steps_per_epoch} steps ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        heads.train()
        running_halt_loss, running_esc_loss = 0.0, 0.0
        for step in range(args.steps_per_epoch):
            examples = mixer.sample_batch(args.batch_size)
            batch = builder.build_batch(examples, device)
            with torch.no_grad():
                with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    logits_per_depth = model(tokenizer, batch, device, k=args.k_max,
                                              option_chunk_size=args.option_chunk_size)
                logits_per_depth = [l.float() for l in logits_per_depth]
                feats_per_depth, probs_per_depth = compute_depth_features(logits_per_depth, batch.valid_mask)
                halt_targets, escalate_targets = compute_targets(probs_per_depth, batch.answer_idx)

            halt_loss = 0.0
            esc_loss = 0.0
            for feats, ht, et in zip(feats_per_depth, halt_targets, escalate_targets):
                halt_logit, esc_logit = heads(feats)
                halt_loss = halt_loss + F.binary_cross_entropy_with_logits(halt_logit, ht)
                esc_loss = esc_loss + F.binary_cross_entropy_with_logits(esc_logit, et)
            halt_loss = halt_loss / args.k_max
            esc_loss = esc_loss / args.k_max
            loss = halt_loss + esc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_halt_loss += halt_loss.item()
            running_esc_loss += esc_loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == args.steps_per_epoch:
                print(f"  epoch {epoch}/{args.epochs}  step {step+1}/{args.steps_per_epoch}  "
                      f"halt_loss {running_halt_loss/(step+1):.4f}  esc_loss {running_esc_loss/(step+1):.4f}",
                      flush=True)
        print(f"epoch {epoch}/{args.epochs}  halt_loss {running_halt_loss/args.steps_per_epoch:.4f}  "
              f"esc_loss {running_esc_loss/args.steps_per_epoch:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- compute-vs-accuracy sweep + conformal calibration, on a fresh held-out split ----
    heads.eval()
    calib_subset = intent.val[args.val_n:args.val_n + args.calib_n]
    rng2 = random.Random(1000)
    calib_packed = [D.build_intent_example(t, l, intent.seen_labels, rng2, n_target=50) for t, l in calib_subset]

    all_feats_per_depth = [[] for _ in range(args.k_max)]
    all_probs_per_depth = [[] for _ in range(args.k_max)]
    all_answer_idx = []
    with torch.no_grad():
        for i in range(0, len(calib_packed), 8):
            batch_ex = calib_packed[i:i + 8]
            batch = builder.build_batch(batch_ex, device)
            with autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                logits_per_depth = model(tokenizer, batch, device, k=args.k_max,
                                          option_chunk_size=args.option_chunk_size)
            logits_per_depth = [l.float() for l in logits_per_depth]
            feats_per_depth, probs_per_depth = compute_depth_features(logits_per_depth, batch.valid_mask)
            for d in range(args.k_max):
                all_feats_per_depth[d].append(feats_per_depth[d])
                all_probs_per_depth[d].append(probs_per_depth[d])
            all_answer_idx.append(batch.answer_idx)
    feats_per_depth = [torch.cat(x, dim=0) for x in all_feats_per_depth]
    probs_per_depth = [torch.cat(x, dim=0) for x in all_probs_per_depth]
    answer_idx = torch.cat(all_answer_idx, dim=0)

    curve = compute_speedup_curve(feats_per_depth, probs_per_depth, heads, answer_idx)
    print("\n=== Compute-vs-accuracy sweep (held-out calibration split) ===", flush=True)
    for row in curve:
        print(f"  theta={row['theta']:.2f}  avg_depth={row['avg_depth']:.2f}/{row['of_k_max']}  "
              f"accuracy={row['accuracy']:.4f}", flush=True)

    # Pick the theta closest to a reasonable default (0.8) for the conformal
    # calibration pass below -- the sweep above is what a person actually
    # picks theta from; this just needs SOME fixed policy to calibrate escalation for.
    calib_theta = 0.8
    from halting import halted_depth_and_prob
    halted, escalate_prob_at_halt = halted_depth_and_prob(feats_per_depth, heads, calib_theta)
    idx = torch.arange(halted.size(0), device=halted.device)
    chosen_pred = torch.stack([p.argmax(-1) for p in probs_per_depth], dim=1)[idx, halted]
    is_wrong = chosen_pred != answer_idx
    tau, n_wrong = calibrate_escalation_threshold(escalate_prob_at_halt[is_wrong], alpha=args.alpha)
    print(f"\n=== Escalation threshold (split conformal, alpha={args.alpha}, theta={calib_theta}) ===",
          flush=True)
    print(f"  calibration set: {halted.size(0)} examples, {n_wrong} actually wrong at their halted depth",
          flush=True)
    print(f"  tau={tau:.4f} -- flag for escalation whenever escalate_head's sigmoid output >= tau. "
          f"Guarantees >= {(1-args.alpha)*100:.0f}% of truly-wrong halted answers get flagged, "
          f"on exchangeable future data.", flush=True)

    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt_prefix}.pt")
    torch.save({
        "halting_heads_state": heads.state_dict(),
        "source_checkpoint": args.resume,
        "k_max": args.k_max,
        "depth_curve": depth_curve,
        "speedup_curve": curve,
        "calib_theta": calib_theta,
        "escalate_tau": tau,
        "alpha": args.alpha,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nSaved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
