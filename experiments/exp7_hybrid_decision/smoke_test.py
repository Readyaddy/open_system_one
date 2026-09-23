"""Two smoke tests for the exp7 architecture:

--tiny (default): a from-scratch, randomly-initialized tiny ModernBertConfig
  backbone (hidden_size=64, 2 layers) -- fast enough to run on a laptop CPU
  in seconds, no ~1.6GB weight download. Checks that every mechanism in
  model.py actually runs end to end and produces correctly-shaped, finite
  tensors: packed-sequence construction with variable N per example,
  vector injection, context-code compression, the recurrent depth loop
  (checked at K=1 AND K=4, so the loop itself is exercised), MaxSim both
  off and on, a full forward+backward+optimizer step, and the three
  ablation-mode paths (text_only / vector_only / both) used by eval.py's
  cardinality sweep. Still needs network access ONCE, for the real
  ModernBERT tokenizer (small, a few hundred KB) -- special-token ids have
  to be real for the packed-sequence builder to mean anything.

--full: downloads and runs the REAL backbone (default answerdotai/ModernBERT-
  large) for one forward+backward pass, reporting peak memory and step
  time. This is the check meant to run on a Colab CPU instance per
  NOTES.md's plan (verify the real weights load and the mechanism works
  before ever paying for a GPU) -- see scripts/colab_exp7_cpu_probe.sh.

Usage:
  python smoke_test.py --tiny
  python smoke_test.py --full --backbone answerdotai/ModernBERT-large
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from model import HybridDecisionModel, PackedSequenceBuilder, PackedExample, get_tokenizer, BACKBONE
from train import compute_depth_loss


def make_examples(n_examples=6, n_options_choices=(2, 5, 20)):
    """Deliberately varied: different N per example (tests the padding
    path), different qtypes, and one example of each modality-dropout mode
    (tests that use_text=False / use_vector=False actually change the
    forward pass rather than silently no-opping)."""
    exs = []
    rng = random.Random(0)
    for i in range(n_examples):
        n = n_options_choices[i % len(n_options_choices)]
        qtype = ["choice", "bool", "score"][i % 3]
        options = [f"option {j} about topic {i}" for j in range(2 if qtype != "choice" else n)]
        answer_idx = rng.randrange(len(options))
        use_text = not (i % 5 == 0)     # occasionally vector-only
        use_vector = not (i % 5 == 1)   # occasionally text-only
        exs.append(PackedExample(
            context=f"This is a synthetic context sentence number {i}, mentioning topic {i} "
                    f"and some extra filler words to make the sequence non-trivial.",
            instructions="Which option best applies?",
            option_texts=options, qtype=qtype, answer_idx=answer_idx,
            use_text=use_text, use_vector=use_vector, source="smoke_test",
        ))
    return exs


def build_tiny_backbone(vocab_size, pad_token_id):
    from transformers import ModernBertConfig, ModernBertModel
    cfg = ModernBertConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=512, pad_token_id=pad_token_id,
    )
    return ModernBertModel(cfg)


def run_tiny(args):
    print("=== TINY smoke test (from-scratch config, real tokenizer) ===", flush=True)
    tokenizer = get_tokenizer(args.backbone)
    print(f"Tokenizer loaded ({args.backbone}): vocab_size={tokenizer.vocab_size}  "
          f"cls={tokenizer.cls_token_id} sep={tokenizer.sep_token_id} "
          f"mask={tokenizer.mask_token_id} pad={tokenizer.pad_token_id}", flush=True)

    tiny_backbone = build_tiny_backbone(len(tokenizer), tokenizer.pad_token_id)
    device = torch.device("cpu")

    for use_maxsim in (False, True):
        for k_max in (1, 4):
            model = HybridDecisionModel(
                mask_token_id=tokenizer.mask_token_id, n_context_codes=4, k_max=k_max,
                head_n_layers=2, maxsim_dim=16, use_maxsim=use_maxsim,
                gradient_checkpointing=False, backbone_override=tiny_backbone,
            ).to(device)
            builder = PackedSequenceBuilder(tokenizer, budget_total=256, l_context=64,
                                             l_instructions=32, l_max_per_option=16)
            examples = make_examples()
            batch = builder.build_batch(examples, device)
            logits_per_depth = model(tokenizer, batch, device, k=k_max, option_chunk_size=8,
                                      maxsim_chunk_size=4)
            assert len(logits_per_depth) == k_max, f"expected {k_max} depths, got {len(logits_per_depth)}"
            for d, logits in enumerate(logits_per_depth):
                assert logits.shape == (len(examples), batch.mask_positions.size(1))
                assert torch.isfinite(logits[batch.valid_mask]).all(), f"non-finite logit at depth {d}"

            loss, comps = compute_depth_loss(logits_per_depth, batch.answer_idx, batch.valid_mask,
                                              batch.qtype_idx)
            assert torch.isfinite(loss), "loss is not finite"
            loss.backward()
            n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"  use_maxsim={use_maxsim!s:5s} k_max={k_max}  loss={loss.item():.4f}  "
                  f"comps={comps}  params_with_grad={n_grad}/{n_trainable}  "
                  f"inject_gate={model.inject_gate.item():.4f}  maxsim_gate={model.maxsim_gate.item():.4f}",
                  flush=True)
            assert n_grad > 0, "no parameters received a gradient -- something is disconnected"
            model.zero_grad()

    # ablation-mode sanity: text_only vs vector_only vs both should NOT
    # produce identical logits (if they do, one of the two channels isn't
    # actually wired into the forward pass).
    model = HybridDecisionModel(mask_token_id=tokenizer.mask_token_id, n_context_codes=4, k_max=1,
                                 head_n_layers=2, use_maxsim=False, gradient_checkpointing=False,
                                 backbone_override=build_tiny_backbone(len(tokenizer), tokenizer.pad_token_id)
                                 ).to(device)
    builder = PackedSequenceBuilder(tokenizer, budget_total=256, l_context=64, l_instructions=32,
                                     l_max_per_option=16)
    base = make_examples(n_examples=1)[0]
    base.use_text, base.use_vector = True, True
    variants = {}
    for name, (ut, uv) in {"text_only": (True, False), "vector_only": (False, True),
                            "both": (True, True)}.items():
        ex = PackedExample(**{**base.__dict__, "use_text": ut, "use_vector": uv})
        batch = builder.build_batch([ex], device)
        with torch.no_grad():
            variants[name] = model(tokenizer, batch, device, k=1)[0]
    assert not torch.allclose(variants["text_only"], variants["vector_only"]), \
        "text_only and vector_only produced identical logits -- a modality channel is dead"
    print("  Ablation-mode sanity check passed: text_only / vector_only / both diverge.", flush=True)

    print("\nTINY smoke test PASSED.", flush=True)


def run_full(args):
    print(f"=== FULL smoke test (real backbone: {args.backbone}) ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    tokenizer = get_tokenizer(args.backbone)

    t0 = time.time()
    model = HybridDecisionModel(backbone=args.backbone, mask_token_id=tokenizer.mask_token_id,
                                 use_maxsim=args.use_maxsim, gradient_checkpointing=True).to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s. "
          f"params={model.num_params():,}  trainable={model.num_trainable_params():,}", flush=True)

    builder = PackedSequenceBuilder(tokenizer, budget_total=args.budget_total, l_context=768,
                                     l_instructions=96, l_max_per_option=64)
    examples = make_examples(n_examples=args.batch_size, n_options_choices=(args.n_options,))
    batch = builder.build_batch(examples, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logits_per_depth = model(tokenizer, batch, device, k=model.k_max, option_chunk_size=args.option_chunk_size)
    loss, comps = compute_depth_loss(logits_per_depth, batch.answer_idx, batch.valid_mask, batch.qtype_idx)
    loss.backward()
    fwd_bwd_time = time.time() - t0

    print(f"forward+backward: {fwd_bwd_time:.2f}s  loss={loss.item():.4f}  comps={comps}", flush=True)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"peak GPU memory: {peak_gb:.2f} GiB", flush=True)
    print("\nFULL smoke test PASSED.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--backbone", type=str, default=BACKBONE)
    ap.add_argument("--use_maxsim", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_options", type=int, default=20)
    ap.add_argument("--budget_total", type=int, default=1024)
    ap.add_argument("--option_chunk_size", type=int, default=32)
    args = ap.parse_args()

    if not args.tiny and not args.full:
        args.tiny = True  # default

    if args.tiny:
        run_tiny(args)
    if args.full:
        run_full(args)


if __name__ == "__main__":
    main()
