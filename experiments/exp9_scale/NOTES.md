# Experiment 9 — the architecture at ~5B parameters, on Colab

Written before the run (2026-09-25). Results are appended at the bottom.

## Questions this run answers

1. **Does injecting the option vector into the slot work?** exp7a's inject gate
   stayed at ~0.02 (the vector path was barely used). exp9 injects Qwen3-Embedding's
   own pretrained sentence embedding and measures a **modality × cardinality
   sweep** on held-out Banking77 (N = 4 / 16 / 71) and zero-shot intents (N = 50):
   text-only vs vector-only vs both.
2. **Does "rethinking" (recurrent depth) help at 4B scale?** Every pass is
   supervised, and every eval logs the accuracy-by-depth curve.
3. **Does a 10× larger backbone move held-out transfer**, which stayed flat
   through exp8 (AG News, Emotion, Banking77, typed-decisions)?

## Model (≈4.9B parameters)

| Part | Setting |
|---|---|
| Backbone | Qwen3-Embedding-4B (4.0B; top open embedding family, Apache-2.0), LoRA r=64 on all linear layers (`--full_finetune` available) |
| Packing | shared prefix (instructions + context) + one **isolated branch per option**: each branch sees the prefix and itself only, and all branches share position ids → option order cannot matter. Verified exact vs independent runs (`test_branch_mask.py`, 1.8e-6). |
| Vector path | Qwen3-Embedding's pretrained last-token embedding of the full option text (adapters off, no grad) → 2-layer projector → norm-matched → gate (init 0.1, own LR) → added at the branch's MARK token |
| Readout | each branch's final EOS, the position the embedding model pools at |
| Context pooling | Perceiver resampler: 128 latents × 2 layers (was 16 codes × 1 cross-attention) |
| Rethinking | recurrent block of 4 decoder layers × **6 passes**, re-attending to all backbone states each pass, depth embeddings, s0 re-anchoring |
| Scratchpad | 8 slots written after every pass (attention pooling), GRU-gated, readable by later passes |
| Head size | ~0.55B at d=2560 (exp7a: 45M) |
| Loss | same RLCD proper-score reward (log + 0.5·spherical − RPS), summed over passes |
| Attention | SDPA with 4D mask by default. FlexAttention (block-sparse, ~50% of blocks skipped) if the Colab preflight shows it is exact and faster. |

## Data

The exp8 corpora, rebuilt on Colab from public sources by `prepare_data.py`
(same scripts, seeds and order as the local build, including AFLite and bias
weights). The mix is exp8b's, with 50% of bool rendered as descriptive criteria.

## Colab plan (60 credits; ~5.3 units/h for A100 as shown in Colab's UI ≈ 11 A100-hours)

1. `scripts/colab_exp9_prep_cpu.sh`: CPU session rebuilds the data and downloads
   the weights to Drive, then releases itself. Almost free.
2. `scripts/colab_exp9_train_gpu.sh`: A100 session. Preflight on the real 4B model
   (branch-mask exactness, FlexAttention check, 30-step smoke train with eval +
   save + reload), ≈15–20 min. **A failed preflight stops the VM.** Then training
   with `--max_hours` (default 8.5): it stops itself, runs the final eval, saves
   to Drive and releases the VM.
3. `scripts/colab_exp9_watch.sh` to watch; W&B project `open-system-one`.

## Results

### exp9a (2026-09-27/28): Qwen3-Embedding-4B + LoRA + rethink head, 4.92B params, A100 40GB

Run history: preflight caught a torchao/peft conflict, then a Drive FUSE drop during a
7.5 GB copy, so data and weights moved to local disk with a retried Drive mirror. The
first launch crashed at step 424: after an OOM, locals in main() still pinned the failed
graph, so every later batch OOMed too. Fixed; resumed from step 250 with 12k-token
micro-batches. The credit budget cut the run to 3 h: 1450 steps (~46k examples), 2.4 h,
0 OOM skips, one final eval on a reduced suite (100/source, 300 intent, 200 typed).

| | exp7a | exp8b (0.4B) | **exp9a (4.9B)** |
|---|---|---|---|
| select / blind-hard | 54.9 / 53.6 | 73.2 / 71.6 | **81.4 / 80.9** |
| intent / mcq / bool / score / diversity | 78.8/57.6/70.3/36.2/54.3 | 83.7/67.0/84.3/59.2/72.8 | **90.5/89.9/88.1/62.3/77.4** |
| zero-shot intents (held out) | 70.6 | 75.7 | **90.0** |
| Banking77 71-way, routing (held out; Laya 42.5, Jev 87.0) | 55.3 | 55.0 | **73.3** |
| AG News (Laya 95.0, Jev 91.0) | 74.5 | 73.2 | **84.2** |
| DAIR Emotion (Laya 59.5, Jev 48.0) | 58.9 | 57.7 | **63.0** |
| typed choice / bool / bool@swapped / score | 23.2/59.6/–/33.2 | 27.8/57.2/57.2/31.8 | **61.0/71.0/71.0/47.0** |
| JevBench easy / original / hard (Laya 94.4/72.9/34.1; Jev 100/99/74.1) | 95.8/72.2/30.6 | 97.9/68.1/35.1 | **97.9/91.7/48.6** |
| order flip | 21.0 | 11.7 | **0.0** |
| utility at 3× / 10× error cost | −0.01 / −0.12 | 0.27 / 0.07 | **0.45 / 0.22** |
| ECE choice / bool / score | 2.4/7.6/10.2 | 3.0/2.5/5.4 | 3.7/4.0/8.4 |

**Vector injection (modality sweep, held out)**: vector-only is at chance at every N
(N=4: 28.0 vs 25.0; N=16: 7.0 vs 6.2; N=71: 1.7 vs 1.4; zero-shot N=50: 3.0 vs 2.0),
and `both` equals `text` everywhere. The gate ended at 0.028. **The injected option vector
carries no usable information into the decision, at 0.4B or at 4.9B. The hypothesis is
rejected.**

**Rethinking depth**: k=1..6 are all 82.3–82.4%. The recurrent passes, scratchpad and
depth embeddings add nothing, as at 0.4B.

**What drove the gains**: the backbone (a strong 4B embedding LLM) plus branch packing
(order invariance by construction, 0.0% flip). Held-out transfer, flat through exp8,
finally moved: +14 zero-shot intents, +18 Banking77, +11 AG News, +33 typed choice,
+23.6 JevBench original, +13.5 JevBench hard (vs the 31–40% ceiling of ≤0.6B encoders).
typed/bool = typed/bool@swapped = 71.0: the model now reads the content, not the
claim-echo.

Caveats: short training (46k examples), reduced val groups (±4–5 pts per group),
JevBench hard n=111. Weights: MyDrive/openjev/runs/exp9a/best_bf16.pt (trainable 0.9B
params; the frozen Qwen3-Embedding-4B base comes from Hugging Face).
