# Project history — all experiments, what was done, what results exist

Covers experiments 1 through 6. For each: what question it was asking, what
changed, and what results actually exist (checkpoint metadata, docstring
references, or nothing recorded — stated plainly rather than guessed at).
Experiment 6 gets a short summary here since it has its own much more
detailed [`exp6_diverse_data_qqp_aux/SESSION_LOG.md`](exp6_diverse_data_qqp_aux/SESSION_LOG.md).

All experiments before exp6 share the same underlying task: score a context
(a user utterance) against candidate outcome descriptions via embedding
compatibility (JEPA-style, no text generation), starting from CLINC150 +
Banking77 + SNIPS (234 intents, ~38k examples) with a genuine zero-shot
holdout. Iteration 3 (before this experiments/ folder existed) had already
fixed a paraphrase-generalization problem via outcome-description
augmentation; iteration 4 introduced the poly-encoder architecture and a
bigger/harder dataset, and saw a paraphrase-generalization *regression*
(~0.14 gap vs iteration 3's ~0.05) — experiments 1-3 exist to isolate which
of iteration 4's three simultaneous changes (backbone, architecture, data
shape) caused that regression.

---

## Experiment 1 — Backbone ablation

**Question**: was iteration 4's paraphrase-generalization regression caused
by the poly-encoder *architecture*, or by a weaker *pretrained backbone*, or
both? Isolates the backbone variable by changing nothing else.

**Change**: swapped `all-roberta-large-v1` (355M, RoBERTa 2019 + later
embedding-tuned) → `Alibaba-NLP/gte-large-en-v1.5` (434M, 2024, purpose-built
embedding model, higher-ranked on MTEB). Architecture, data, and the rest of
the fine-tuning stack held identical to iteration 4.

**Result**: `exp1_best.pt` / `exp1_latest.pt` both show **epoch 7, val_acc
94.12%**. That's in the same range as iteration 4's own trajectory — no
recorded paraphrase-gap number survives (never appended to `NOTES.md`, no
log file kept). **Cannot honestly report whether the backbone swap closed
the paraphrase gap** — only that in-scope accuracy landed in a comparable
range to iteration 4 at a similar epoch. This experiment was effectively
superseded before that harder number was collected: `exp4_llm_encoder/
NOTES.md`'s status note explains experiment 1 (there run with `e5-large-v2`)
"was stopped early at epoch 6/30 — its val_acc had plateaued flat with
iteration 4's own trajectory (93.7% vs 94.0% at the same epoch), showing no
sign of a clear win on the easier metric, so continuing it to get the harder
paraphrase-gap number stopped seeming worth the wait" — i.e. abandoned in
favor of experiment 4's more ambitious backbone test.

## Experiment 2 — Small option-set reframing

**Question**: does training against realistic small candidate sets (3-8
options, matching Jev's actual per-request API shape) instead of one giant
fixed 199-way classification bank change discrimination/generalization —
independent of any architecture change?

**Change**: same poly-encoder architecture and same backbone
(`all-roberta-large-v1`, held constant deliberately, *not* experiment 1's
backbone) as iteration 4, but each example gets its own sampled 6-candidate
set (1 correct + 5 distractors, mixing hard negatives from the same source
dataset with random ones) instead of scoring against a shared 199-item bank.

**Result**: **no checkpoint, no recorded results exist.** `__pycache__`
shows the code was at least imported/smoke-tested, but no training run was
completed and saved (or if one was, nothing survived). Notably, this is the
exact same core idea later revisited and actually completed in exp6 — dynamic
per-example candidate sampling instead of a fixed global bank — which did
get a full run and real numbers (see exp6's log). Experiment 2's original
`NOTES.md` framing (compare directly against experiment 3, which used the
same data shape with a different architecture) never got its comparison run.

## Experiment 3 — Joint-sequence, marker-token readout

**Question**: does the architecture we deduced Jev (the external product this
project reverse-engineers) most likely uses — one shared transformer pass
over context and options together with full token-level self-attention,
instead of two separate encoders that only meet after pooling — actually
work, and how does it compare to the poly-encoder family?

**Change**: reuses experiment 2's small-option-set data framing exactly (to
isolate architecture as the one variable vs. experiment 2), but replaces the
poly-encoder with a genuinely different architecture: context text and all
candidate texts concatenated into **one sequence**, processed in **one
transformer forward pass** with full self-attention between every token
(context and candidates can directly attend to each other, unlike a
dual-encoder). Backbone also had to change (`answerdotai/ModernBERT-large`,
8192-token context) since RoBERTa's 512-token limit can't fit context +
options packed together — this means experiment 3 isn't a pure single-variable
ablation against experiment 2, stated explicitly in its own `NOTES.md`.
Uses marker tokens (`[OPT]`, `[NOUL]`, `[LVL]`) with cold-start (randomly
initialized) embeddings to read out per-option scores from the shared
sequence.

**Result**: **no checkpoint, no recorded results exist**, same as experiment
2 — code was set up (`__pycache__` present) but no completed, saved run.

## Experiment 4 — LLM-derived encoder backbone

**Question**: does a backbone with dramatically more pretraining scale and
world knowledge — a decoder LLM converted into an embedding encoder — do
meaningfully better than the encoder-family backbones (RoBERTa,
`e5-large-v2`) tried so far, on both in-scope accuracy and paraphrase
generalization?

**Change**: `gte-Qwen2-1.5B-instruct` (1.5B params, a Qwen2 decoder LLM with
bidirectional attention turned on and contrastively tuned for embeddings by
Alibaba) as the backbone, poly-encoder architecture and everything else held
identical to iteration 4/experiment 1.

**Pivot mid-experiment**: the first full-scale attempt (partial freeze, top
~1 layer trainable) technically ran without crashing but at ~5.4
hours/epoch — impractical. Rather than shrink the model, the backbone was
**frozen entirely** (all 28 layers + token embeddings) — linear-probing only
the poly-encoder's lightweight components (16 learned code vectors, their
attention layer, projection heads, temperature).

**Result**: `exp4_best.pt` shows **epoch 5, val_acc 92.61%**;
`exp4_latest.pt` shows **epoch 9, val_acc 92.46%** — both notably *lower*
than experiment 1's 94.12% and iteration 4's own ~94% trajectory, despite
the much larger backbone. The harder number, quoted directly from
`exp5_small_llm_unfrozen/model.py`'s docstring (written once this result was
known): **"experiment 4 found a 1.5B frozen decoder had the worst paraphrase
gap in the whole project — 22-32 points, vs 14 for full-fine-tuned encoder
backbones."** This is a real, informative negative result: more pretraining
scale did *not* help once the backbone was frozen — worse than every prior
full-fine-tuned smaller backbone on the metric that actually mattered.
Leading hypothesis carried into experiment 5: freezing removed the
adaptation capacity that the paraphrase fix (iteration 3's outcome-
description augmentation) actually depends on — the outcome encoder has to
be *able to move* for the augmentation trick to teach genuine invariance,
not just memorize a frozen embedding space's existing structure.

## Experiment 5 — Small LLM, unfrozen

**Question** (direct follow-on from experiment 4's finding): is "frozen vs.
adaptable" the real variable, independent of decoder-LLM-ness itself? Tests
by going 3x smaller (`Qwen2.5-0.5B` instead of 1.5B) specifically to make
*partial* fine-tuning affordable again on this hardware, rather than
concluding "decoder LLMs don't work" from a confounded frozen-only test.

**Change**: `Qwen2.5-0.5B`, the plain natively-supported decoder LLM (no
`trust_remote_code`, no pre-existing embedding conversion — this project's
own fine-tuning is what turns it into an embedder). Required a real
architectural adaptation: Qwen2.5 is causal (each token only sees itself and
earlier tokens), so the outcome side switched to **last-token pooling** with
left-padding (the last token is the only position that has attended to the
whole sequence) instead of the mean-pooling used by every prior
bidirectional-encoder backbone.

**Result**: **no checkpoint, no `NOTES.md`, no recorded accuracy numbers
exist for this experiment as its own thing.** What's known comes only from
exp6's own docstring, which references it purely as a *feasibility/memory
validation*, not a completed accuracy experiment: "same backbone as
experiment 5 (validated: fits comfortably fully unfrozen, ~11.6GB,
reasonable speed)." I.e. experiment 5 established that `Qwen2.5-0.5B` could
be fine-tuned *fully unfrozen* (not just partially, as its own `NOTES.md`-
absent design originally planned) within memory/speed budget on this
hardware — that finding is what let exp6 start from "fully unfrozen" as an
already-validated default rather than re-discovering it. No paraphrase-gap
or zero-shot number from experiment 5 itself survives anywhere.

## Experiment 6 — Diverse data + QQP auxiliary (full detail in its own log)

The current, actively-developed experiment — same `Qwen2.5-0.5B` backbone as
experiment 5, fully unfrozen (validated there), on a much larger 5-source
intent corpus (44.4k examples, 300 intents) plus a QQP paraphrase-pair
auxiliary objective, later extended within one long session to also include
context-grounded MCQ answer-selection training (RACE/SciQ passages + 5 more
sources) and a fundamental reframing of the intent task itself from
closed-set classification to dynamic per-example candidate scoring — directly
revisiting experiment 2's original, never-completed idea, this time actually
finished with real results.

**Best result**: `exp6_best_zeroshot.pt`, epoch 13 — val_acc 94.66%,
zero_shot_acc 47.5% (vs. this session's own pre-fix baseline of 21.5%, and
far above anything experiments 1-5 recorded on comparable metrics), qqp_val_acc
77.0%, mcq_val_acc 37.4% (blended; per-source breakdown in the log shows a
2.6x-chance result on SciQ down to 1.4x-chance on CommonsenseQA/ARC-Challenge
— consistent with known small-model scale limits on reasoning-heavy MCQ
sources, not a flaw in the approach).

Full detail — data-quality fixes, the exact training-mechanism changes, two
real OOM crashes and their fixes, an entire day's Colab-CLI reliability
investigation (with an actual root cause found and a working fix), the W&B
integration, the WiSE-FT interpolation experiment (clean negative result,
with an explanation of why it doesn't apply to this architecture), and the
per-source MCQ/SOTA-context breakdown — is all in
[`exp6_diverse_data_qqp_aux/SESSION_LOG.md`](exp6_diverse_data_qqp_aux/SESSION_LOG.md).

---

## Cross-experiment pattern, stated plainly

Every backbone/architecture change from experiments 1-5 either regressed
(experiment 4's frozen 1.5B LLM) or was never actually measured to completion
(experiments 1, 2, 3, 5 — either abandoned early once a competing idea looked
more promising, or set up but never run to a saved, recorded result).
**The single biggest, best-documented, actually-measured win in the entire
project's history is exp6's this-session combination**: full fine-tuning
(not freezing) of a small decoder LLM, on genuinely diverse multi-source
data, with the *training task shape itself* reframed to match the real
target capability (dynamic per-example option sets, context-grounded
answer selection) rather than either a fixed closed-set benchmark or an
architecture swap alone. That reframing — not a bigger/different backbone —
is what experiments 2 and 3 originally set out to test and never finished;
exp6 is the first time in this project's history that idea actually got
carried through to a measured, real result.

---

## Experiment 7 — Hybrid dual-path decision, prompted by finding Laya

Full rationale, architecture, training plan, and staging in
[`exp7_hybrid_decision/NOTES.md`](exp7_hybrid_decision/NOTES.md). Short
version: `LAYA_COMPARISON_REPORT.md` found that Convai's open-source Laya
had already shipped essentially experiment 3 (ModernBERT-large,
joint-sequence packing, `[MASK]`-marker readout) — it beats Jev at low
option counts and collapses at high ones (Banking77: 42.5% vs Jev's 87.0%)
because of a shared, fixed token budget across all options. Exp7 combines
Laya's joint-sequence mechanism with a second, independently-encoded
option-vector path injected at each `[MASK]` position (constant one-token
cost regardless of option-text length, so it doesn't hit that same wall),
plus a third gated MaxSim/late-interaction path, adaptive latent depth in
the decision head (not the backbone — recurrent-depth pretraining doesn't
transfer to a frozen pretrained stack), and Laya's own proper-scoring-rule
losses minimized directly instead of via RLCD/GRPO.

**Status: implemented, not yet trained.** `model.py`/`data.py`/`train.py`/
`eval.py`/`calibrate.py` all exist and were verified locally end-to-end
(real tokenizer + a from-scratch tiny backbone, real local intent/mcq/qqp
data through the training pipeline) — see the "Implementation status"
section at the bottom of `exp7_hybrid_decision/NOTES.md` for exactly what
was and wasn't checked. `scripts/build_exp7_data.py` (new bool/score/
label-diversity/typed-decisions corpora) and six `colab_exp7_*.sh` launch
scripts exist but have not been run against the real Colab/A100
environment. No accuracy numbers of any kind exist yet for this experiment.
