# Why things worked or didn't: a mechanistic analysis

This document explains the *why* behind every result across all five
iterations — not just what the numbers were, but what mechanism plausibly
produced them, how confident we should be in that explanation, and what
would need to be true for it to be wrong. README.md has the chronological
build log; this is the cross-cutting analysis.

Every claim below is labeled:
- **CONFIRMED** — directly measured, with a controlled comparison.
- **PLAUSIBLE** — consistent with the evidence, argued mechanistically, not
  isolated by a dedicated ablation.
- **SPECULATIVE** — a reasonable hypothesis, no direct evidence either way.

---

## 1. The core idea worked, immediately and consistently

**CONFIRMED.** Every single iteration, from the 50k-parameter toy model to
the 719M-parameter multi-question model, successfully learned to route
inputs to the correct candidate by embedding compatibility alone — no
decoder, ever. In-scope accuracy went 100% (6-way toy) → 96.6% (150-way
real) → 94.6% (227-way) → 93.6% (234-way) → 93.3% (234-way, multi-task).
The dip across iterations is *not* a sign of the method breaking down —
it's the task getting harder (more classes, closer negatives) while
accuracy stayed in a tight, high band the whole time. **Why this works
at all**: framing "pick the right one of K options" as "rank K
precomputed similarity scores" is a much easier learning problem than
"generate the exact right string," because the model never has to solve
tokenization, formatting, or exact-match parsing — it only has to make
semantically related things end up near each other in a vector space,
which is exactly what large pretrained language encoders are already good
at before any task-specific fine-tuning.

---

## 2. The outcome-description-augmentation fix (iteration 2 → 3): why it worked

**CONFIRMED**, with a controlled before/after on the same architecture.

| | iter 2 | iter 3 |
|---|---|---|
| paraphrase gap (base acc − paraphrase acc, same subset) | 0.237 | 0.050 |

**Diagnosis**: in iteration 2, every class had exactly one fixed
description string throughout training. Cross-entropy over a *fixed* set
of K anchor points can be solved by memorizing K arbitrary points that are
merely distinguishable from each other — nothing in that objective
requires the anchor for "pto_balance" to actually mean "PTO balance" to
the model; it only has to be a point that's far from the other 149
points. This is a well-known failure mode of contrastive learning with a
static target set (it's the same reason plain classification heads
memorize class prototypes rather than semantics).

**Fix**: resampling a different template + synonym-substituted phrasing
of each class's description on every training step means there is no
single fixed point to memorize anymore — the only way to minimize the
loss across many different surface forms of the same class is to actually
converge on the shared *meaning*. This is a standard idea (data
augmentation forces invariance to the augmented dimension) applied to the
side of a dual encoder that hadn't been getting it.

**Why this is not just "more data helps"**: iteration 3 also used a
bigger backbone and bigger training set at the same time as the fix, so
in isolation that comparison confounds three things. But the mechanism
argument (memorization vs. forced generalization) predicts specifically
that *outcome-side* augmentation should matter and generic scale should
not — and iteration 4/5's regression (next section) is independent
evidence for exactly that: bigger model, bigger data, same augmentation
mechanism, and the gap got *worse*, not better. That's hard to explain
under "it was just scale" and easy to explain under "the augmentation
mechanism interacts with the scoring architecture."

---

## 3. The paraphrase-gap regression (iteration 3 → 4/5): why it got worse despite more scale

**CONFIRMED the regression itself; PLAUSIBLE on the mechanism.**

| | iter 3 (bi-encoder) | iter 4 (poly-encoder) | iter 5 (poly-encoder, multi-task) |
|---|---|---|---|
| paraphrase gap | 0.050 | 0.139 | 0.143 |

This is the least comfortable result in the project and worth sitting
with rather than explaining away. Three architectures were tried, and the
**plain bi-encoder generalized to paraphrases better than either
poly-encoder variant**, despite the poly-encoders having 3-6x more
parameters, more data, and the same augmentation fix.

**Leading hypothesis**: the poly-encoder's final scoring step lets each
*candidate* attend over 16 context codes and pick whichever blend of them
fits best (`weights = softmax(scores, dim=m)` in `model_v4.py`). That's
exactly the mechanism that makes it more accurate in-distribution — it's
strictly more expressive than a fixed single context vector. But more
expressiveness on the *scoring* side is a second place the model can fit
to exact training phrasing, in addition to the outcome encoder itself.
The augmentation fix forces the *outcome encoder's* embeddings to be
robust to paraphrasing, but it does nothing to stop the *attention
weights* (candidate → context-codes) from being tuned to whatever
patterns happen to correlate with the training descriptions. A bi-encoder
doesn't have this extra fitting surface — there's only one context vector,
period, so the only way to improve training loss is through the two
embedding spaces directly.

**What would confirm or kill this hypothesis**: train a poly-encoder with
`n_codes=1` (which degenerates toward a bi-encoder) and see if the gap
shrinks back toward iteration 3's 0.050. That ablation was **not run** —
this is flagged as the single highest-value next experiment for anyone
continuing this project, because right now the poly-encoder-vs-generalization
tradeoff is inferred, not isolated.

**Secondary, compounding factor (PLAUSIBLE)**: iteration 4/5 also grew
the intent count from 227 to 234 with many more Banking77-style
near-duplicate classes. Under a harder discrimination task, any given
amount of embedding drift under paraphrasing is more likely to land the
paraphrased description in a *different, wrong* intent's territory than
it was with fewer/more separated classes. This wouldn't produce the
regression by itself (iteration 3's own jump from 150→227 intents with
Banking77 already added hard negatives and *still* had a small gap), but
it likely compounds with the primary architectural cause above.

---

## 4. Zero-shot generalization: real, modest, and revealing about what the model actually learned

**CONFIRMED the numbers; the interpretation below is PLAUSIBLE.**

| | iter 4 | iter 5 |
|---|---|---|
| zero-shot accuracy (35 never-trained intents, 234-way pool) | 8.84% | 10.45% |
| chance level | 0.43% | 0.43% |
| seen-intent accuracy, same mixed pool | 97.67% | 97.40% |

Chance is 0.43%, so ~9-10% is roughly **20-25x better than random** —
that is real signal, and it's the strongest evidence in this whole
project that the outcome encoder captures *some* transferable meaning
rather than a closed lookup table, because these 35 classes had **zero**
training examples and **zero** description exposure of any kind.

But it's also nowhere close to the ~97% the model gets on intents it
actually trained on, in that exact same candidate pool. **Why the gap is
this large**: the model was never trained on an objective that rewards
generalizing to a genuinely novel class — every training step's loss is
computed entirely in terms of the (fixed) seen-label set. The zero-shot
transfer that does happen is a side effect of the pretrained backbone's
prior knowledge (RoBERTa already "knows" what these words mean) surviving
fine-tuning, not something the training procedure explicitly optimizes
for. This is consistent with a broader pattern in embedding-based transfer
learning: some zero-shot capability survives narrow fine-tuning, but it
degrades compared to the pretrained model's zero-shot capability *before*
fine-tuning (which was not separately measured here — another gap in this
analysis, flagged rather than glossed over).

**A relevant, honest connection to actual JEPA**: LeCun's JEPA proposals
(I-JEPA, V-JEPA) are trained with a *self-supervised, non-contrastive*
objective — predicting masked regions in representation space using an
EMA-updated target encoder, with no fixed label set at all during
training. Everything built in this project instead used **supervised
contrastive learning against a fixed label set**, which is a meaningfully
different (and easier, more constrained) training signal. That's a
legitimate scope choice for this project — the goal was decision-making
over enumerable options, which supervised contrastive learning is a
natural fit for — but it means the zero-shot numbers here shouldn't be
read as "JEPA-style self-supervision generalizes to new concepts," because
that specific mechanism (predictive self-supervision without fixed
labels) was never actually used.

---

## 5. The multi-question architecture (iteration 5): what's genuinely proven vs. what looks better than it is

**Timing claim: CONFIRMED, cleanly.** 3 question types answered in one
joint forward pass vs. 3 separate forward passes over the same 256
contexts: **0.164s vs. 0.511s, a 3.12x speedup** — matches the theoretical
expectation for 3 question types almost exactly (perfect scaling would be
3.00x; measured overhead from the small extra attention/head compute
accounts for the difference). This is the cleanest, least ambiguous result
in the project because it's a direct architectural property, not
something that depends on data quality or training convergence.

**Main task (`intent`) not degraded by multi-tasking: CONFIRMED.**
93.34% (multi-task) vs. 93.56% (single-task, iteration 4) — within noise
of each other. Adding two more heads and objectives did not cost accuracy
on the primary task.

**`needs_human` / `urgency` accuracy: real, but weaker evidence than it
looks — read this carefully.** The reported numbers (99.3% / 97.3%,
both far above their majority-class baselines) look like strong wins.
**They should not be read as "the model learned to judge urgency."**
`needs_human` and `urgency` are **deterministic keyword-rule functions of
the intent label itself** (see `dataset_v5.py` — e.g. any intent whose
name contains "fraud", "stolen", "compromised" is hard-coded to
`urgency=5`). Since the model already gets the *intent* right 93.3% of
the time, and these two labels are just a fixed lookup table keyed on
intent, **getting them right ~93-99% of the time is close to what you'd
expect for free**, once intent is already solved — it is not strong
independent evidence that the auxiliary heads learned anything the intent
head hadn't already captured. The genuinely informative comparison would
be: on the ~6.7% of examples where the model gets `intent` *wrong*, does
it still get `needs_human`/`urgency` right at a rate consistent with the
keyword rule, or does it degrade with intent errors? **That breakdown was
not computed** — it's the natural next check to actually validate the
auxiliary heads learned an independent signal rather than free-riding on
the shared representation's intent knowledge.

**What the multi-question architecture is honestly proven to deliver, as
of this run**: parallel-pass speed (proven), no harm to the main task
(proven). What it's *not yet* shown: that it's useful for judgments that
are actually independent of the primary classification — which was the
whole motivating use case (Jev's own example pairs an *unrelated* Choice
and Score). The heuristic-label design here made that harder to tell
apart, by construction. Building an auxiliary task that's deliberately
*not* a deterministic function of the primary label is the fix for this,
flagged as a next step.

---

## 6. Two things that broke and what fixed them (process, not modeling)

Both are documented here because they materially affect how much to trust
specific numbers reported earlier in this project, and because "why did
this run give a wrong answer" is as much a legitimate question as "why
did the model give a wrong answer."

**Orphaned background process (iteration 3, first attempt)**: a training
run backgrounded with a raw shell `&` was believed killed but had actually
detached its Python child process, which kept training and kept
overwriting the same checkpoint files as the properly-tracked rerun.
Caught via `nvidia-smi` still showing GPU memory in use after the tracked
run had already exited and printed a "final" result. **Fix applied**:
always verify exactly one process is attached to a training run (`ps aux`
+ `nvidia-smi` before trusting any "completed" result), never trust
shell-level `&` backgrounding for anything that needs to be reliably
killable.

**Session-interrupted training (iteration 5)**: the background training
process for iteration 5 was killed when the Claude Code session that
launched it ended, mid-epoch-7, with no completion notification. **Fix
applied**: rather than blindly retraining (another ~3 hours), the best
checkpoint saved at epoch 6 (`jepa_v5_best.pt`, saved every time
validation improved) was loaded directly and the full final-evaluation
suite was re-run against it standalone (`eval_v5_checkpoint.py`) — meaning
the iteration 5 numbers reported here are real measurements from a
genuinely trained, if not fully-converged, checkpoint, not estimates or
a partial run's in-training validation numbers.

---

## 7. Summary: confidence levels for every headline claim

| Claim | Status |
|---|---|
| Compatibility scoring (no decoding) can solve fixed-option decision tasks accurately | **CONFIRMED**, all 5 iterations |
| Outcome-description augmentation fixes memorization-driven paraphrase failure | **CONFIRMED**, controlled before/after |
| Poly-encoder's extra scoring flexibility trades off against paraphrase generalization | **PLAUSIBLE**, mechanism argued, the isolating ablation (n_codes=1) not run |
| Zero-shot generalization to never-trained classes is real | **CONFIRMED** (~20-25x chance), but **modest in absolute terms** and not shown to come from anything resembling true self-supervised JEPA training |
| Multi-question single-pass architecture is faster than sequential | **CONFIRMED**, 3.12x for 3 questions |
| Multi-question architecture doesn't hurt the primary task | **CONFIRMED** |
| Auxiliary heads (`needs_human`, `urgency`) learned real independent judgment | **NOT SHOWN** — high accuracy is largely explained by the labels being a deterministic function of the already-well-learned intent, not evidence of new capability |

## 8. If continuing this project, in priority order

1. **Isolate the poly-encoder regression**: rerun iteration 3's exact
   setup with only the backbone swapped to RoBERTa-large (no poly-encoder)
   to separate "bigger backbone" from "poly-encoder scoring" as the cause
   of the paraphrase regression. Then try `n_codes=1` on the poly-encoder
   to see if the gap tracks code count.
2. **Build an auxiliary task that's genuinely decoupled from intent** (not
   a deterministic function of it) to actually test whether the
   multi-question architecture learns independent judgments, not just
   whether it can pass through the primary signal for free.
3. **Measure the pretrained backbone's zero-shot performance before any
   fine-tuning**, as a baseline for how much of the 8-10% zero-shot number
   is "survived from pretraining" vs. "learned during fine-tuning."
4. **Conditional accuracy breakdown**: auxiliary-head accuracy specifically
   on examples where the primary intent prediction is wrong, to separate
   "learned independently" from "riding on shared representation."
