# Experiment 7 — Hybrid dual-path decision model with adaptive latent depth

Written before the run, in this project's usual convention. Nothing here is a
result yet. Results get appended at the bottom.

## The one-sentence question

Can a ~400M encoder beat Laya (the shipped open-source Jev-compatible System-1
model) by removing the one mechanical weakness its architecture provably has —
the shared token budget that collapses accuracy at high option counts — without
giving up the low-cardinality strength that joint-sequence packing buys?

## Background: why this experiment exists

[`LAYA_COMPARISON_REPORT.md`](../LAYA_COMPARISON_REPORT.md) established that
someone already shipped what this project set out to build. Laya is
ModernBERT-large + joint-sequence packing + `[MASK]` marker readout — i.e.
essentially [experiment 3](../exp3_joint_sequence/NOTES.md), finished. It
beats Jev at low cardinality (AG News 95.0 vs 91.0, DAIR Emotion 59.5 vs 48.0)
and loses catastrophically at high cardinality (Banking77 42.5 vs 87.0).

That split is not a training deficiency. Section 4 of the comparison report
traced it to code: `head_max_len` (192-256 tokens) is a **shared budget across
instructions AND all options combined**, so 77 options get ~3-4 tokens each.
The wall is mechanical.

Two conclusions follow, and they define this experiment:

1. **Jev is probably not doing pure joint-sequence packing.** If it were it
   would hit the same wall. Experiment 3's deduction about Jev's architecture
   is at minimum incomplete. (Inference from the 42.5-vs-87.0 split, not proof.)
2. **There is an architecture with neither wall**, and no one has published it
   for this task. That is what exp7 builds.

## Core idea: two paths that degrade in opposite directions

Option information reaches the decision through **two independent channels**:

- **Text path** — the option's text sits in the packed sequence next to its
  `[MASK]`, giving real token-level self-attention between context and options.
  This is Laya's mechanism and Laya's strength. It degrades as N grows because
  the shared budget squeezes each option's text toward nothing.
- **Vector path** — the option's *full, untruncated* text is encoded separately
  (weight-tied backbone, independent forward pass, mean-pooled) and that vector
  is injected into its `[MASK]` position's input embedding. Costs **one token
  per option regardless of option text length**. Does not degrade with N.

So the model slides continuously from cross-encoder-like (low N) to
bi-encoder-like (high N) with no cliff, through a single code path. Laya has a
cliff. exp6's dual encoder has no peak. This is meant to have both ends.

A third, initially-disabled channel (**MaxSim**, §2.6) fills the remaining gap
at very high N, where the text path is dead and a single pooled vector is the
only option-side signal.

---

## 1. Request shape

Every request is one typed question against a context:

```
context/state:  "my card was swallowed by the ATM yesterday and never came back"
question type:  choice | bool | score
instructions:   "Which category should this be routed to?"
options:        ["card swallowed", "card arrival", "lost or stolen card", ...]
-> answer:      a probability distribution over the options
```

- `choice` — N options, answer is argmax.
- `bool` — 2 options (yes/no), same mechanism.
- `score` — N **ordinal** options (levels 1..N), answer is the
  probability-weighted expected value `sum_k k * p_k`, not argmax.

The question type changes the loss and the readout interpretation. It does not
change the architecture.

---

## 2. Architecture

### 2.1 Backbone

`answerdotai/ModernBERT-large` — 24 layers, d=1024, ~395M params, RoPE,
unpadded attention, 8192-token context.

**Fully fine-tuned, not frozen.** Two independent reasons:

- [Experiment 4](../exp4_llm_encoder/NOTES.md) already ran the frozen version
  and lost badly: a frozen 1.5B decoder had the **worst paraphrase gap in the
  whole project — 22-32 points, vs 14 for full-fine-tuned encoder backbones**.
  The recorded diagnosis: freezing removes the adaptation capacity the
  paraphrase fix depends on. exp5 and exp6 exist because of that finding, and
  exp6 fully-unfrozen is the best result in the project's history.
- **This design structurally requires backbone adaptation.** The `[MASK]`
  injection (§2.3) only works if the backbone learns to read a foreign vector
  at the mask position. Frozen, the projector would have to find a pre-existing
  interpretive channel nobody trained to exist.

Laya trains its backbone too, so this also holds the variable constant for
comparison.

Forgetting control is via **layer-wise LR decay**, not freezing (§4.2). LoRA is
documented as a fallback lever in §8 if the zero-shot decline reappears.

ModernBERT specifically (rather than Ettin, which benchmarks higher) so the
Laya comparison means something. Ettin swap is a clean single-variable
follow-up.

### 2.2 The packed sequence

```
[CLS] <type_tok> <instructions> [SEP] [MASK] opt_0_text [MASK] opt_1_text ... [SEP] <context> [SEP]
```

Context goes **last**, after every option — same as Laya.

**Budget allocator** decides how much text each option gets:

```python
B_total     = 2048                      # not 8192: attention is quadratic
avail       = B_total - len(instr) - len(context) - N - overhead
per_option  = clamp(avail // N, 0, L_max)     # L_max = 64
```

- N=4   -> ~64 tokens/option -> effectively a cross-encoder
- N=20  -> ~40 tokens/option -> partial text
- N=255 -> 0 tokens/option   -> `[MASK]` slots only, vector path carries it

One code path. No branching. No special case.

### 2.3 Vector injection at the mask positions

Each option's full untruncated text goes through the **same backbone**
(weight-tied, independent forward pass, mean-pooled) -> `v_i` in R^1024.

```python
inj_i = LayerNorm(MLP_proj(v_i)) * emb_scale
input_emb[mask_pos_i] = mask_token_emb + g * inj_i + type_emb(qtype)
```

- `MLP_proj`: 2-layer, 1024 -> 1024 -> 1024, GELU. **Not** a bare linear —
  this is the LLaVA projector pattern (mapping one encoder's pooled output into
  another's token-embedding space) and a single linear underperforms there
  consistently.
- `emb_scale`: matches the projected vector's norm to the embedding table's
  typical token norm. **Not optional.** If `inj` is large it swamps the
  `[MASK]` embedding and destroys the MLM prior that is the entire reason for
  using `[MASK]` rather than a fresh marker token.
- `g`: a **learned scalar gate, initialized at 0.01**. At step 0 the model is
  effectively Laya; it learns how much vector to mix in. This makes exp7 a
  strict superset of the baseline at initialization — it cannot be worse for
  purely architectural reasons. `g` is also a live diagnostic: if it stays near
  zero, the vector path is not earning its cost.

**Why `[MASK]` rather than a new marker token**: exp3's `[OPT]`/`[NOUL]`/`[LVL]`
markers were randomly initialized and had to learn their role from scratch on a
small corpus — flagged in exp3's own notes as a real risk. `[MASK]` already
carries enormous MLM pretraining signal for exactly the role needed here
("something belongs at this position, infer it from surrounding context").
Laya's trick, taken directly.

### 2.4 Context compression

m = 16 learned query codes cross-attend over the backbone's context-token
hidden states -> `C` in R^{16x1024}.

This reuses the poly-encoder machinery from v4/v5, but the codes now have an
honest job (compress context for the deliberation loop) instead of being the
scoring mechanism itself — which is what [`ANALYSIS.md` §3](../../ANALYSIS.md)
blamed for the v4 paraphrase regression. Scoring is done by the `[MASK]`
readout now, not by the codes.

### 2.5 Decision head — 1 entry layer + 2-layer recurrent block

```
backbone output (full packed sequence, d=1024)
        |
        +-- m=16 context codes cross-attend over context tokens -> C  (16 x 1024)
        +-- gather N mask positions                             -> O  (N  x 1024)
        |
    s_0 = EntryLayer([CLS ; C ; O])          # 1 TransformerEncoderLayer, NOT looped
        |
    for k in 1..K:                           # SHARED weights every pass
        s_k     = RecurrentBlock(s_k-1 + s_0)  # 2 TransformerEncoderLayers
        score_k = Scorer(gather_options(s_k))
        p_k     = masked_softmax(score_k)
```

Design decisions and why:

- **The head operates on `16 + N + 1` vectors, not the 2048-token sequence.**
  At N=255 that is 272 positions; looping 6 times is nearly free. Looping the
  full sequence 6 times would cost more than the backbone.
- **1 entry layer, non-recurrent** — adapts backbone output into the head's
  working space once. The "prelude" in Geiping et al's recurrent-depth
  formulation.
- **2-layer recurrent block** — minimal block with two rounds of
  attention+FFN. One layer is too weak to do meaningful per-pass work; four
  makes each pass expensive and the depth curve coarse.
- **`s_k-1 + s_0` re-injection every pass** — without re-injecting the initial
  state, a looped block drifts away from its input and degenerates after a few
  passes. Standard fix in recurrent-depth models.
- **No step/depth embedding** — deliberately. K is therefore unbounded at
  inference: you can run more passes than you trained with and measure whether
  it still helps.
- **Loop the head, never the backbone.** Pretrained transformer layers are not
  depth-interchangeable — layer 14 of ModernBERT learned a specific role at a
  specific depth. Feeding it its own output produces garbage. Recurrent-depth
  models (Ouro, Universal Transformer) get their parameter efficiency because
  they were *pretrained* with weight sharing; that cannot be retrofitted. The
  head is trained from scratch, so it can be trained depth-invariant from
  step one.

Parameter cost: ~38M on top of 395M, roughly 10%.

`K_max = 6`.

### 2.6 MaxSim path (built, gated OFF for run 1)

```python
E_i = proj_late(H_opt_i)              # (L_opt, 1024) -> (L_opt, 128), L2-normed
C_t = proj_late(H_ctx)                # (L_ctx, 1024) -> (L_ctx, 128), L2-normed
maxsim_i = sum_t max_s (C_t[t] . E_i[s]) / L_ctx

score_i = Scorer(s_K[opt_i]) + w * maxsim_i      # w = learned gate, init 0.0
```

- **Projection to 128 dims is mandatory**, not an optimization. At d=1024,
  per-token option embeddings for N=255, L=64 cost ~33MB *per example* — over
  1GB for a batch of 32. At d=128 it is ~4MB/example.
- Added to the logit rather than injected at the mask, so it ablates cleanly by
  setting `w=0`.
- **Disabled in run 1** (`--use_maxsim false`). See §6 for why, and for the
  measurement that decides whether to enable it.

Where it earns its place: at N>=77 the text path is dead and an option's entire
meaning collapses to one pooled vector — the exact information bottleneck
`ANALYSIS.md` §1 identified back in v1. MaxSim supplies token-level matching at
zero sequence-budget cost, giving the degradation ladder a middle rung:

```
full cross-attention  ->  MaxSim token-level  ->  single pooled vector
     (low N)                  (high N)                (fallback)
```

Secondary benefit worth recording: MaxSim is **interpretable**. You can read
which option token matched which context token. For a model whose selling point
is escalating to a human when unsure, being able to show *why* it routed
somewhere is a real product feature. A scalar off a `[MASK]` position explains
nothing.

### 2.7 Scorer and calibration

```python
score_i = Linear(GELU(Linear(LayerNorm(s_K[opt_i]), 1024)), 1)
p       = softmax(scores.masked_fill(pad_mask, -1e4) / tau[qtype, card_bucket])
```

Per-(question type, cardinality bucket) temperature, **fit post-hoc on
validation**, not learned during training. Buckets: `2`, `3-5`, `6-10`,
`11-30`, `31+`.

Laya's published fitted values are the warning: `choice:2 -> 1.91` (soften),
`choice:11+ -> 0.10` (aggressively sharpen). Raw logits get badly
miscalibrated as option count grows. Expect the same and measure it.

---

## 3. Loss

### 3.1 Answer loss — proper scoring rules, minimized directly

```
L_answer(p, t) = CE(p, t)                      # log score
               + 0.5 * (1 - spherical(p, t))   # spherical = (t.p) / ||p||
               + 1.0 * RPS(p, t)               # ORDINAL question types only

RPS(p, t) = sum_j ( cumsum(p)_j - cumsum(t)_j )^2
```

**Deliberate departure from Laya**: they optimize these with a GRPO-style
policy gradient. All three terms are differentiable functions of `p` and can be
minimized directly by gradient descent, at far lower variance. RLCD's policy
gradient buys them multi-turn TD(lambda) bootstrapping across conversation
prefixes; this experiment has single-turn supervised data, so it would be paying
RL's variance cost for nothing. Revisit if multi-turn data is added.

**RPS is what makes `score` questions work.** Plain cross-entropy gives zero
credit for predicting urgency 4 when the truth is 5 — which is why v5's
`urgency` head was weak and why its headline number was, as
[`ANALYSIS.md` §5](../../ANALYSIS.md) recorded, largely free-riding on the
already-solved intent label.

### 3.2 Depth loss — supervise every pass

```python
L_depth = (1 / K_max) * sum_{k=1..K_max} L_answer(p_k, t)
```

Run all `K_max` passes, compute the loss at each. Every depth learns to emit a
valid answer, and the **depth-vs-accuracy curve comes out as a free training
artifact**.

Uniform weighting, deliberately. Weighting deep passes more would undertrain
shallow ones and make adaptive halting useless.

### 3.3 Total (run 1)

```
L = L_depth + 0.3 * L_qqp
```

The QQP paraphrase-pair auxiliary carries over from exp6. Cheap, and it targets
the option-side paraphrase invariance that has been the documented weak point
since v2.

Halting loss and escalation loss are **not** in run 1 — see §6.

---

## 4. Training

### 4.1 Input augmentation — the make-or-break section

Applied per example, every step:

| Augmentation | Setting | Why |
|---|---|---|
| **Option count N** | log-uniform 2-128; 5% of steps at 255 | Model must work across the whole cardinality range, not just where it trained |
| **Modality dropout** | 20% vector-only, 20% text-only, 60% both | **Critical — see below** |
| **Option order** | reshuffled every step | Kills positional selection bias (a documented MCQ failure mode) |
| **Option rendering** | random per example: bare label / 8-template / full description / label+description | Fixes the format-shift bug found in `eval_external_benchmarks.py:36` |
| **Instructions** | sampled from a paraphrase bank; sometimes empty | Invariance to how the question is asked |
| **Distractors** | hard (same source/domain) + random, positive-aware filtered | NV-Retriever-style filtering to drop distractors that are secretly correct |

**Modality dropout is the single detail least safe to skip.** At low N the two
paths are redundant, the text path is richer, gradients flow there, and the
vector path starves into decoration. Then at high N when text disappears the
model falls off Laya's cliff anyway — having paid for two encoders to get
there. Dropout forces each path to carry the task alone. It is also what makes
the §7 ablation modes *valid*, since both paths will have been trained to
stand alone.

**The option-rendering fix matters more than it looks.** Every option string
the model saw in exp6 was one of 8 templates (`dataset_v7.py:46`,
`"The user wants help with: {x}."`). The external benchmark eval fed bare
strings (`"World news"`, `"sadness"`). Part of exp6's 50.4% on AG News is
plausibly format shift, not capability. Randomizing rendering during training
removes the confound permanently.

### 4.2 Optimizer and schedule

```
backbone LR      1e-4 top layer, layer-wise decay 0.9 downward
head LR          3e-4  (entry layer, recurrent block, scorer, codes, projector, gates)
optimizer        8-bit AdamW (bitsandbytes)
precision        bf16 autocast
grad clip        1.0
schedule         cosine, 5% warmup
grad checkpoint  on
batch            dynamic token-budgeted (sequence length varies wildly with N)
K_max            6
epochs           up to 20, early stop on ZERO-SHOT accuracy (see below)
```

**Early stopping is keyed to zero-shot accuracy, not val_acc.** Flagged as an
open thread in exp6's session log and never acted on. `val_acc` has sat in a
tight 93-96% band across a 14,000x parameter range and across architectures
that differed enormously on what actually matters — it is an actively
misleading stopping signal. exp6's zero-shot peaked at epoch 13 while val_acc
kept climbing to epoch 16.

Memory: ModernBERT-large at 395M in bf16 with 8-bit Adam and gradient
checkpointing is roughly 2.5GB of weights/grads/optimizer state. Memory is not
the constraint this time; exp4's forced freeze does not apply.

---

## 5. Data

### 5.1 Existing, with changes

| Source | Use | Change vs exp6 |
|---|---|---|
| `intent_corpus` (44.4k, 300 intents) | choice | **Banking77 removed entirely** (see §5.3) |
| `mcq_corpus` (148k) | choice w/ context | **Reweighted**: HellaSwag 57% -> ~15% |
| `qqp_paraphrase_pairs` (100k) | auxiliary | unchanged |

**Why reweight HellaSwag**: it is 39,796 of 69,993 MCQ training examples and was
constructed adversarially against models below roughly GPT-3 scale. exp6's
per-source breakdown measured 38.9% on it — right on-trend for the size class
*before* accounting for any training benefit. Most of the MCQ gradient is
currently spent on something unlearnable at this scale, and it drags the
blended number so hard that the metric stops being informative.

### 5.2 New data that has to be built

- **Ordinal / `score` data** — none currently exists, and RPS needs it. Free
  sources: Amazon and Yelp star ratings (genuinely ordinal 1-5), SST-5, any
  Likert-scale corpus.
- **Bool data** — MNLI/SNLI entailment, BoolQ, FEVER.
- **Label-vocabulary diversity** — the biggest single lever, and the reason
  Laya gets 95% zero-shot on AG News while exp6 gets 50.4%. Reformat every
  public classification dataset reachable into option-set form: DBpedia-14,
  TREC, Yahoo Answers, GoEmotions, stance, toxicity, topic sets. **Each dataset
  is a new label vocabulary.** exp6's model has seen exactly one vocabulary in
  its life (255 intents). GLiClass — the closest published analogue — trained
  on 1.2M examples spanning many.
- **Synthetic option sets** — LLM-generated `(context, question, options,
  answer)` across domains, varying N (2-128), option style, question type, and
  including explicit none-of-the-above cases.

### 5.3 Held out, never trained on

- **AG News, DAIR Emotion** — the Laya/Jev comparison benchmarks.
- **Banking77** — **and this costs 12.4k training examples, deliberately.**
  The entire thesis of exp7 is "we beat Laya at high cardinality." Laya scores
  42.5% there zero-shot. exp6 scores 89.35% *having trained on it*, which
  proves nothing and is labeled as such in the comparison report. Without the
  holdout there is no headline result.
- **`LocalLLaMA/typed-decisions`** (Apache-2.0, ~1.6k rows) — used as a
  **target-domain eval set, not training data.** Too small to train on
  meaningfully, synthetic from a competitor's pipeline, and it is the only
  decision-routing-flavored benchmark in reach. A `laya-typed-decisions`
  checkpoint exists for direct comparison.
- **45 zero-shot intents** — as in exp6.

---

## 6. Staging — one variable at a time

exp7a carries four interdependent new ideas (injection, weight-tying,
variable-N, modality dropout). That is already more simultaneous change than
[v4's lesson](../../ANALYSIS.md) supports — v4 changed three things at once and
it took six experiments to partially untangle which one caused the paraphrase
regression. It is defensible here only because those four are mutually
dependent: none is individually testable without the others. The §7 ablation
modes are the insurance, and they are not optional.

Everything that *is* separable gets staged:

### exp7a — fixed depth, MaxSim off

Full architecture, `K = 1` (no looping), `w = 0`, `--use_maxsim false`.
Deliverable: the **cardinality sweep x ablation mode** table (§7). This is the
headline result and it stands alone.

### exp7b — depth-conditioned

Same model, `K_max = 6`, loss at every depth. Deliverable: the
**depth-vs-accuracy curve**, broken out by source.

> **If the curve is flat, stop here.** Looping is a dead end for this task and
> exp7c is cancelled. Expect flat on lexical-overlap sources (SciQ, ARC-Easy)
> and rising on ARC-Challenge / CommonsenseQA if it rises anywhere — those are
> the weak sources and exactly where deliberation should pay.

### exp7c — learned halting + escalation

Freeze the head. For every example the full trajectory `p_1..p_6` now exists.

**Halting head** — small MLP over features computed at each depth:

```
[ top_prob(p_k), margin(p_k), entropy(p_k), N, k, KL(p_k || p_k-1) ]  ->  halt?
```

`KL(p_k || p_k-1)` is the strongest feature: if the distribution has stopped
moving, another pass will not help.

Target is **self-supervised, no new labels**: `halt = 1` if
`argmax(p_k) == argmax(p_K_max)`, i.e. "more passes will not change my mind."
Threshold `theta` tuned on validation to trace a compute-vs-accuracy curve.

**Escalation head** — same features at the halted depth, separate MLP, target
= "is the final answer actually wrong." Conformal-calibrated on a held-out set
so the escalate decision carries a distribution-free error guarantee rather
than a hand-tuned threshold.

This gives one coherent mechanism for both questions:

- confident early -> answer in 1 pass (System 1, cheap)
- unresolved -> keep looping (graded deliberation, no tokens emitted)
- still unresolved at `K_max` -> **escalate** (System 2 handoff)

### exp7d — MaxSim on

Single variable: `--use_maxsim true`, `w` trainable. Only run if exp7a's sweep
justifies it (see §7).

---

## 7. Evaluation

### The headline figure

Cardinality sweep x ablation mode. One trained model, three eval modes.

| | N=2 | N=4 | N=8 | N=20 | N=77 | N=255 |
|---|---|---|---|---|---|---|
| text only (`g=0`) ~ Laya | | | | | | |
| vector only (no option text) ~ bi-encoder | | | | | | |
| **both (exp7)** | | | | | | |

**The prediction the whole design rests on**: text-only falls off past N~20,
vector-only is flat but mediocre, and *both* tracks the upper envelope of the
two. If "both" does not beat the max of the other two anywhere, the dual path
is not earning its cost and the design is wrong.

**This table also decides exp7d.** Read the vector-only column at N=77 and
N=255: if it holds up, MaxSim has little room to gain and should be skipped; if
it collapses, that collapse is both the justification and the size of the
available win.

Note what this table is worth beyond exp7: it answers cross-encoder vs
bi-encoder **inside a single training run, with no confounds**. Every
architecture comparison in this project since v4 has been across separately
trained runs that changed several things at once, which is why `ANALYSIS.md`
still lists the question unresolved. Modality dropout is what makes these modes
valid.

### Everything else

| Metric | Reference point |
|---|---|
| AG News (4-way, zero-shot) | Laya 95.0, Jev 91.0, exp6 50.4 |
| DAIR Emotion (6-way, zero-shot) | Laya 59.5, Jev 48.0, exp6 26.25 |
| **Banking77 (77-way, TRUE zero-shot)** | Laya 42.5, Jev 87.0 |
| `typed-decisions` (target domain) | `laya-typed-decisions` |
| Zero-shot intents (45 held out) | exp6 47.5 |
| Depth-accuracy curve, per source | new |
| Order-invariance (shuffle options, measure flip rate) | new |
| ECE per cardinality bucket | new |
| Latency, per cardinality | Laya 33-40ms, Jev 236-276ms |

---

## 8. Known risks, stated before the run

1. **The gate `g` may stay near zero** and the model ignores the vector path
   entirely. Modality dropout is the countermeasure. Watch `g` every epoch as a
   live diagnostic — if it is still <0.05 by epoch 3, the dropout rates are
   wrong or the norm matching is off.
2. **Depth may not help at all.** Genuinely possible. exp7b answers it cheaply
   and cancels exp7c if flat.
3. **Catastrophic forgetting may return.** This is exp6's documented failure
   pattern (zero-shot peaking then declining). Layer-wise LR decay is the first
   line of defense. **Fallback lever**: LoRA (r=32, alpha=64, on q/k/v/o +
   FFN, backbone otherwise frozen) — bounds drift structurally, and its
   inference-time `alpha` scaling gives a working version of what WiSE-FT tried
   and failed to deliver in exp6 (it failed there because the heads had no
   pretrained counterpart to interpolate toward; low-rank deltas perturb far
   less). Not guaranteed, but cheap to test.
4. **Holding out Banking77 costs 12.4k real training examples** to buy one
   honest number. Judged clearly worth it, but it is a real cost.
5. **exp7a is four changes at once.** Diagnosing an underperforming run will be
   work. The ablation modes are the mitigation.
6. **The Jev deduction in Background is inference, not proof.** The 42.5-vs-87
   split is the cleanest available reading, but Jev's actual architecture is
   undisclosed.

---

## 9. Glossary of terms introduced by this experiment

**Budget allocator** — the rule deciding how many tokens of each option's text
fit in the packed sequence, given a fixed total budget and N options. What
produces the continuous cross-encoder -> bi-encoder slide.

**Vector injection** — adding a separately-computed pooled option embedding
into the input embedding at that option's `[MASK]` position, so the slot carries
the option's full meaning at a cost of one token.

**Modality dropout** — randomly disabling one of the two (or three) option
information channels during training so each is forced to carry the task alone.
Borrowed from multimodal training.

**Proper scoring rule** — a reward for a predicted distribution whose expected
value is maximized only by reporting your honest belief. Log score, spherical
score and RPS are all proper; ordinary accuracy is not.

**RPS (ranked probability score)** — `sum_j (cumsum(p)_j - cumsum(t)_j)^2`.
Penalizes an ordinal prediction by *how far off* it is, which log score cannot.

**MaxSim / late interaction** — ColBERT's scoring operator: keep per-token
embeddings on both sides, score as the sum over context tokens of each one's
maximum similarity to any option token. Token-level matching without full
cross-attention.

**Recurrent depth** — applying the same transformer block repeatedly to scale
effective depth without adding parameters. Distinct from a deep stack because
the weights are shared, and distinct from chain-of-thought because the
iteration happens in representation space with no tokens emitted.

**`s_0` re-injection** — adding the initial head state back in at every
recurrent pass, to stop the looped block from drifting away from its input.

**Conformal calibration** — a distribution-free procedure that turns a
confidence score into a set-valued prediction with a finite-sample coverage
guarantee. Used here so "escalate" means something precise rather than
"below a threshold someone picked."

---

## 10. Files in this folder

To be written:

- `model.py` — backbone, packed-sequence builder, budget allocator, projector,
  injection, context codes, entry layer, recurrent block, scorer, MaxSim path.
- `data.py` — option-set request builder, augmentation pipeline (§4.1),
  distractor sampling with positive-aware filtering, source reweighting.
- `train.py` — training loop, depth loss, QQP auxiliary, layer-wise LR decay,
  zero-shot-keyed early stopping.
- `eval.py` — cardinality sweep, the three ablation modes, external benchmarks,
  depth curve, order-invariance, ECE, latency.
- `calibrate.py` — post-hoc per-(type, cardinality-bucket) temperature fitting.
- Results get appended to the bottom of this file once runs complete.

---

## Implementation status (code written, not yet trained)

All files above exist and are wired together (`smoke_test.py`, `data.py`,
`model.py`, `train.py`, `eval.py`, `calibrate.py`), plus
`scripts/build_exp7_data.py` (bool/score/diversity/typed-decisions corpora)
and six `scripts/colab_exp7_*.sh` launch scripts mirroring exp6's Colab
workflow (CPU probe to download+verify the real backbone before paying for
GPU time, full setup+launch, resume, relaunch-only, status check, watch).

**Verified locally** (real ModernBERT tokenizer + a from-scratch tiny
ModernBertConfig backbone, to avoid downloading the real ~395M-param
checkpoint just to check the mechanism):
- The `inputs_embeds` injection path produces bit-identical output to the
  ordinary `input_ids` path when no vector is injected — confirms the
  embeddings-level norm/dropout is applied correctly either way (Sec 2.3's
  design assumption, checked against the actual installed `transformers`
  source, not just the docs).
- Full forward + depth loss + backward at `k_max` in `{1, 4}`, `use_maxsim`
  in `{False, True}`, and a mixed batch with `N` in `{2, 5, 6, 18, 48, 77}`
  in the SAME batch (exercises the budget allocator and padding paths).
  All gradients finite, all logits finite at valid positions.
  `text_only` / `vector_only` / `both` ablation modes produce different
  logits (neither modality channel is silently dead).
- `train.py`'s `TaskMixer` end-to-end against the REAL local `intent_corpus`
  (Banking77 held out: 184 seen labels, 71 banking-labeled holdout
  intents — note this is 71, not 77, because dataset_v7's cross-source
  label-collision merge already folded a handful of Banking77 names into
  other sources' canonical labels; the "77-way" language elsewhere in this
  file is the source dataset's own size, not always this corpus's count)
  and reweighted `mcq_corpus` (HellaSwag downweighted as designed).

**Not yet run**: any real training (exp7a/7b/7c/7d), the cardinality sweep,
external benchmarks, or `scripts/build_exp7_data.py` itself (its network
calls — MNLI/BoolQ/Yelp/SST-5/DBpedia/TREC/Yahoo/GoEmotions/typed-decisions —
were not exercised in this session; expect to debug real HF dataset
schema/config-name issues on first run, per that script's own "best-effort"
disclaimer). The Colab CPU probe script is where the real backbone gets
downloaded and checked for the first time.
