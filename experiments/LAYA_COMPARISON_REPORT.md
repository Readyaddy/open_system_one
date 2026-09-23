# Laya comparison report — architecture, training, our results, and data reuse

## Executive summary

While reverse-engineering TypeSafe's "Jev" System-1 decision model has been this
project's whole premise, someone else already shipped a real, open-source,
Apache-2.0-licensed attempt at the same target: **Laya**, by Convai Innovations
([HF](https://huggingface.co/convaiinnovations/laya),
[GitHub](https://github.com/receptron/laya)). Its own GitHub description calls
it "the open-source Jev-compatible System-1 decision model," and it matches
Jev's `system_one` request/response API shape to 4 decimal places.

We pulled its actual source code (not just marketing pages) to understand the
real architecture and training method, ran our own exp6 checkpoint on the same
public benchmarks Laya/Jev were compared on, and checked what data is legally
reusable. Bottom line: **Laya wins decisively on genuine zero-shot
classification** (our model trails by 40-90 percentage points depending on
task); **our fine-tuned high-cardinality performance beats their zero-shot
number on Banking77**, but that's not an apples-to-apples comparison since we
trained on that data and they didn't; and **there's a small, real, Apache-2.0
dataset (`LocalLLaMA/typed-decisions`) we can legally pull in** as another MCQ
training source, closer to the actual target domain (business decision
routing) than anything currently in `mcq_corpus`.

---

## 1. What Laya is

- **Backbone**: ModernBERT-large (395-421M params) for English; mmBERT-base
  (322M) for a 100+-language multilingual variant.
- **Checkpoints**: `laya` (general), `laya-multilingual`, and
  `laya-typed-decisions` (fine-tuned specifically on the typed-decisions
  benchmark's own training split).
- **License**: Apache 2.0, weights and code both, published on Hugging Face
  and GitHub.
- **Positioning**: explicitly a drop-in, faster (33-40ms vs. Jev's
  236-276ms), open alternative to TypeSafe's closed Jev API.

## 2. Architecture (verified from actual source code, not just docs)

Pulled and read `rl_common.py` (the real model definition, hosted alongside
the weights on their HF repo) and `sequence.ts` (a faithful TypeScript port of
their Python request-building logic, in the `receptron/laya` inference repo).

### Sequence layout

Every question gets packed into **one sequence**:

```
[CLS] <type> question: <instructions> [SEP] [MASK] <option0 text> [MASK] <option1 text> ... [SEP] <state> [SEP]
```

Each candidate option gets its own `[MASK]` token immediately before its
rendered text. The actual data being decided about (the ticket, the email,
the JSON payload) goes **last**, after every option. This repurposes BERT's
own masked-language-model pretraining mechanism directly: instead of
predicting a vocabulary word at the `[MASK]` position (what the position was
originally trained for), they read out a *scalar score* there instead — reuse
of a mechanism the backbone already deeply understands, not a bolted-on new
one.

### Decision head (on top of the ModernBERT backbone)

- `nn.Embedding(3, d)` — a learned embedding for question type
  (choice/score/noul), added into the representation.
- **2 more `nn.TransformerEncoderLayer`s** stacked after the backbone's own
  output — a second, smaller round of self-attention specifically for the
  decision task.
- **Marker-gather**: `torch.gather` pulls the hidden state at each option's
  `[MASK]` position out of the sequence.
- **Scorer**: `LayerNorm → Linear → GELU → Linear(→1)` turns each gathered
  option vector into one scalar logit; softmax (padding masked to `-1e4`)
  gives the probability distribution over that question's actual options.
- **Separate act/escalate head**: pools the `[CLS]` token, concatenates 4
  hand-engineered features from the answer distribution itself (top
  probability, margin between top-2, entropy, option count), feeds that
  through its own small MLP — a genuinely separate signal from the answer
  itself, deciding whether the system should act autonomously or escalate.

### Full sequence → answer flow

One ModernBERT forward pass (context, instructions, and every option's mask +
text all attend to each other) → 2-layer decision-head self-attention →
gather each option's mask-position vector → per-option MLP scorer → masked
softmax → per-cardinality temperature scaling → final calibrated
probabilities.

## 3. Training — RLCD (Reinforcement Learning for Calibrated Decisions)

Not ordinary cross-entropy classification. The reward for a predicted
distribution `q` against the true answer is a **strictly proper scoring
rule** — mathematically, the only way to maximize expected reward is to
report your *honest* belief:

- **Log score**: `(target · log q).sum()` — standard log-likelihood.
- **Spherical score**: `(target · q).sum() / ||q||`, weighted 0.5.
- **Ranked probability score (RPS)**, ordinal `score` questions only:
  `Σ(cdf_q - cdf_target)²`, weighted 1.0 — specifically rewards getting an
  ordinal answer *close* even when not exact, which log score alone can't
  capture.

Combined: `reward = log_score + 0.5·spherical − 1.0·RPS·(if ordinal)`. This
drives a **GRPO-style policy-gradient update** (group-mean-baseline
REINFORCE) — the model is trained to directly maximize calibration-aware
reward, not to minimize cross-entropy.

For multi-turn conversations, **TD(λ=1.0)** bootstraps soft targets across a
conversation's prefixes — each earlier turn's target is pulled toward the
model's own belief at the *next* turn.

Calibration is fit **separately per (question type, option-count bucket)** —
`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, etc. Real fitted
temperatures from their published config are revealing: `choice:11+` gets
**0.10** (heavy sharpening) while `choice:2` gets **1.91** (heavy softening)
— direct evidence their raw logits get wildly overconfident specifically as
option count grows.

## 4. The concrete, code-verified reason Laya struggles at high cardinality

`head_max_len` (192-256 tokens) is a **shared, fixed token budget covering
the instructions AND all options combined** — not per-option, total. Their
own sequence-builder code shrinks every option's text evenly if the full set
doesn't fit. For Banking77's 77 options, that leaves roughly **3-4 tokens per
option** — not enough to meaningfully distinguish 77 fine-grained banking
intents from each other. Their published 0.425 accuracy on Banking77 isn't a
mysterious generalization failure; it's a **direct, mechanical consequence of
packing every option into one fixed-budget sequence**. This is exactly the
wall a joint-sequence architecture hits at high cardinality that a
dual-encoder (our own exp6 approach) structurally avoids — we encode each
option independently, so a 77-way candidate set costs the same per-option
token budget as a 4-way one.

## 5. Our results vs. Laya vs. Jev

Ran our exp6 checkpoint (`exp6_latest.pt`, epoch 16) on the same public
benchmarks, using `eval_external_benchmarks.py`:

| Task | Ours (exp6) | Laya | Jev | Chance |
|---|---|---|---|---|
| AG News (4-way) | **50.4%** (zero-shot) | 95.0% | 91.0% | 25% |
| DAIR Emotion (6-way) | **26.25%** (zero-shot) | 59.5% | 48.0% | 16.7% |
| Banking77 (71-77-way) | **89.35%** (⚠️ trained on it) | 42.5% (zero-shot) | 87.0% (zero-shot) | ~1.3% |

**Honest reading**: on genuine zero-shot classification (AG News, DAIR
Emotion), we're clearly behind — both tasks show a 25-45 percentage point
gap. Still meaningfully above chance (2x and 1.6x respectively), so the
model isn't doing nothing, but nowhere near Laya/Jev's level. Expected:
Laya's joint-sequence attention lets context and options directly interact
through the transformer; ours is dual-encoder compatibility scoring, a
structurally weaker mechanism for pure classification. Laya is also
purpose-built and RLCD-trained specifically for calibrated decisions; ours
splits capacity across three simultaneous objectives (intent + QQP + MCQ).

The Banking77 number is **not a fair comparison** — Banking77 is literally
one of our five training data sources, so 89.35% measures in-distribution
fine-tuned performance, not zero-shot generalization the way Laya's 42.5%
and Jev's 87.0% are. It does show something real: high-cardinality
classification isn't fundamentally unsolvable for our architecture *given
real training data on that task* — matching what section 4 above already
predicts, since our dual-encoder doesn't share Laya's fixed-token-budget
bottleneck.

## 6. Can we copy their data?

**Yes, partially, legally.** Checked license and provenance directly:

- **`LocalLLaMA/typed-decisions`** ([HF dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)) —
  **Apache 2.0**, publicly downloadable via `datasets.load_dataset`. 1.6k
  rows (~2,000 decisions) across four synthetic business workflows: customer
  service, invoice processing, security incidents, agent-trace
  observability. Fields include `state` (context), `questions`, `gold`
  labels, and structured prediction targets (outcome/action/risk/urgency).
  **This is directly reusable** — and notably closer to the actual target
  domain (business decision routing) than anything currently in our
  `mcq_corpus` (RACE/SciQ/ARC/CommonsenseQA/HellaSwag are all reading-
  comprehension/trivia-style, not decision-routing-style). Could be added as
  another MCQ-style training source the same way RACE/SciQ were.
- **Their code** (`rl_common.py`, the sequence-building logic, the RLCD
  reward/training utilities) is **also Apache 2.0** — legally reusable if we
  wanted to experiment with their RLCD/proper-scoring-rule training paradigm
  ourselves instead of plain cross-entropy.
- **What's NOT confirmed public**: whatever larger base-pretraining/RLCD
  corpus was used to train the *general* `laya` checkpoint (not the
  typed-decisions-specific fine-tune). One source mentioned "~30k questions"
  for a training notebook, but we found no published dataset at that scale —
  only the ~2,000-decision typed-decisions set is confirmed available. Don't
  assume a bigger hidden corpus is copyable; only the one dataset link above
  is actually verified.

## 7. Recommendations

1. **Pull in `LocalLLaMA/typed-decisions`** as a new `mcq_corpus` source (or
   a dedicated small auxiliary set) — small, but the only dataset in reach
   that's actually decision-routing-flavored rather than generic QA.
2. **The dual-encoder vs. joint-sequence tradeoff is now concretely
   understood, not just theorized**: joint-sequence (Laya, our own
   unfinished experiment 3) wins on raw classification accuracy at low-to-
   moderate cardinality, but hits a hard, mechanical wall at high
   cardinality from shared token budget. Dual-encoder (our exp6) avoids that
   specific wall structurally, at the cost of not letting context and
   options directly attend to each other, which likely explains our much
   weaker zero-shot numbers on AG News/Emotion. A real next experiment:
   finish what experiment 3 started, now with a concrete, evidenced
   understanding of exactly where it will and won't hold up.
3. **RLCD (proper scoring rules + policy gradient) is worth trying** as an
   alternative to our current cross-entropy losses, specifically for the
   calibration angle — our own zero-shot decline story has always been about
   generalization, but we've never directly optimized for calibrated
   honesty the way RLCD does.
