# Experiment 3: Joint-Sequence, Marker-Token Readout

## The one-sentence question

Does the architecture we deduced Jev most likely uses — one shared
transformer pass over state *and* options together, with full
token-level cross-attention, instead of two separate encoders that only
meet after pooling — actually work, and how does it compare to the
poly-encoder family we've used since iteration 4?

## What changes vs. experiment 2 (not vs. iteration 4 directly)

This experiment is deliberately compared against **experiment 2**, not
directly against iteration 4, because it reuses experiment 2's exact data
framing (small option sets). That keeps the comparison to one variable:
architecture family.

| | experiment 2 | experiment 3 |
|---|---|---|
| Data framing | small option sets, hard negatives | identical |
| Architecture | poly-encoder (two separate encoders, pooled, then attention over pooled codes) | **joint-sequence** (one encoder, one sequence, full token-level self-attention between state and options) |
| Backbone | `all-roberta-large-v1` (512-token context) | `answerdotai/ModernBERT-large` (8192-token context — needed because state+options now share one sequence) |
| Output for Score | discrete 5-way classification, argmax | discrete 5-way classification, **probability-weighted expected value** (produces fractional outputs) |

Backbone *also* changes here, which does mean this experiment isn't a
pure single-variable ablation against experiment 2 — RoBERTa's 512-token
limit makes it unusable for packing state + all options into one
sequence, so there wasn't a way to hold backbone constant and still build
this architecture. This is flagged explicitly rather than glossed over:
if experiment 3 differs from experiment 2, we won't be able to fully
separate "architecture" from "backbone" as the cause without a follow-up
run. What we *can* still learn cleanly: whether the joint-sequence
approach is viable at all, and how its raw numbers compare.

## Every technical term used here, explained

**Joint-sequence architecture**: instead of encoding the context and each
candidate separately (as every previous iteration did) and only combining
them via a similarity score at the end, the context text and all the
candidate option texts are concatenated into **one single sequence of
tokens** and passed through the transformer **together, in one forward
pass**.

**Self-attention**: the mechanism inside a transformer where every token's
representation gets updated by looking at (attending to) every other
token in the same sequence. In a joint sequence, this means a token from
the *context* can directly attend to a token from *one of the candidate
options*, and vice versa — something that never happens in a dual-encoder
(where the context encoder never sees a single token of any candidate,
and the candidate encoder never sees a single token of the context).

**Cross-attention** (as used loosely in earlier discussion): attention
between two different sequences/sources. A true joint-sequence model
doesn't need a *separate* cross-attention mechanism because self-attention
over the single combined sequence already lets every part attend to every
other part — this is actually a simpler mechanism than the poly-encoder's
explicit two-stage attention (context tokens → codes, then candidates →
codes), at the cost of no longer being able to compute the context and
candidates independently of each other.

**Marker token**: a special vocabulary token, added specifically for this
architecture (`[OPT]`, `[NOUL]`, `[LVL]`), inserted into the sequence
right before each candidate option's text, each ordinal level's text, or
the single yes/no question. After the transformer processes the whole
sequence, we don't try to summarize the whole thing into one vector —
instead we look at the specific hidden state *at each marker's position*
and treat that as "the representation of this particular option, having
attended to everything else in the sequence including the context." This
is a standard trick (comparable to how BERT's `[CLS]` token or "span
markers" in extraction models work) for getting several distinct,
addressable outputs out of one shared transformer pass.

**Read-out**: the act of extracting a specific position's hidden state
(here, a marker token's) and passing it through a small head (a linear
layer, in our case) to get a final score. "Reading out" a marker means:
after the big transformer, apply the small comparatively cheap head just
at that one position.

**Cold-start embedding**: the marker tokens (`[OPT]`, `[NOUL]`, `[LVL]`)
are brand new — they weren't in ModernBERT's original vocabulary, so
their embeddings start off **randomly initialized**, carrying zero
pretrained meaning, unlike every other token in the sequence (which
already has a meaningful pretrained embedding). This is a real weakness
worth tracking: the marker tokens have to learn their role entirely from
our (comparatively small) fine-tuning data, which is a similar cold-start
problem to the poly-encoder's learned "code" query vectors in iterations
4–5 (also randomly initialized), so it's not a new risk introduced here,
but it's present here too.

**Probability-weighted expected value** (for the Score output): rather
than just taking the single most likely ordinal level (argmax), we
compute `Σ (probability of level k) × k` across all levels. This is the
same math as an average, weighted by how confident the model is in each
level — it's what produces a fractional answer like `1.035` from a
softmax over discrete levels `[1, 2, 3, 4, 5]`, matching what TypeSafe's
own docs show Jev's Score output looks like.

**Full self-attention cost**: self-attention's compute cost grows with
the *square* of sequence length (a sequence twice as long costs
~4x the compute, not 2x). This is why joint-sequence only becomes cheap
with *small* option sets (experiment 2's data framing) — packing all 199
intents into one sequence for every training step would have made this
prohibitively expensive on our hardware; packing 6 options is not.

## Why this matters

This is the most direct test of the Jev-architecture hypothesis built up
across the earlier conversation: shared context-token budget in their
docs (state + all questions together, one number) is much better
explained by one joint sequence than by two independently-limited
encoders. If this experiment trains successfully and performs
competitively, that's real (not just documentary) evidence the deduction
was on the right track. If it trains poorly or generalizes worse than
experiment 2's poly-encoder, that's evidence either the deduction is
wrong, or that whatever makes Jev's real (undisclosed) implementation
work involves engineering/training details well beyond what we can
recover from public docs alone (a genuinely possible outcome, stated
plainly rather than assumed away).

## What would confirm or refute what

- **If experiment 3 trains stably and gets comparable-or-better accuracy
  and paraphrase/generalization behavior than experiment 2**: the
  joint-sequence hypothesis holds up as a viable, plausible architecture
  for this kind of task — genuine support for the "this is probably close
  to what Jev does" deduction.
- **If it trains but generalizes distinctly worse than experiment 2**:
  consistent with (though doesn't prove) the "more expressiveness trades
  against generalization" pattern from `ANALYSIS.md` — this would be the
  *most* expressive architecture in the whole project (full token-level
  cross-attention beats even the poly-encoder's pooled attention), so if
  that pattern is real, this is where it should show up most clearly.
- **If it's unstable or fails to train well at all** (cold-start marker
  tokens are a real risk here): that's useful information too — it would
  suggest Jev's real implementation, if it resembles this, must include
  training tricks we don't have visibility into (larger-scale synthetic
  data, a dedicated pretraining stage for the marker mechanism, etc.),
  not just "the architecture shape alone."

## Files in this folder

- `dataset.py` — same small-option-set request format as experiment 2's,
  reused directly (adapted from the `dataset_v6.py` draft).
- `model.py` — the joint-sequence model with marker-token readout,
  refined from the `model_v6.py` draft.
- `train.py` — training loop: tokenize the packed sequence, one backbone
  forward pass, gather marker positions, apply the three heads, combined
  loss.
- Results get appended to the bottom of this file once the run completes.
