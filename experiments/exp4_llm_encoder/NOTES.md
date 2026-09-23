# Experiment 4: LLM-derived Encoder Backbone

## Status note

Experiment 1 (backbone ablation with `e5-large-v2`) was stopped early at
epoch 6/30 — its val_acc had plateaued flat with iteration 4's own
trajectory (93.7% vs 94.0% at the same epoch), showing no sign of a clear
win on the easier metric, so continuing it to get the harder
paraphrase-gap number stopped seeming worth the wait. This experiment
replaces it as the active backbone-quality test, going one step further:
not just a better *encoder-family* checkpoint, but a checkpoint derived
from an actual large language model.

## The one-sentence question

Does a backbone with dramatically more pretraining scale and world
knowledge — a decoder LLM converted into an embedding encoder — do
meaningfully better than either of our two prior backbones (RoBERTa-large,
e5-large-v2), on both in-scope accuracy and paraphrase generalization?

## What changes vs. iteration 4 / experiment 1

| | iteration 4 | experiment 1 | experiment 4 |
|---|---|---|---|
| Architecture | Poly-encoder | Poly-encoder | Poly-encoder (identical code) |
| Backbone | `all-roberta-large-v1` (355M) | `e5-large-v2` (335M) | `gte-Qwen2-1.5B-instruct` (1.5B) |
| Backbone origin | encoder pretrained from scratch (RoBERTa, 2019), then embedding-tuned | encoder pretrained from scratch (BERT-style, 2023), purpose-built for embeddings | **decoder LLM (Qwen2) pretrained on a much larger, more recent corpus, then converted to bidirectional attention and contrastively tuned for embeddings** |
| Everything else (data, splits, fine-tuning stack) | — | identical | identical |

## Every technical term used here, explained

**Decoder-only LLM**: the architecture family behind GPT, Llama, Qwen,
Mistral, etc. Each token can only attend to tokens *before* it in the
sequence (causal / left-to-right attention) — this is what makes them
good at generating text one token at a time, but it also means a decoder
LLM's raw hidden states aren't naturally suited to producing one good
summary vector for a whole piece of text, the way an encoder's
bidirectional attention is.

**Encoder (bidirectional) vs. decoder (causal) attention**: an encoder
(BERT, RoBERTa, ModernBERT) lets every token attend to every other token
in both directions when building its representation — well suited to
"understand this whole piece of text." A decoder restricts attention to
only look backward — well suited to "predict what comes next," which is
what generation needs. Everything we used before experiment 4 (MiniLM,
mpnet, RoBERTa, e5) was an encoder from the start.

**Converting a decoder into an encoder**: `gte-Qwen2-1.5B-instruct` takes
the pretrained Qwen2 decoder LLM and (a) turns off the causal attention
mask so every token can see every other token, then (b) further trains it
contrastively (the same kind of "pull matching pairs together, push
mismatched pairs apart" objective our whole project uses) specifically to
produce good embeddings. This general recipe is published and reasonably
well established (see: LLM2Vec, GritLM, NV-Embed, E5-Mistral in the
research literature) — Alibaba's team did this conversion for us, so we
get the benefit (a decoder LLM's much larger pretraining) without having
to run that conversion training ourselves.

**Why "more world knowledge" is a specific, checkable claim, not just
marketing**: decoder LLMs in general are trained on much larger and more
diverse text corpora than the encoder-only models used in experiments 1-3
and iteration 4 — this backbone is 1.5B parameters (4-4.5x bigger than
either prior backbone) and its base model (Qwen2) was pretrained on
trillions of tokens of recent, broad text. More parameters and more/better
pretraining data is what "more world knowledge" concretely cashes out to
here — not a vague claim.

**`trust_remote_code`**: this checkpoint ships its own custom Python model
code (not one of the architectures built into the `transformers` library),
which `transformers` will download and execute when asked
(`trust_remote_code=True`). This is the same mechanism that caused
`gte-large-en-v1.5`'s bug during experiment 1's original backbone attempt
(a custom RoPE implementation threw garbage index errors outside its
expected code path) — worth being alert to the same risk-class of failure
here, and verifying the model actually produces sane output on a real
forward pass before committing a full training run to it (see smoke-test
results below, appended once run).

## Why this matters

If experiment 4 clearly outperforms both prior backbones on **paraphrase
generalization specifically** (not just in-scope accuracy, which is a
different, easier question), that's real evidence the previous two
backbones' pretraining scale — not the poly-encoder architecture, and not
the fixed 234-way benchmark task shape — was the dominant bottleneck all
along. If it doesn't move the paraphrase number despite dramatically more
scale, that's evidence the bottleneck is architectural (the poly-encoder
scoring mechanism itself, per the `ANALYSIS.md` hypothesis) or
data-shape-related (experiment 2's question), not backbone quality at
any achievable scale.

## Pivot: full backbone freeze (linear probing), not partial fine-tuning

The first full-scale attempt (partial freeze, top ~1 layer trainable,
chunked outcome-bank encoding to fix an OOM) technically ran without
crashing, but at ~5.4 hours/epoch -- impractical on this hardware. Rather
than shrink the model (which would dilute the actual thing this
experiment tests: does more LLM-scale pretraining help), the backbone is
now **frozen entirely** -- all 28 layers plus the token embeddings. Only
the poly-encoder's lightweight, always-separate components stay
trainable: the 16 learned "code" query vectors and their attention layer,
the projection heads, and the temperature scalar.

**Why this fixes the speed problem, not just works around it**: when
every parameter in a stack of operations has `requires_grad=False`,
PyTorch's autograd never builds a computation graph through it at all --
there is nothing to differentiate, so no backward pass, no gradient
checkpointing recomputation, no optimizer state for the backbone's 3B
frozen parameters. The frozen backbone becomes exactly as cheap as a
plain inference forward pass; only the small trainable heads sitting on
top of its output need any gradient bookkeeping.

This is a real, standard technique (usually called linear probing or
frozen-feature extraction) for exactly this situation: a pretrained
backbone is too large or too valuable to fine-tune cheaply, so only a
lightweight adapter on top is trained. The tradeoff, stated plainly: the
backbone's own representations never adapt to this specific task -- all
of the learning has to happen in the poly-encoder's query/attention/
projection layers reading from a fixed, frozen feature space. Whether
that's enough signal for this task to work well is itself part of what
this run will show.

## Practical risk, stated plainly

This is a bigger model (1.5B vs. 335-355M) on the same 12GB GPU. Fitting
it requires more aggressive layer freezing and/or a smaller batch size
than iteration 4 used — this will be tuned during the smoke test before
the full run, not guessed at.

## Files in this folder

- `model.py` — poly-encoder model code, adapted from experiment 1's, with
  the backbone swapped and freeze-layer logic adjusted to match this
  checkpoint's actual module structure (confirmed before writing the
  freeze code, not assumed).
- `train.py` — training script, adapted from experiment 1's.
- Results get appended to the bottom of this file once the run completes.
