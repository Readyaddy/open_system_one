# Experiment 1: Backbone Ablation

## The one-sentence question

Was iteration 4's worse paraphrase generalization (vs. iteration 3) caused
by the **poly-encoder architecture**, or by us happening to pick a
**weaker pretrained backbone**, or both? This experiment isolates the
backbone variable by changing nothing else.

## What changes vs. the iteration 4 baseline

Exactly **one thing**: the pretrained checkpoint used for both encoders.

| | iteration 4 (baseline) | experiment 1 |
|---|---|---|
| Architecture | Poly-encoder | Poly-encoder (identical code) |
| Backbone | `sentence-transformers/all-roberta-large-v1` | `Alibaba-NLP/gte-large-en-v1.5` |
| Dataset | CLINC150+Banking77+SNIPS, 234-way fixed classification | identical |
| Fine-tuning stack | frozen layers, layer-wise LR decay, 8-bit AdamW, label smoothing, etc. | identical |

Everything else — data, splits, training loop, evaluation code, loss
function, learning rate schedule shape — is copy-pasted from iteration 4
unchanged. That's the whole point of an ablation: if only one thing
differs, any difference in outcome can be attributed to that one thing
(modulo random seed noise, which we can't fully rule out with a single
run each, but a change this large would be a big result even with that
caveat).

## Every technical term used here, explained

**Backbone**: the large pretrained transformer that does the actual
language understanding; everything else in our model (the projection
heads, the poly-encoder's attention codes) is a small trainable layer
bolted on top of it. Swapping the backbone means swapping what the model
"already knows" before we've trained it on anything of ours.

**Pretrained checkpoint**: a backbone's weights aren't random — they were
already trained by someone else, usually on a huge amount of text, before
we ever touch them. Different checkpoints were trained on different data,
in different amounts, with different objectives, and therefore encode
different "priors" about language even before our fine-tuning starts.

**`all-roberta-large-v1`**: a RoBERTa-large (355M params) checkpoint,
further contrastively fine-tuned by the `sentence-transformers` project
specifically to produce good sentence embeddings. RoBERTa itself was
pretrained in 2019 on ~160GB of text (BookCorpus, Wikipedia, CC-News,
OpenWebText, Stories) — respectable at the time, small by today's
standards.

**`gte-large-en-v1.5`**: a 2024 embedding model from Alibaba's NLP team
(434M params), trained on a much larger and more recent web-scale corpus,
specifically optimized end-to-end as a text embedder (not adapted
after-the-fact like the RoBERTa checkpoint above). It currently ranks
much higher than `all-roberta-large-v1` on MTEB (the standard public
leaderboard for embedding model quality across retrieval, classification,
and semantic similarity tasks) — this is the concrete, measurable claim
behind calling it a "better" pretrained prior, not just a newer name.

**Paraphrase generalization gap**: (accuracy on the original training-time
description of a class) minus (accuracy on a hand-written paraphrase of
the same class's description, never seen during training). A small gap
means the model responds to *meaning*; a large gap means it's sensitive to
the *exact wording* it happened to train on — i.e. it memorized surface
form rather than generalizing.

**Ablation**: an experiment that removes or swaps exactly one component of
a system to measure that component's specific contribution, holding
everything else fixed. The scientific value comes entirely from changing
only one thing — an experiment that changes the backbone *and* the
architecture *and* the data at once (which is what iteration 4 did
relative to iteration 3) tells you that *something* in that bundle
mattered, but not *which* thing.

## Why this matters

`ANALYSIS.md` flagged this as unresolved: the paraphrase-generalization
regression from iteration 3 to iteration 4 happened at the same time as
three simultaneous changes (bigger/different backbone, poly-encoder
architecture, bigger/harder dataset). We built a mechanistic hypothesis
(the poly-encoder's extra attention flexibility gives it more surface
area to overfit exact phrasing) but never isolated it. This experiment is
the cheapest possible way to rule in or rule out one of the three
confounded variables.

## What would confirm or refute what

- **If the gap shrinks back toward iteration 3's ~0.05** with the better
  backbone (architecture unchanged): backbone quality was a real
  contributor, maybe the dominant one. The poly-encoder-overfitting
  hypothesis becomes less necessary to explain the iteration 4 result
  (though it could still be a smaller, secondary effect).
- **If the gap stays close to iteration 4's ~0.14** despite the better
  backbone: backbone quality is *not* the explanation, and the
  poly-encoder-architecture hypothesis gets stronger (nothing else
  differs between this run and iteration 4 except the thing that didn't
  help).
- **Either result is useful.** This is exactly the kind of experiment
  where a "negative" result (the swap didn't help) is as informative as a
  positive one, because it eliminates a candidate explanation rather than
  just adding another number to a pile.

## Files in this folder

- `model.py` — poly-encoder model code, identical to `src/model_v4.py`
  except the `BACKBONE` constant.
- `train.py` — training script, identical to `src/train_v4.py` except it
  imports from this folder's `model.py`.
- Results get appended to the bottom of this file once the run completes.
