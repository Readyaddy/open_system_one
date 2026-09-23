# Experiment 2: Small Option-Set Reframing

## The one-sentence question

Does training against realistic small candidate sets (3–8 options, like
every real Jev API example) instead of one giant fixed classification
task change how well the model discriminates and generalizes — separate
from any architecture change?

## What changes vs. the iteration 4 baseline

| | iteration 4 (baseline) | experiment 2 |
|---|---|---|
| Architecture | Poly-encoder | Poly-encoder (identical code) |
| Backbone | `all-roberta-large-v1` | `all-roberta-large-v1` (identical — deliberately, see below) |
| Task shape | one shared bank of 199 candidate descriptions; every example scored against all 199 at once | **each example gets its own sampled set of 6 candidates** (the correct one + 5 distractors); scored only against those 6 |
| Distractor sampling | n/a | mix of "hard" (same source dataset, semantically close) and random |

Backbone is held at iteration 4's choice on purpose, **not** experiment
1's winner — if experiment 2 used a different backbone too, any change in
results could come from the backbone or the task shape, and we'd be back
to confounding variables. This is why experiments 1 and 2 both branch
directly off iteration 4 independently, rather than experiment 2 building
on top of experiment 1's result.

## Every technical term used here, explained

**Fixed classification / closed-set classification**: every example is
scored against the *same* full list of possible classes (here, all 199
trained intents), every time. This is the standard way benchmarks like
CLINC150 and Banking77 are normally evaluated, and it's what iterations
1–5 all did.

**Candidate set / option set**: the specific list of choices offered for
a *single decision*. In Jev's actual API, this is defined per-request by
the developer (e.g. `{"billing": ..., "technical": ..., "sales": ...}`)
— typically 3–6 items, capped at 255. It is emphatically **not** "every
intent the model has ever seen," because a real application usually only
cares about a handful of relevant outcomes for a given decision point.

**Distractor**: a wrong-but-plausible option included alongside the
correct answer, so the model has to actively discriminate rather than
just recognize the right answer in isolation. Without distractors,
"classification" degenerates into a much easier task (is this text
compatible with THIS ONE description, yes or no).

**Hard negative**: a distractor deliberately chosen to be *close* to the
correct answer (here: another intent from the same source dataset, e.g.
another Banking77 intent when the correct answer is also Banking77),
because it's semantically more likely to be confused with the right
answer than a randomly chosen one. Training with hard negatives is a
standard technique in contrastive/retrieval learning — models trained
only against easy (randomly dissimilar) negatives often learn a much
coarser, less useful notion of similarity, because "different enough from
a random unrelated thing" is a much lower bar than "different enough from
something that's actually similar."

**Closed-set vs. open-set framing**: closed-set means the model can
implicitly rely on "the answer is one of these fixed K things I've seen
labeled before." Open-set / per-request framing means the model has to
treat the option list itself as part of the input to be understood, not a
fixed target vocabulary — much closer to how a real Choice/Score question
in Jev's API works, where option text is arbitrary and defined at
request time.

## Why this matters

This is the piece of the earlier discussion that's true regardless of
which architecture we eventually prefer: a 234-way benchmark and a
6-option realistic request are genuinely different tasks, and treating
"solves the 234-way benchmark" as the target may not be the same as
"behaves like a well-calibrated small-decision model." Two concrete
reasons to expect a difference:

1. **Difficulty composition changes.** In a 234-way softmax, most of the
   234 candidates on any given step are *trivially* wrong (totally
   unrelated domains) — only a handful are genuinely close. In a 6-option
   set built with hard negatives, most or all of the distractors are
   deliberately close. The effective difficulty-per-decision goes up even
   though the raw option count goes down.
2. **What "generalization" would even mean changes.** Paraphrase
   generalization in the closed-set framing was measured by swapping in a
   different description for a *known* class inside the same fixed
   234-way pool. In the small-set framing, every request's pool is a
   fresh sample — generalizing well now means correctly discriminating
   among *whichever* small set of candidate descriptions shows up, not
   memorizing a fixed pool's geometry at all. This is arguably a strictly
   harder and more honest generalization test.

## What would confirm or refute what

- **If accuracy on the small-set task is high and paraphrase-style
  generalization checks (see below) hold up well**: task framing alone,
  independent of architecture, might be enough to get good behavior — the
  earlier regression may have had as much to do with "234-way fixed
  benchmark" training as with the poly-encoder itself.
- **If accuracy is high on trained-style option sets but degrades sharply
  on sets built with more/harder hard negatives, or on paraphrased option
  text**: the small-set framing alone isn't sufficient — supports that the
  earlier scoring-architecture hypothesis (poly-encoder's flexibility)
  still matters independently.
- Compare directly against **experiment 3**, which uses this exact same
  data framing but the joint-sequence architecture — the cleanest
  available comparison for isolating architecture-family effects, since
  data framing is held constant between experiments 2 and 3.

## Files in this folder

- `dataset.py` — small-option-set request builder, adapted from the
  `dataset_v6.py` draft, reused here without the joint-sequence-specific
  formatting (no marker tokens needed for a poly-encoder).
- `model.py` — poly-encoder model code (same as experiment 1's, RoBERTa
  backbone).
- `train.py` — training loop modified for per-example candidate sets
  instead of one shared per-batch bank (see NOTES below in the file
  itself for the specific mechanical change).
- Results get appended to the bottom of this file once the run completes.
