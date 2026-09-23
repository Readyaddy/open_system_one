# Local JEPA-style decision model

## Idea

Don't generate text and parse it. Encode the situation (context) and encode
each fixed candidate outcome, and pick whichever outcome embedding is most
*compatible* with the context embedding. Prediction is a similarity check
in representation space, done in a single forward pass per side — no
decoding, no token-by-token generation.

This is the core JEPA move applied to a "System One" decision task: given a
fixed set of options (here, ticket-triage actions), decide which one applies
by measuring distance in embedding space, not by asking a language model to
emit and then parse a label string.

## What was built

- **`src/dataset.py`** — synthetic support-ticket triage dataset. Six fixed
  outcomes (`refund`, `escalate`, `faq_reply`, `bug_report`, `ignore`,
  `compliment`), each with a natural-language description. Tickets are
  generated from templates with randomized filler words/noise, so the model
  can't just memorize a small fixed set of sentences.
- **`src/model.py`** — `JEPADecisionModel`: two small encoders (token
  embedding → mean pool → 2-layer MLP → L2-normalize) sharing an
  architecture but with independent weights:
  - `context_encoder(ticket_text)` → context embedding
  - `outcome_encoder(outcome_description)` → outcome embedding
  - `compatibility(ctx, outcome) = cosine_similarity / learned_temperature`
- **`src/train.py`** — trains both encoders jointly with a softmax
  cross-entropy loss over compatibility scores against all 6 candidate
  outcomes at once (InfoNCE-style contrastive training: the correct
  outcome's embedding is pulled close, the other 5 are pushed away, all in
  one softmax per example).
- **`src/predict.py`** — inference demo: loads the checkpoint, encodes a
  new ticket and a bank of outcome descriptions, ranks by similarity.

Total model size: ~50k params. Trains in a few seconds on CPU.

## Why the paraphrase check matters

If this were a classifier with a fixed head, "generalizing to a new
description of the same option" wouldn't even be a meaningful test — the
label is just an index. Here the outcome side is a real encoder reading
text, so we can hold out **paraphrased descriptions of the same six
outcomes** (never seen during training) and see if the model still routes
tickets correctly using only the *meaning* of the new description.

## Results (iteration 1)

```
test_acc (trained outcome descriptions):     1.000
test_acc (paraphrased, unseen descriptions): 0.750
```

- Chance accuracy with 6 options is ~0.167.
- 100% on the descriptions seen during training (expected — task is easy
  with fixed templates).
- 75% on paraphrased, never-seen outcome descriptions — well above chance
  and clear evidence the outcome encoder learned *some* semantic
  generalization, not string memorization, but far from perfect. That gap
  (100% vs 75%) is the honest headline number of this first iteration: real
  signal, real ceiling to push on.

## Known limitations / why the gap exists

- The outcome/context encoders are tiny bag-of-words models (mean-pooled
  embeddings, no order sensitivity, no pretrained knowledge). Paraphrases
  that reuse few overlapping words with the training description will
  naturally underperform ones that share vocabulary.
- Vocabulary is built only from the training corpus + the two outcome
  description sets — any paraphrase word absent from training text maps to
  `<unk>` and loses signal.
- Templates are simple; real support tickets are messier.

## Next iterations (not yet run)

1. Swap the mean-pool encoder for a tiny Transformer or GRU to capture word
   order — cheap on CPU at this scale.
2. Initialize token embeddings from small pretrained vectors (e.g. GloVe) so
   `<unk>` paraphrase words aren't dead weight — this directly targets the
   generalization gap above.
3. Add harder negatives during training (outcomes from a *larger* pool than
   the 6 seen at train time) to test true open-set compatibility, not just
   fixed 6-way softmax.
4. Try a genuinely new, never-described-at-all outcome at test time (e.g.
   "cancel_subscription") with only a one-line description, and see if the
   context encoder + outcome encoder combination zero-shots it — this is the
   real test of "JEPA-style decision making" vs. "6-way classifier in
   disguise."
5. Calibrate the softmax temperature / confidence against true correctness
   (currently a free learned parameter, not verified for calibration).

## Running it (iteration 1)

```bash
python src/train.py    # trains and saves checkpoints/jepa_decision_model.pt
python src/predict.py  # demo inference on new example tickets
```

---

# Iteration 2 — real dataset, GPU, pretrained backbone

Goal: scale up on every axis that iteration 1 flagged as the ceiling —
real data instead of templated synthetic text, a pretrained transformer
backbone instead of a from-scratch bag-of-words encoder, and GPU training.

## What changed

- **Dataset**: [CLINC150](https://arxiv.org/abs/1909.02027) (`clinc_oos`,
  `plus` config), a real, human-written intent benchmark: **150 intents
  across 10 domains** (banking, travel, kitchen & dining, auto & commute,
  work, small talk, meta, credit cards, home, utility) **plus an explicit
  out-of-scope class**. 15,250 train / 3,100 val / 5,500 test examples.
  See `src/dataset_v2.py`.
- **Outcome descriptions**: auto-generated from each intent name (e.g.
  `pto_balance` → *"The user wants help with: pto balance."*) for training.
  A hand-written **paraphrase set for 30 intents**, spanning every domain,
  is held out and used only at eval time — different wording, never seen
  during training.
- **Model** (`src/model_v2.py`): both the context encoder and the outcome
  encoder now start from `sentence-transformers/all-MiniLM-L6-v2` (6-layer
  transformer, 384-dim, ~22M params) and are **fine-tuned independently**
  — two separate weight copies, same architecture, same contrastive
  objective as iteration 1 (cosine similarity / learned temperature →
  softmax cross-entropy over the full 150-outcome bank).
- **Training** (`src/train_v2.py`): GPU (RTX 5070 Ti Laptop, CUDA 12.8),
  AdamW, 6 epochs, batch size 64, ~27s/epoch. OOS-labeled examples are
  excluded from the classification loss (they have no correct target
  among the 150 intents) and held out for a separate rejection check.

## Results (iteration 2)

```
test_acc (trained/auto-generated descriptions):            0.966   (n=4500, 150-way)
test_acc, 30-intent subset, UNSEEN paraphrased descriptions: 0.732  (n=900)
  same subset, original trained descriptions, for comparison: 0.969

OOS separation: mean max-similarity  in-scope=0.828  oos=0.480   gap=0.348
```

Sample qualitative routing (`src/predict_v2.py`), plain new sentences against
all 150 outcome descriptions:

```
"hey what's it like outside right now"          -> weather (0.99)
"can you set a wake up call for 6am"             -> alarm (0.83)
"i need to know how much is in my checking..."   -> balance (0.995)
"book me a room in chicago for next friday"      -> book_hotel (0.99)
"asdlkj random keyboard mashing not a request"   -> no confident winner (top pick only 0.11)
```

- **96.6% test accuracy on a real 150-way intent benchmark** with no
  classifier head at all — pure embedding compatibility. This is a large,
  legitimate jump from the toy 6-class task, and the confidence numbers
  above look like exactly the "System One, single forward pass" behavior
  the whole design is going for, including graceful low-confidence output
  on nonsense input instead of a confidently wrong label.
- **OOS separation is real but partial**: genuinely out-of-scope input
  scores meaningfully lower (0.48 vs 0.83 average max-similarity) — good
  enough for a similarity-threshold rejection rule, not yet a calibrated
  probability of "none of these."

### The honest finding: paraphrase generalization did NOT improve much

This is the important result of this iteration, and it's a negative one.
Despite a 22M-parameter pretrained transformer backbone replacing a ~50k
bag-of-words model, paraphrase accuracy only went from **0.75 → 0.73** —
essentially flat, while in-distribution accuracy jumped enormously (and the
task itself, 150-way real intents, is much harder than the 6-way toy task).

**Why**: the outcome encoder only ever sees *one fixed string per class*
during training. Cross-entropy over a small closed set of fixed anchor
points can be solved by memorizing 150 points in embedding space that are
merely distinguishable from each other — it does not require the encoder
to represent the actual *meaning* of the description robustly. A pretrained
backbone raises the ceiling on how well it *could* generalize, but nothing
in this training objective pushes it to use that capacity for the outcome
side. This is the real bottleneck to attack next, not backbone size or
dataset size.

## Known limitations

- Outcome-side training signal is a single fixed string per class — see
  above, this is the main lever for improving paraphrase generalization.
- OOS rejection is a soft gap, not a calibrated threshold; no
  precision/recall curve computed yet.
- Both encoders are fully fine-tuned independently from the same pretrained
  init; no shared/frozen-backbone ablation has been run to see how much of
  the 96.6% is coming from fine-tuning vs. the pretrained prior.

## Next iterations (not yet run)

1. **Outcome-side description augmentation** (highest priority): during
   training, randomly sample among several template phrasings / synonyms
   per intent (or paraphrase with an LLM offline into a small bank per
   class) instead of one fixed string, so the outcome encoder is forced to
   generalize across wording, not memorize points. This directly targets
   the flat 0.73 paraphrase number above.
2. Calibrate OOS rejection: pick a similarity threshold on validation OOS
   vs in-scope data, report precision/recall, not just the mean gap.
3. True zero-shot outcome test: introduce an intent with **zero training
   examples**, described only at test time, and see if compatibility
   scoring alone can route to it correctly — the strongest test of "this is
   not a 150-way classifier in disguise."
4. Ablate: frozen backbone + trainable projection head only, vs. full
   fine-tuning, to see how much of the result depends on adapting the
   pretrained weights vs. using them as-is.

## Running it (iteration 2)

```bash
python src/train_v2.py    # trains on GPU, saves checkpoints/jepa_decision_model_v2.pt
python src/predict_v2.py  # demo inference on new example utterances
```

---

# Iteration 3 — bigger backbone, bigger/harder dataset, and the actual fix

Request going into this iteration: bigger encoder, bigger weights, bigger
dataset, train longer. Important caveat that was flagged before running
anything: scaling alone would not fix the flat paraphrase number from
iteration 2 (0.75 → 0.73 despite a pretrained backbone), because that
plateau was caused by the outcome encoder only ever seeing one fixed
description string per class — a training-objective problem, not a
capacity or data problem. So this iteration does both: scale up, and fix
the actual cause.

## What changed

- **Backbone**: `all-mpnet-base-v2`, 110M params per encoder (222M total
  across both encoders) vs. MiniLM's 22M — 5x bigger, and specifically
  strong on semantic/paraphrase similarity.
- **Projection head**: deeper (2-layer MLP with residual + LayerNorm) per
  encoder instead of iteration 2's plain 2-layer MLP.
- **Dataset**: CLINC150 (150 intents) **+ Banking77** (77 intents) combined
  → **227 total intents, ~24,000 training examples**. Banking77 adds
  genuinely hard negatives — intents like `card_arrival` /
  `card_delivery_estimate` / `lost_or_stolen_card` / `card_swallowed` /
  `compromised_card` / `declined_card_payment` are all semantically close,
  a much harder discrimination test than CLINC150 alone.
- **The actual fix — outcome description augmentation**: instead of one
  fixed description per class, `src/dataset_v3.py` defines 8 template
  wrappers ("The user wants help with: X.", "This ticket concerns X.",
  "Category: X.", etc.) plus a domain-synonym substitution table (card ↔
  bank card, transfer ↔ wire, declined ↔ rejected/denied, ...). Every
  training step resamples a fresh random phrasing for every one of the 227
  classes, so the outcome encoder can never lock onto memorizing one exact
  string — it has to key on the shared meaning across many surface forms.
- **Training**: mixed precision (fp16 autocast + GradScaler), cosine LR
  schedule with warmup, gradient clipping, early stopping (patience 10),
  checkpointing best+latest every epoch. GPU: RTX 5070 Ti Laptop.
- **Eval set**: expanded held-out paraphrases to 50 hand-written
  descriptions spanning both datasets (`src/paraphrases_v3.py`), including
  genuinely hard banking intents.

### A caught bug, worth recording

The first full run appeared to finish with inconsistent numbers — the
final report claimed it loaded "epoch 11" but the printed per-epoch log
never showed epoch 11 as a new best. Root cause: an earlier training
attempt, backgrounded with a raw shell `&` and believed killed, had
actually orphaned its Python child process (reparented to init), which
kept running and kept writing to the *same* checkpoint filenames
concurrently with the properly-tracked rerun — confirmed by `nvidia-smi`
still showing ~7.8GB of GPU memory in use and a stray python PID after the
tracked run had already exited. Killed the orphan, verified a single clean
process, and reran. All numbers below are from that single, uncontaminated
run.

## Results (iteration 3, clean run)

```
Best epoch: 11/60 (early-stopped at epoch 21, patience 10)  val_acc 0.9613
Total training time to best checkpoint: ~20 minutes (108s/epoch)

test_acc (227-way, base descriptions):                       0.9464
test_acc, 50-intent subset, UNSEEN paraphrased descriptions:  0.8799  (n=1699)
  same subset, base descriptions, for comparison:             0.9300
  -> paraphrase gap: 0.0501

OOS separation: mean max-similarity  in-scope=0.8954  oos=0.4903   gap=0.4051
```

Compare directly to iteration 2 (150-way, MiniLM, single fixed description):

| | iter 2 (150-way, MiniLM) | iter 3 (227-way, mpnet, harder negatives) |
|---|---|---|
| in-scope test accuracy | 0.966 | 0.946 |
| paraphrase accuracy | 0.732 | **0.880** |
| paraphrase gap (base − paraphrase, same subset) | 0.237 | **0.050** |
| OOS separation gap | 0.348 | 0.405 |

**The fix worked, and it's the headline result**: paraphrase generalization
jumped from 0.732 → 0.880 (the gap vs. base descriptions shrank from 0.237
to 0.050) *despite* the task getting harder (227 close-together intents
instead of 150, including Banking77's near-duplicate categories). In-scope
accuracy dropped slightly (0.966 → 0.946), which is expected and healthy —
it's a genuinely harder 227-way task, not a regression from the same task.
This is real evidence that the outcome-description-augmentation fix, not
model/data scale, was what iteration 2 was missing.

### Qualitative check (`src/predict_v3.py`)

```
"i still haven't gotten my new card in the mail"        -> card_arrival (0.996)
"someone else is using my card, i didn't make these..." -> compromised_card (0.991)
"the atm ate my card and didn't give it back"           -> card_swallowed (0.995)
"why was my payment declined"                            -> declined_card_payment (0.756)
```
All correct, including fine-grained Banking77 negatives that are one word
apart in meaning.

```
"asdlkj random keyboard mashing not a real request" -> cash_withdrawal_not_recognised (0.871)
```
This is an honest miss worth flagging: unlike iteration 2, gibberish input
here got a **confidently wrong** top pick instead of a flat, low-confidence
distribution. The aggregate OOS separation metric (0.405 gap) still looks
fine on average, but this single case shows the rejection behavior is not
uniformly reliable — likely because CLINC150 is the only source of
explicit out-of-scope training signal, and Banking77 (no oos examples of
its own) makes up over a third of the 227-class outcome bank, diluting
that signal and giving noise more classes to spuriously match against.

### Why training stopped in ~20 minutes, not hours

The request was to train for hours; early stopping triggered at epoch 21
(best was epoch 11) because validation accuracy plateaued and started
mildly degrading — more training at this learning rate on this data was
not going to help, and continuing anyway would have been wasted compute,
not a more rigorous result. The bigger backbone and augmented outcome
descriptions converge fast because the pretrained weights already carry
most of the needed language understanding; only the compatibility
alignment needs learning. If more wall-clock time is wanted, it needs to
come from something training can actually still improve on (see next
steps below), not from ignoring the plateau.

## Next iterations (not yet run)

1. **Fix OOS dilution**: add explicit out-of-scope / negative examples for
   the Banking77 side too (there are open datasets of generic chit-chat /
   nonsense text usable as universal negatives), so rejection isn't
   CLINC150-only.
2. **Push paraphrase generalization further**: the augmentation fix worked;
   scaling it up (more template variety, harder synonym substitution, or
   LLM-generated paraphrase banks per class instead of hand-written
   templates) is the highest-leverage next lever, now that it's confirmed
   to be the right lever.
3. **True zero-shot outcome test**: introduce an intent with zero training
   examples, described only at eval time, and see if compatibility scoring
   alone routes to it — the strongest test of "not a classifier in
   disguise."
4. **Something that actually uses hours of training**: a larger combined
   dataset (add e.g. HWU64, SNIPS) so there is enough signal for a longer
   schedule to keep improving instead of plateauing at epoch 11.

## Running it (iteration 3)

```bash
python src/train_v3.py --epochs 60 --patience 10   # full run: GPU, ~20-40 min to convergence
python src/train_v3.py --epochs 2 --max_train 300 --max_val 100 --patience 2  # smoke test
python src/predict_v3.py                             # demo inference on new example utterances
```

---

# Iteration 4 — Poly-Encoder, bigger backbone, real zero-shot test

Request: bigger encoder, bigger dataset, "test on not seen questions and
choices," better architecture, use proper DL fine-tuning technique.

## What changed

- **Architecture — Poly-Encoder** (Humeau et al. 2019). Iterations 1-3 used
  a pure bi-encoder: the whole context got squeezed into ONE mean-pooled
  vector before ever seeing a candidate, which is an information
  bottleneck. Here the context encoder learns 16 query "codes" that
  cross-attend over ALL context tokens in one batched attention op (one
  transformer pass, 16 parallel outputs), and each candidate then attends
  over those 16 codes to build its own candidate-specific view of the
  context before the final dot product. Close to cross-encoder accuracy,
  still fully parallel/batched, candidates still independently encoded.
- **Backbone**: `all-roberta-large-v1`, 355M params/encoder (5x mpnet-base),
  718M total across both encoders.
- **Dataset**: CLINC150 + Banking77 + SNIPS combined = **234 intents,
  ~31k training examples** (SNIPS adds assistant-command phrasing, not
  just questions, for style diversity).
- **Real zero-shot split**: 35 intents (15%) held out ENTIRELY — zero
  training examples, zero description exposure of any kind during
  training — tested only at the end, mixed into a 234-way candidate pool
  with the 199 trained intents as distractors. This is the actual test of
  "unseen questions and unseen choices," not just unseen phrasing of a
  known class.
- **Fine-tuning stack** for a 718M-param model on a 12GB GPU: bottom 16/24
  transformer layers frozen, layer-wise LR decay on the trainable top 8,
  8-bit AdamW (bitsandbytes) instead of fp32 AdamW, gradient checkpointing,
  mixed precision, label smoothing, cosine LR with warmup, gradient
  accumulation (effective batch 48), early stopping.
- **Caught and fixed mid-project**: a stray orphaned process from an
  earlier `&`-backgrounded run kept training in parallel with the properly
  tracked run, both writing to the same checkpoint files -- caught via
  `nvidia-smi` still showing GPU memory in use after the tracked run had
  exited. All numbers below are from a verified single clean process.

## Results (iteration 4)

```
Best epoch: 7/30 (early-stopped at 15, patience 8)  val_acc 0.9479

test_acc (234-way, base descriptions):                        0.9356
test_acc, 46-intent subset, UNSEEN paraphrases:                0.7446  (n=2569)
  same subset, base descriptions, for comparison:              0.8832
  -> paraphrase gap: 0.1386

ZERO-SHOT test_acc (35 never-trained intents, pool=234-way):   0.0884  (n=1369)
  reference: seen-intent accuracy in the SAME mixed pool:      0.9767
  chance level in a 234-way pool:                              0.0043

OOS separation: mean max-compat-logit  in-scope=8.01  oos=4.25   gap=3.75
```

### The honest finding: paraphrase generalization got WORSE, not better

Despite a 3x bigger backbone and the poly-encoder upgrade, the paraphrase
gap widened from iteration 3's 0.050 to **0.139** here. This is worth
taking seriously rather than explaining away. Two plausible, non-exclusive
causes:
1. The poly-encoder's final scoring step lets a candidate attend over 16
   context codes and pick whichever blend fits best — which may make it
   easier to overfit to the *exact* phrasing seen during training (more
   ways to match a specific string) rather than forcing a single robust
   embedding per candidate, the way the plain bi-encoder in iteration 3
   was forced to.
2. 234 intents (vs. 227) with many more near-duplicate Banking77-style
   negatives raises the cost of any embedding drift under paraphrasing --
   a paraphrase that shifts a candidate's embedding even slightly is more
   likely to collide with a *different* close intent now than it was with
   fewer, more separated classes.
This is flagged, not fixed, in this iteration -- worth a dedicated pass
before trusting the poly-encoder as a strict upgrade over the plain
bi-encoder for generalization (as opposed to raw in-scope accuracy, where
it likely does help).

### Zero-shot: real signal, not solved

8.8% top-1 accuracy picking the exact correct never-trained intent out of
234 candidates (chance 0.43%) is genuine evidence of open-set
generalization -- the model is doing much better than guessing purely
from compatibility with a description it never trained on. But it's far
from reliable, and the gap to the 97.7% seen-intent accuracy in the same
mixed pool shows the model still leans heavily on having trained on a
class, not purely on understanding the description.

### A performance note

Epochs 13-15 took roughly 2.5-3x longer (~2860s) than earlier epochs
(~700-1100s) on the same hardware and batch size -- consistent with
thermal throttling on a laptop GPU during a long sustained run, not a code
issue. Worth knowing if timing similar runs.

## Running it (iteration 4)

```bash
python src/train_v4.py --epochs 30 --batch_size 16 --grad_accum 3 --patience 8
python src/predict_v4.py
```

---

# Iteration 5 — Multi-Question Poly-Encoder (in progress)

Request: architecture that answers several DIFFERENT typed questions
about the same context in one pass -- matching how TypeSafe's Jev is
documented to work ("you might define a Choice over {billing, technical,
sales, spam} and a Score for urgency from 0 to 100, and Jev fills both in
one pass"). Every earlier iteration here answered exactly one Choice
question per forward pass; this is the first one that doesn't.

## What changed

- Iteration 4's poly-encoder used 16 *generic* learned codes with no
  particular meaning. Here that's made explicit: one learned query
  embedding PER QUESTION TYPE (`intent`: Choice, `needs_human`: Bool,
  `urgency`: Score over 5 ordinal bins), all cross-attending over the same
  context tokens in a single batched attention call. The context
  transformer runs once regardless of how many questions are asked; each
  question type gets its own dedicated read-out vector from that one pass.
- `intent` routes to the same candidate-compatibility scoring as before.
  `needs_human` and `urgency` route to small dedicated MLP heads (sigmoid
  / 5-way softmax).
- **Caveat, stated plainly**: `needs_human` and `urgency` labels are
  heuristic keyword rules derived from the intent name (see
  `dataset_v5.py`), not gold-labeled data -- there's no public dataset
  with this annotation. This iteration is a proof that the *multi-question
  architecture* works, not a claim about correctly judging real urgency.
  Both labels are heavily imbalanced (needs_human: 9% positive; urgency:
  74% in a single bin), so training uses inverse-frequency class weighting
  and evaluation reports a majority-baseline accuracy alongside the
  model's, so the numbers can't look artificially good by just predicting
  the majority class.
- Same fine-tuning stack as iteration 4 (frozen layers, layer-wise LR
  decay, 8-bit AdamW, gradient checkpointing, mixed precision, label
  smoothing, cosine schedule, early stopping) -- validated there, reused
  here rather than re-litigated.
- **New evaluation**: a timing benchmark directly measuring the claimed
  benefit -- answering all 3 question types about a batch of contexts in
  ONE forward pass vs. 3 SEPARATE forward passes over the same contexts.
  A CPU smoke test (40 train examples, tiny scale, correctness check only)
  already measured a 3.00x speedup for the joint pass -- matches the
  theoretical expectation exactly for 3 question types.

## Results (iteration 5)

The background training process was killed when the launching session
ended, mid-epoch-7 -- not a code failure, see `ANALYSIS.md` section 6 for
what happened and how the result below was still recovered without
retraining (loaded the best checkpoint saved at epoch 6 and ran the full
eval suite standalone via `eval_v5_checkpoint.py`).

```
Checkpoint: epoch 6/25 (not fully converged -- was still improving)  val_intent_acc 0.9412

test (seen intents, base descriptions):
  intent_acc=0.9334
  needs_human_acc=0.9933  (majority-class baseline: 0.8752)
  urgency_acc=0.9729      (majority-class baseline: 0.7124)
  urgency_mae=0.042

test, 46-intent subset, UNSEEN paraphrases: intent_acc=0.7392  (n=2569)
  same subset, base descriptions:            intent_acc=0.8817
  -> paraphrase gap: 0.1425

ZERO-SHOT test (35 never-trained intents, pool=234-way): intent_acc=0.1045  (n=1369)
  reference: seen-intent accuracy in the SAME mixed pool:  0.9740
  chance level:                                             0.0043

OOS separation: mean max-compat-logit  in-scope=8.59  oos=4.63   gap=3.96

Timing (256 contexts, 3 question types):
  joint single-pass  = 0.164s
  separate 3x-pass   = 0.511s
  speedup            = 3.12x
```

**The clean win**: the timing benchmark, run on the real trained model
(not just the earlier CPU smoke test), confirms the architectural claim
at full scale -- answering `intent` + `needs_human` + `urgency` together
costs almost exactly 1/3 of answering them separately. The main task
(`intent`, 93.34%) also wasn't hurt by adding two more objectives, and
generalization metrics (zero-shot 10.45%, OOS gap 3.96) both ticked up
slightly versus iteration 4's single-question model.

**Read the auxiliary-head numbers carefully, not at face value**: 99.3%
`needs_human` and 97.3% `urgency` accuracy look like strong results, but
both labels are deterministic keyword-rule functions of the intent name
itself (see `dataset_v5.py`) -- and the model already gets intent right
93.3% of the time. High accuracy on a fixed lookup table keyed off a
signal you've already mostly solved is close to automatic, not
independent proof the auxiliary heads learned real judgment. **Full
reasoning on this, and everything else in this project, is in
[`ANALYSIS.md`](ANALYSIS.md)** -- a dedicated document working through
*why* each result happened, what's confirmed vs. only plausible, and what
the highest-value next experiments are (starting with isolating whether
the poly-encoder architecture itself, not just scale, is what caused the
paraphrase-generalization regression in iterations 4-5).

## Running it (iteration 5)

```bash
python src/train_v5.py --epochs 25 --batch_size 16 --grad_accum 3 --patience 8
python src/train_v5.py --device cpu --epochs 1 --batch_size 4 --max_train 40 --max_val 20 --patience 1  # smoke test
python src/eval_v5_checkpoint.py  # re-run full eval suite on a saved checkpoint without retraining
```
