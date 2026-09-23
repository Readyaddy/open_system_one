# Data

This folder holds the materialized, versioned output of `dataset_v7.py`
and `paraphrase_aux.py` (in `src/`) — built once via
`scripts/build_data_v7.py` so the data is inspectable as plain files
rather than only existing inside a HuggingFace `datasets` cache.

Regenerate with:
```bash
python scripts/build_data_v7.py
```

## Why this data exists

Earlier iterations of this project (v1–v6) trained on CLINC150 + Banking77
+ SNIPS — three datasets, ~31k examples, all short voice-assistant-style
utterances collected in similar ways. Two problems came up when we tried
full-fine-tuning a larger backbone on that data: (1) not enough raw
examples for a bigger model, and (2) the "paraphrase diversity" the model
ever saw came from 8 fixed template sentences, not real varied writing —
risking the model learning to be invariant to *those 8 templates*
specifically, not genuine paraphrase invariance, while also risking
catastrophic forgetting of whatever broad pretrained knowledge motivated
using a bigger backbone in the first place. This data build fixes both:
more independent sources, more raw examples, and a genuine human-written
paraphrase-pair dataset as a separate training signal.

## `intent_corpus/` — the main decision task

Five real, independently-collected intent-classification datasets,
combined into one corpus with a fixed 15%-of-intents zero-shot holdout
(same methodology as prior iterations, for comparability).

| Source | What it is | Raw unique utterances | Kept in final corpus (after dedup) |
|---|---|---|---|
| CLINC150 (`clinc_oos`, plus config) | 150 intents, 10 broad domains (banking, travel, kitchen, auto, work, small talk, credit cards, home, utility, meta) + an explicit out-of-scope class | 23,845 | 20,384 |
| Banking77 (`mteb/banking77`) | 77 fine-grained banking intents, deliberately hard near-duplicate negatives (e.g. `card_arrival` vs `card_delivery_estimate`) | 13,069 | 12,455 |
| SNIPS (`benayas/snips`) | 7 assistant-command intents (AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent) | 14,306 | 10,616 |
| HWU64 (`FastFit/hwu_64`) | 64 intents across ~21 domains (alarm, calendar, cooking, email, IoT, music, news, ...), separately crowdsourced | 11,033 | 9,467 |
| Amazon MASSIVE en-US (`SetFit/amazon_massive_intent_en-US`) | 60 intents, 18 domains, Amazon's own crowdsourced assistant corpus | 16,432 | 14,469 |

### Two real data-quality problems found and fixed while building this

**1. Cross-source label collisions (57 found, all fixed).** HWU64 and
Amazon MASSIVE turned out to share the same underlying intent taxonomy
design (MASSIVE's schema explicitly extends the earlier research data
HWU64 also derives from) — 55 of the 57 collisions were HWU/MASSIVE pairs
with the literal same intent name (e.g. `hwu::alarm set` /
`massive::alarm_set`), plus 1 banking/clinc coincidence
(`exchange_rate`) and 1 three-way overlap (`play_music`, across CLINC,
HWU, and MASSIVE). Treating these as separate classes would have been a
real labeling bug — the model would be taught a false distinction between
two names for one concept, and if one sibling landed in the zero-shot
holdout while the other stayed in training, the "zero-shot" test would
have been silently contaminated. **Fix**: exact-name matches across
sources are merged into one canonical label, pooling both sources'
examples under it — this is a genuine improvement, not just a
correction, since HWU64 and MASSIVE were collected by different
crowdworkers, so pooling their phrasings of the same concept adds real
surface-form diversity under one *correct* label.

**2. Cross-split text leakage (1,260 exact-text duplicates between train
and test found, all fixed).** Five independently-collected datasets
turned out to have real overlapping utterances — 1,220 cases were the
same text with the same label (straightforward leakage: the model could
memorize these instead of generalizing), and a further 40 were the same
text with *conflicting* labels across sources (the deeper case the
exact-name merge didn't catch, e.g. `clinc::weather` vs
`hwu::weather query` for the literal phrase "what is the weather like" —
same concept, different naming convention). **Fix**: every exact text is
kept in only the highest-priority split it appears in (test > zero-shot
eval > out-of-scope > validation > train), dropped from every
lower-priority one. After this fix: **zero** text overlap between any two
splits, verified directly (see `scripts/build_data_v7.py`'s counterpart
checks; the same checks are reproducible by loading the `.jsonl` files
here and comparing text sets).

### Final numbers (after both fixes)

```
train:            44,429 examples
val:                6,335 examples
test:              10,983 examples
test_oos:           1,000 examples  (CLINC150's explicit out-of-scope class)
test_zero_shot:     2,043 examples  (45 intents, ZERO training exposure)

total distinct intents: 300
  seen (trained on):    255
  zero-shot (held out):  45
```

### Files

- `train.jsonl`, `val.jsonl`, `test.jsonl`, `test_oos.jsonl`,
  `test_zero_shot.jsonl` — one JSON object per line, fields `text`
  (string) and `label` (string, `"<source>::<intent_name>"`, e.g.
  `"banking::card_arrival"`; a merged label uses whichever source's name
  sorted first alphabetically, see the fix above).
- `labels.json` — `seen_labels` (the 255 labels examples are ever trained
  on), `all_labels` (all 300, `seen_labels` + the 45 zero-shot ones),
  `zero_shot_labels` (the 45 held out entirely).

### Known limitation, stated plainly

Outcome/candidate descriptions for the Choice task are still
auto-generated from intent names via 8 fixed templates + a small synonym
dictionary (`dataset_v7.sample_outcome_description`) — this data build
fixes the *context* side's diversity and scale, not that. The QQP data
below is what targets the description-side paraphrase problem directly.

## `qqp_paraphrase_pairs/` — auxiliary paraphrase signal

Real, human-written question pairs from Quora (`SetFit/qqp`), subsampled
from the full ~363k/40k train/validation splits. Not derived from
anything in `intent_corpus/` — a genuinely independent signal, meant to
teach the same encoders actual paraphrase invariance directly, rather
than relying solely on the 8 synthetic templates above.

```
train.jsonl:  20,000 pairs, 37.2% positive (is_paraphrase=true)
val.jsonl:     2,000 pairs
```

Fields: `text1`, `text2` (strings), `is_paraphrase` (bool). Verified: no
empty texts, no exact-duplicate pairs in the subsample.

### How this is meant to be used

Mixed into training as an auxiliary objective alongside the main intent
task, through the *same* context/outcome encoders: `text1` and `text2`
encoded the same way as a context/candidate pair would be, with a
contrastive objective pulling `is_paraphrase=true` pairs' embeddings
together and pushing `is_paraphrase=false` pairs' apart. Wired into
`experiments/exp6_diverse_data_qqp_aux/train.py` since the exp6 launch.

## `mcq_corpus/` — the actual target task

Intent classification (above) is a proxy task: a fixed 255-way closed
label set. The real target capability is more general — given a query
(and, whenever available, a passage of CONTEXT it depends on) and a
small set of candidate answers, pick the correct one, where the
candidate set is different for every example, not one shared global
bank. This is the same compatibility-scoring mechanism, just with a
per-example bank instead of intent_corpus's single shared one.

**Revision note:** the first version of this corpus deliberately
excluded every context/passage field (RACE entirely, SciQ's `support`)
to keep one "bare question" task shape. That was backwards for the
actual goal — the target capability is specifically *use the context
you're given* to pick the right answer, not answer trivia from
parametric knowledge alone. This version restores context wherever a
source naturally has it, and context-grounded examples are now the
majority (59.3%) of the corpus, not zero.

Seven independent, real MCQ datasets, all pooled from every officially
labeled split (some official test splits are unlabeled/hidden for a
leaderboard — CommonsenseQA and HellaSwag — those are excluded, not
included with fake labels), then given our own deterministic 80/10/10
train/val/test split per source (fixed seed):

| Source | What it is | Options per Q | Has context? |
|---|---|---|---|
| RACE (`ehovy/race`, "all") | English-exam reading comprehension (real passage + question) | 4 | **Yes — real passage** |
| SciQ (`allenai/sciq`) | Science facts, generated from a passage | 4 | **Yes — `support` passage** |
| CommonsenseQA (`tau/commonsense_qa`) | Commonsense reasoning | 5 | No |
| OpenBookQA (`allenai/openbookqa`) | Elementary science facts | 4 | No |
| ARC-Easy (`allenai/ai2_arc`) | Grade-school science | 3–5 | No |
| ARC-Challenge (`allenai/ai2_arc`) | Harder grade-school science | 3–5 | No |
| HellaSwag (`Rowan/hellaswag`) | Commonsense "what happens next" | 4 | No (the `ctx` field IS the question, no separate passage) |

The context-free sources aren't a mistake — "no extra context was given,
answer from what you know" is a legitimate case the same skill needs to
cover too — they're just deliberately no longer the majority shape of
the corpus.

### Two real data-quality problems found and fixed (same categories as intent_corpus)

**1. Duplicate option text within one example (0.2%).**
Upstream defect, mostly CommonsenseQA/OpenBookQA: if two options are the
literal same string, a text-only embedding model cannot distinguish "the
correct one" from "the identical-text wrong one" — unanswerable by
construction. Dropped outright.

**2. Cross-split ambiguity from bare, context-free questions.** Some
short context-free questions ("which is true?", "more sunlight will be
absorbed by") appear verbatim across multiple different source items
that had DIFFERENT correct answers depending on information that source
didn't carry forward. This is keyed on `(context, question)` together,
not question text alone — a real passage (RACE, SciQ) automatically
resolves what a bare question couldn't, since two examples only collide
if BOTH the passage and the question are identical. Unlike intent_corpus's
leakage (same text, same true label — fixable by split-priority
reassignment), a genuine `(context, question)` collision with different
answers is unresolvable given the fields kept, so every example under it
is dropped everywhere. Remaining exact duplicates (same context+question,
same correct answer, genuinely just repeated) are then collapsed by
priority test > val > train, same convention as intent_corpus. Verified
after cleaning: 0 duplicate-option examples, 0 ambiguous (context,
question) pairs, 0 such pairs appearing in more than one split.

### Final numbers (after cleaning)

```
train: 148,171 examples  (RACE 78,078 · SciQ 10,901 · CommonsenseQA 8,649 ·
                           OpenBookQA 4,566 · ARC-Easy 4,126 · ARC-Challenge 2,055 ·
                           HellaSwag 39,796)
val:    18,543 examples
test:   18,554 examples

with real context (RACE + SciQ): 109,809 of 185,268 total (59.3%)
```

Context length (when present) is mostly 500–2,000 characters (~roughly
150–500 tokens), with a long tail up to ~6,000 characters for a handful
of the longest RACE passages.

### Files

`train.jsonl`, `val.jsonl`, `test.jsonl` — one JSON object per line,
fields `context` (string, **empty when the source has none** — always
present as a key, never missing), `question` (string), `options` (list
of strings, 3–5 items), `answer_idx` (int, index into `options`),
`source` (string, one of the seven above).

### How this is meant to be used

Unlike `intent_corpus` (one global 255-item outcome bank shared across
the whole batch) each MCQ example carries its OWN small candidate set,
so batched training needs padding: tokenize each example's options,
pad every example in the batch to the batch's max option count, mask
the padding positions to `-inf` before the softmax/cross-entropy so
they can never be predicted or contribute gradient. The context side's
input text is built as `context + "\n\nQuestion: " + question` when
context is non-empty, else just `question` — tokenized with a longer
max length than the short-utterance intent/QQP inputs, to accommodate
real passages (truncated beyond that length; no passage-retrieval or
sentence-selection step exists yet, a known limitation for the longest
passages). Meant to be trained as a primary objective alongside the
intent task and QQP, through the same context/outcome encoders.
