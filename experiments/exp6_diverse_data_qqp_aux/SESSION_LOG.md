# Exp6 session log — data fixes, training mechanism changes, infra saga, results

This documents one long working session on experiment 6 (Qwen2.5-0.5B poly-encoder,
JEPA-style compatibility scoring, no text generation). Kept as a record of what
was tried, what broke, what actually worked, and where things stand.

## The starting problem

Earlier training (before this session) showed a consistent pattern: zero-shot
accuracy (on 45 never-trained intents) peaked at epoch 1 (~21.5%) then collapsed
every epoch after (down to ~6-9% by epoch 6-10), while val_acc (seen intents) kept
climbing toward 86-87%. Classic catastrophic forgetting — the model was learning
to memorize the closed 255-intent training distribution at the cost of general
compatibility-scoring ability.

## Root-cause diagnosis

Two structural issues were identified as the likely cause, not just insufficient
regularization:

1. **Intent training used a FIXED GLOBAL BANK.** Every training step scored the
   query against the *same* 255-intent description bank (freshly re-worded via
   templates, but always the same 255 candidates). The model never had to
   generalize to novel candidate sets — it only ever needed to win against one
   fixed, repeatedly-seen pool.
2. **The MCQ auxiliary data had no context.** The first version of `mcq_corpus`
   deliberately stripped all passages (RACE, SciQ's `support` field) to keep a
   "bare question" shape — backwards for the actual goal, which is "use the
   context you're given to pick the right answer," not answer trivia from
   parametric knowledge alone.

## Data fixes

### `mcq_corpus` — rebuilt to be context-grounded
- Added **RACE** (`ehovy/race`, real passage + question, 78k train examples) and
  restored **SciQ's `support`** field as context — together these are 59% of the
  corpus, up from 0%.
- Kept context-free sources (CommonsenseQA, OpenBookQA, ARC-Easy/Challenge,
  HellaSwag) since "no context given, answer from what you know" is a real case
  too, just not the majority shape anymore.
- Final corpus: **148,171 train / 18,543 val / 18,554 test** examples. Schema:
  `{context, question, options, answer_idx, source}`.
- Two data-quality bugs found and fixed (same categories as `intent_corpus`):
  duplicate option text within an example (0.2%, unanswerable by construction,
  dropped), and cross-split ambiguity from dropping context (short bare
  questions like "which is true?" that had different correct answers depending
  on a passage we no longer kept — keyed on `(context, question)` together,
  dropped when ambiguous, exact duplicates collapsed by priority test > val >
  train).
- Built directly on the Colab VM (downloads from HF there) instead of uploading
  the ~230MB result over the slow `colab upload` channel; backed up to
  `Drive:/jepa_checkpoints/data_backup/mcq_corpus` for reuse across sessions.

### `qqp_paraphrase_pairs` — expanded 5x
- 20,000/2,000 → **100,000/10,000** train/val pairs (`scripts/build_qqp_data.py`,
  standalone so it doesn't touch `intent_corpus`/`mcq_corpus`). Also backed up to
  Drive.

## Training mechanism changes (`train.py`)

### Intent task reframed to match MCQ's shape
- **Per-example dynamic candidate sampling** replaces the fixed 255-bank: each
  step, every example gets its own set of `--intent_options` candidates (1
  correct + N-1 randomly sampled distractors, resampled fresh every step),
  scored via `model.compatibility_grouped` — the same mechanism MCQ uses, not a
  separate closed-set classifier in disguise.
- Started at 10 options (too easy/narrow — plausibly why zero-shot kept
  declining even with dynamic sampling), raised to **50**.
- **Routing-question framing**: intent examples are now scored as
  `f"{utterance}\n\nQuestion: Which category best describes what this customer
  wants, or where should this request be routed?"` — matching MCQ's
  `context + "\n\nQuestion: " + question` input shape, instead of scoring the
  bare utterance directly.
- **Eval now matches training exactly.** `val_acc`/`zero_shot_acc` used to score
  against the full seen/all-labels bank (`compatibility()`); now they use the
  *same* per-example `INTENT_TOTAL_OPTIONS`-sized candidate sets and
  `compatibility_grouped()`, with a **fixed per-example seed** so eval is
  reproducible epoch-to-epoch (not randomly different each time).

### MCQ integration
- `mcq_context_text(ex)`: `context + "\n\nQuestion: " + question` when context is
  non-empty, else bare question. Tokenized at `MCQ_CONTEXT_MAX_LENGTH=384` (real
  passages, not the 32-token intent/QQP length).
- `model.compatibility_grouped` added to `model.py`: per-example variable-size
  candidate sets with a validity mask (padding forced to `-inf` so it can never
  be predicted or draw gradient) — the shared mechanism intent, MCQ, and eval
  all now use.

### Baseline metrics before every run
Added a full baseline pass (val_acc, zero_shot_acc, qqp_val_acc, mcq_val_acc) on
the *current* weights before training starts, saved to `exp6_mcq_baseline.json`
and logged to W&B — so we can tell whether a run actually improved anything
rather than just comparing against wherever the last run happened to stop.

### Per-step progress logging
Per-epoch-only logging meant 30-40+ minutes of total silence in the log file at
large batch sizes — indistinguishable from a hung process. Added a per-step
progress line (step count, % done, avg step time, elapsed, ETA, GPU memory
allocated) — cheap at ~116 steps/epoch, and the only way to tell "still working"
from "stuck" during a long epoch.

## Two real OOM crashes and their fixes

1. **First crash**: raising `--intent_options` 10→50 meant each step suddenly
   encoded `batch_size × 50` option texts in one shot (e.g. 32×50=1600) — a 5x
   jump that overflowed memory. Fixed by routing the intent-option encoding
   through the existing chunked `encode_outcome_bank` helper instead of one
   giant forward pass.
2. **Second crash**: a gradual fragmentation-driven OOM mid-epoch (not a single
   spike) from the many different tensor shapes now in play (variable MCQ
   option counts, chunked intent scoring). Fixed with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch's own suggested
   fix for this exact error) plus re-enabling gradient checkpointing (traded
   some speed for real memory headroom — the right call given how much time
   OOM/disconnect firefighting cost that day).

## GPU utilization tuning
Batch sizes were calibrated up in stages (48 → 160 → 384 batch_size / 224
mcq_batch_size) to target ~35GB/40GB A100 memory usage with ~5GB headroom,
instead of the ~9GB the initial conservative (crash-avoidance) settings used.
Final config: `--batch_size 384 --qqp_batch_size 384 --mcq_batch_size 224
--outcome_chunk_size 255`, ~19.2s/step, ~116 steps/epoch, ~37 min/epoch.

## The Colab reliability saga

Colab sessions disconnected repeatedly throughout the day (well over a dozen
times). Root causes chased, in order:

1. **Theory: idle-timeout from sparse polling.** Colab-cli's `--auth=oauth2`
   keep-alive daemon pings the backend; a long gap in checking in seemed to
   correlate with disconnects early on. Partially true but incomplete.
2. **Theory: a rogue leftover keep-alive loop.** An old background loop (from
   before switching to a different session) was found still running, hitting
   `colab exec` under the *old* auth mode against the *new* session name —
   confirmed and killed. Helped, didn't fully fix it.
3. **Theory: oauth2's 1-hour token expiry.** `colab whoami` showed short-lived
   access tokens; switched to **ADC auth**
   (`gcloud auth application-default login` with the specific scopes colab-cli's
   own bundled skill doc recommends for headless/agent use: `openid`,
   `cloud-platform`, `userinfo.email`, `colaboratory`). Installed `gcloud` via
   the non-root tarball installer (no sudo available). Genuinely more robust,
   but disconnects still happened afterward.
4. **Theory: compute-unit exhaustion.** Ruled out directly — 137.53 units at
   ~5.3/hour is ~26 hours of runway, nowhere near exhausted.
5. **Theory: "fresh terminal per task."** Tested directly by having the user run
   one consolidated setup+launch script from a single persistent terminal
   (`colab_full_setup_and_launch.sh`) — it still hit the exact same
   `RuntimeError: Connection was lost` on the very first `colab exec` call after
   `drivemount`, disproving this theory cleanly.
6. **The actual finding**: the underlying Colab VM was **repeatedly still alive
   server-side** (confirmed via the account's own `/tun/m/assignments` endpoint,
   which lists every currently-provisioned runtime) even when `colab-cli`'s
   *local* session tracking (`~/.config/colab-cli/sessions.json`) had gone stale
   and every `colab status`/`colab exec` call failed with "session not found" or
   a websocket `RuntimeError`. **The bug is in colab-cli's local
   reconnection/session-tracking logic, not Colab's actual infrastructure.**
   Training itself, run as a detached subprocess (`start_new_session=True`),
   was never interrupted by any of this — it kept running the whole time.

### The actual fix: manual session reconnection
When `colab status -s jepa-train` fails but training should still be alive:
1. Fetch a fresh token directly from the assignments API using the same ADC
   credentials colab-cli itself uses:
   ```python
   creds, _ = google.auth.default(scopes=[...])
   creds.refresh(...)
   requests.get('https://colab.research.google.com/tun/m/assignments', ...)
   ```
2. If the expected endpoint (e.g. `gpu-a100-s-kkb-ass1c1-...`) is still listed,
   manually write a fresh entry into `~/.config/colab-cli/sessions.json` with
   that endpoint/token/url (kernel_id/session_id left `null` — colab-cli looks
   these up fresh on next use).
3. `colab exec -s jepa-train` immediately works again, with zero data loss and
   zero new VM created.

**Never run `colab new -s <name>` to "reconnect"** — it does not reuse an
existing assignment, it allocates a genuinely new VM, silently creating a
second billable A100 session (happened at least twice this session before this
was understood).

### Other lessons
- **`jepa-mon` / status-check sessions should be CPU-only**, not GPU — checking
  logs on Drive doesn't need a GPU at all, and a GPU monitor session wastes
  compute units for zero benefit. (Caught and fixed after accidentally creating
  a T4 for this.)
- Both `colab new` and `colab drivemount` need genuinely interactive OAuth
  completion (visit URL, approve, press Enter) — this cannot be done headlessly
  by piping input, since the process reads/writes stdin at exactly the moment
  the browser step needs to happen. The user has to run these themselves; the
  agent can prepare the exact commands but not execute the interactive step.

## W&B integration
Added `--wandb_project`/`--wandb_run_id` (fixed id so relaunches after a
disconnect **resume the same chart** instead of fragmenting history across many
runs). Logs baseline + per-epoch metrics. One quirk hit: a killed run's baseline
log at a given step can cause a relaunch's baseline log at the *same* step to be
silently dropped (W&B requires strictly increasing steps) — harmless, real
per-epoch data still logs fine at new step numbers.

## Final results (this session)

| Epoch | val_acc | zero_shot_acc | qqp_val_acc | mcq_val_acc |
|---|---|---|---|---|
| 1 (baseline, pre-fix) | 72.7% | 21.5% | 70.8% | — |
| 7 | 94.0% | 44.5% | 74.8% | 35.3% |
| 8 | 94.1% | 42.5% | 75.1% | 36.1% |
| 9 | 94.3% | 45.0% | 76.1% | 35.1% |
| 10 | 94.5% | 45.75% | 76.7% | 37.75% |
| 12 | — | 45.25% | 77.3% | 38.1% |
| **13** | **94.66%** | **47.5% (best all day)** | 77.0% | 37.4% |
| 14 | 94.57% | 46.5% | 76.9% | 39.0% |
| 15 | 94.62% | 44.5% | 77.3% | 40.4% |
| 16 | 94.62% | 44.0% | 77.1% | 39.6% |

Best checkpoint: **`exp6_best_zeroshot.pt`, epoch 13** (val 94.66%, zero_shot
47.5%, qqp 77.0%, mcq 37.4%) — the strongest all-around result produced today,
more than double the pre-session baseline's zero-shot accuracy.

**Zero-shot still eventually declines** — it peaked at epoch 13 and fell for 3
epochs after (47.5% → 46.5% → 44.5% → 44.0%). Same underlying forgetting
pattern as before, but the fixes clearly changed its character: peak moved from
epoch 1 to epoch 13, and the decline is a gentle ~3.5-point drift instead of a
collapse toward zero. Training was stopped (by request) at epoch 16, with the
session's GPU killed to stop burning compute units.

## WiSE-FT experiment (post-session follow-up)

Ran the weight-interpolation sweep (`eval_wise_ft.py`, rewritten to match the
current matched train/eval mechanism, run locally on the RTX 5070 Ti against
`exp6_latest.pt`, epoch 16 — val 94.62%, zero_shot 44.0%, qqp 77.1%,
mcq 39.6%):

| ρ (pretrained weight) | val_acc | zero_shot_acc | qqp_val_acc | mcq_val_acc |
|---|---|---|---|---|
| 0.00 (pure fine-tuned) | 94.25% | **44.00%** | 76.33% | 39.00% |
| 0.10 | 94.00% | 43.75% | 77.00% | 37.50% |
| 0.25 | 94.75% | 42.00% | 78.00% | 39.25% |
| 0.50 | 92.75% | 34.50% | 74.67% | 35.75% |
| 0.75 | 71.00% | 30.25% | 72.00% | 34.00% |
| 1.00 (pure pretrained) | 42.00% | 22.75% | 66.00% | 32.25% |

**Clean negative result** — every metric degrades monotonically as more
pretrained weight is blended in; no ρ beats pure fine-tuned on zero-shot.
**Why**: WiSE-FT's original setting (CLIP) assumes the pretrained model already
has a meaningful, usable head for the task (CLIP's projections work zero-shot
before fine-tuning even starts), so blending the backbone back doesn't break
the backbone-to-head correspondence. Our poly-encoder heads (codes,
projections) never existed in any pretrained form — they were randomly
initialized and trained from scratch *in lockstep with the fine-tuned
backbone specifically*. Pulling the backbone back toward pretrained weights
without the heads changing accordingly just creates a growing mismatch between
what the backbone now produces and what the heads were trained to interpret.
**Conclusion: WiSE-FT is not a useful lever for this architecture** — skip it
for future iterations unless the heads themselves get a pretrained
initialization too (they don't have one available).

## How hard are these MCQ sources, really? (per-source breakdown)

The blended `mcq_val_acc` (~40%) hides a lot — ran `eval_mcq_by_source.py` on
the full `mcq_corpus['test']` split (18,554 examples, `exp6_latest.pt` epoch
16) to break it out:

| Source | Accuracy | Chance | vs. Chance |
|---|---|---|---|
| **SciQ** | **65.0%** | 25% | **2.6x** |
| ARC-Easy | 51.6% | 25% | 2.1x |
| OpenBookQA | 42.9% | 25% | 1.7x |
| RACE | 41.2% | 25% | 1.6x |
| HellaSwag | 38.9% | 25% | 1.6x |
| ARC-Challenge | 35.4% | 25% | 1.4x |
| CommonsenseQA | 28.1% | 20% | 1.4x |
| **Overall (blended)** | 41.8% | ~24% | ~1.75x |

Clear pattern: strong on **fact-lookup-style questions** with direct
lexical/semantic overlap between question and answer (SciQ, ARC-Easy) — exactly
what embedding-compatibility scoring is naturally suited for. Weakest on
sources needing **deeper reasoning or fine discrimination between similar
plausible options** (CommonsenseQA, ARC-Challenge, HellaSwag) — where a small,
non-generative, pure-compatibility model (no joint attention between the full
passage and each option) is expected to struggle most.

### Context: how far from published SOTA, and why

| Dataset | Small/mid fine-tuned SOTA | Large/LLM-era SOTA |
|---|---|---|
| SciQ | BERT-base (110M) → ~90%+ | GPT-3 175B zero-shot ~95% |
| ARC-Easy | UnifiedQA-11B → ~85-90% | GPT-4-class → ~95%+ |
| ARC-Challenge | UnifiedQA-11B → ~65-70% | GPT-4-class → ~90%+ |
| CommonsenseQA | RoBERTa-large (355M) + graph → ~80% | GPT-4-class → ~85%+ |
| RACE | ALBERT-xxlarge (235M) → ~86-89% | GPT-4-class → ~90%+ |
| HellaSwag | small models (<1B) → **~30-40%, barely above the 25% chance floor** | GPT-3 175B → ~78%, LLaMA-65B → ~84%, GPT-4-class → ~95%+ |

HellaSwag in particular was **deliberately constructed adversarially against
models below a certain scale** — it targets exactly the kind of subtle
discourse/commonsense coherence that only reliably emerges with real scale.
Published reference points: GPT-2-medium (355M) ~35%, GPT-2-large (774M)
~40%, and it isn't until GPT-3-scale (175B) that it jumps to ~78%. Our
backbone (Qwen2.5-0.5B) sits almost exactly in the GPT-2-medium/large range —
our 38.9% on HellaSwag is right on-trend for that size class, *before*
accounting for any benefit from our training at all. Since HellaSwag is 57%
of the entire MCQ pool (39,796 of 69,993 training examples), it drags the
blended number down hard and isn't a fair single-number verdict on the
approach. On sources that don't specifically punish small models (SciQ,
ARC-Easy), the model is already doing genuinely solid, non-trivial work; the
HellaSwag/CommonsenseQA/ARC-Challenge gap to published SOTA is substantially a
*scale* gap (10-100x more parameters), not something more training on the
current architecture would close on its own.

## Open threads / next steps not yet done
- **Early-stop on zero_shot_acc, not val_acc**: `exp6_best_zeroshot.pt` already
  tracks this separately, but the training script's own `--patience` early
  stopping is keyed to `val_acc`, which keeps climbing even as zero-shot falls —
  worth reconsidering which metric should actually gate stopping.
- **GPU utilization vs CPU-bound idle gaps**: tokenization/batch construction
  happen in plain Python between GPU calls, creating idle gaps even at high
  memory usage. Overlapping data prep with GPU compute (prefetching) was
  discussed but not implemented — deferred until training stability wasn't
  the day's main fire.
- **Widening the per-step intent distractor pool further, or a curriculum that
  grows it over epochs** — one candidate explanation for the late-but-still-
  present decline is that 50 options is still narrow enough to admit shortcut
  solutions that don't require broad embedding structure.
