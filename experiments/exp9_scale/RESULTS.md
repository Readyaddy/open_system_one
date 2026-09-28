# Experiment 9 — Results and Explanation

**Open System-1 (OpenJev) at ~5B parameters: what worked, what didn't, and why**

Run `exp9a`, trained 2026-09-27/28 on a Colab A100 (40 GB). W&B: project `open-system-one`, run `exp9a`.
Weights: `MyDrive/openjev/runs/exp9a/best_bf16.pt`. Design notes: [`NOTES.md`](NOTES.md).

---

## 1. Summary

Experiment 9 scaled the Open System-1 decision model from 0.4B to **4.92B parameters**. It
replaced the ModernBERT-large backbone with **Qwen3-Embedding-4B**, packed each answer
option into its own isolated branch, and scaled up every component of the decision head:
the context pooling, the recurrent "rethinking" block and the scratchpad. It used the same
audited training data and the same calibration-first training objective as experiment 8.

The run gives three clear answers:

| Question | Answer |
|---|---|
| Does a larger embedding backbone improve decisions on tasks the model never trained on? | **Yes, by a wide margin.** Held-out accuracy improved by 5 to 33 points across every benchmark. The previous model, exp8, had not moved on these benchmarks at all. |
| Does injecting a precomputed option vector into the option slot help? | **No.** With only the vector available, the model scores at chance at every option count. With both vector and text, it scores the same as text alone. The model learned to switch the vector off. |
| Do repeated "rethinking" passes over the decision help? | **No.** Accuracy is the same (82.3–82.4%) after 1 pass as after 6. |

On the held-out benchmarks, exp9a now **beats Laya** (the open-source Jev-compatible
model) on Banking77, DAIR Emotion, and all three JevBench tiers. It **beats Jev** on DAIR
Emotion. It is still behind Jev on Banking77, AG News and the JevBench hard tier.

---

## 2. What was trained

### 2.1 Model

| Component | Setting | Parameters |
|---|---|---|
| Backbone | Qwen3-Embedding-4B (frozen weights), adapted with LoRA rank 64 on all linear layers | 4.02B frozen + 0.13B LoRA |
| Option packing | One shared prefix (instructions + context), then one **isolated branch per option**. Each branch can see the prefix and itself, never another option, and every branch starts at the same position id. | — |
| Vector path | Qwen3-Embedding's own pretrained sentence embedding of each option, projected into the input space and added at the option's marker token, scaled by a learned gate | ~13M |
| Context pooling | Perceiver-style resampler: 128 latents × 2 layers | part of head |
| Rethinking block | 4 decoder layers, run 6 times with shared weights; each pass re-reads the whole backbone output | part of head |
| Scratchpad | 8 gated memory slots written after every pass and readable by later passes | part of head |
| Decision head, total | | 0.77B |
| **Total** | | **4.92B** (0.90B trained) |

### 2.2 Data

The exp8 corpora, rebuilt and audited on 2026-09-24/25 (see `data/exp7_manifest.json`,
`data/exp7_data_audit.json`):

| Family | Question type | Training rows | Sources |
|---|---|---|---|
| Intent | choice (2–128 options) | 36k | CLINC150, HWU64, SNIPS, MASSIVE (Banking77 held out) |
| MCQ | choice | 131k | RACE, SciQ, CommonsenseQA, OpenBookQA, ARC, HellaSwag (shortcut-filtered) |
| Bool | yes/no, half rendered as descriptive criteria | 57k | BoolQ, MNLI, PAWS, QQP (bias-reweighted) |
| Score | ordinal, 5 levels | 34k | Yelp, Amazon, SST-5, STS-B |
| Diversity | choice, 27 label vocabularies | 73k | DBpedia, Yahoo, 20 Newsgroups, TREC, GoEmotions, tweet tasks |

Never trained on: Banking77, AG News, DAIR Emotion, typed-decisions, JevBench.

### 2.3 Training

| | |
|---|---|
| Objective | RLCD-style: maximise a proper scoring reward (log score + 0.5 × spherical score − RPS for ordinal questions), summed over all 6 rethinking passes |
| Steps | 1,450 × 32 examples = **~46k examples** (about 14% of one pass over the data) |
| Optimiser | AdamW; LoRA and head learning rate 1e-4, gate 3e-3, cosine decay |
| Hardware and time | 1× A100 40 GB, **2.4 hours**, ~7.5 s per step, 31–35 GB GPU memory |
| Starting point | resumed from its own step-250 checkpoint after an out-of-memory crash was fixed (see §6) |
| Evaluation | one final evaluation on 7,277 validation examples plus the modality sweep and depth curve |

The run was shortened from the planned 4,000 steps to fit the remaining Colab credits.

---

## 3. Results

### 3.1 Held-out benchmarks (never trained on)

These are the numbers that matter most: they measure whether the model can make decisions
over label sets it has never seen.

| Benchmark | exp7a (0.4B, start) | exp8b (0.4B, best) | **exp9a (4.9B)** | Laya | Jev |
|---|---|---|---|---|---|
| Banking77, 71-way routing | 55.3 | 55.0 | **73.3** | 42.5 | 87.0 |
| AG News, 4-way | 74.5 | 73.2 | **84.2** | 95.0 | 91.0 |
| DAIR Emotion, 6-way | 58.9 | 57.7 | **63.0** | 59.5 | 48.0 |
| Zero-shot intents, 50 options | 70.6 | 75.7 | **90.0** | — | — |
| typed-decisions: choice | 23.2 | 27.8 | **61.0** | — | — |
| typed-decisions: yes/no | 59.6 | 57.2 | **71.0** | — | — |
| typed-decisions: yes/no, options swapped | — | 57.2 | **71.0** | — | — |
| typed-decisions: ordinal | 33.2 | 31.8 | **47.0** | — | — |

### 3.2 JevBench (official protocol: criteria as options, full state as context)

| Tier (public half) | exp7a | exp8b | **exp9a** | Laya | Jev |
|---|---|---|---|---|---|
| easy (48 items) | 95.8 | 97.9 | **97.9** | 94.4 | 100 |
| original (72 items) | 72.2 | 68.1 | **91.7** | 72.9* | 99* |
| hard (111 items) | 30.6 | 35.1 | **48.6** | 34.1 | 74.1 |

\*Laya and Jev figures are from the JevBench v1.2 leaderboard, measured on the full tier;
the "original" tier is compared with the leaderboard's "standard" column. Ours are on the
public half, so the comparison is indicative. Every encoder of 0.6B or less on the
leaderboard scores 31–40% on the hard tier, and scores above 55% only appear with
LLM-based systems of about 4B and up. exp9a's 48.6% sits between those groups.

Per question type, exp9a scores: easy bool 91.7 / choice 100; original bool 83.3 /
choice 94.4 / score 100 (12 items); hard bool 47.4 / choice 52.2 / score 16.7 (6 items).

### 3.3 In-distribution (validation splits of the training sources)

| Family | exp7a | exp8b | **exp9a** |
|---|---|---|---|
| Selection score (macro over all groups) | 54.9 | 73.2 | **81.4** |
| Same, on shortcut-free ("blind-hard") items only | 53.6 | 71.6 | **80.9** |
| Intent | 78.8 | 83.7 | **90.5** |
| MCQ | 57.6 | 67.0 | **89.9** |
| Bool | 70.3 | 84.3 | **88.1** |
| Score (ordinal, exact level) | 36.2 | 59.2 | **62.3** |
| Diversity | 54.3 | 72.8 | **77.4** |

The shortcut-free score (80.9) is almost identical to the full score (81.4). The gains come
from the tasks themselves, not from the dataset shortcuts identified in the audit.

### 3.4 Calibration and decision quality

| | exp7a | exp8b | **exp9a** |
|---|---|---|---|
| How often reordering the options changes the answer | 21.0% | 11.7% | **0.0%** |
| Ordinal answers within one level | 76.2% | 95.2% | **97.2%** |
| Calibration error (ECE): choice / bool / ordinal | 2.4 / 7.6 / 10.2 | 3.0 / 2.5 / 5.4 | 3.7 / 4.0 / 8.4 |
| Best answer-or-escalate utility, wrong answer costs 3× | −0.01 | 0.27 | **0.45** |
| Same, wrong answer costs 10× | −0.12 | 0.07 | **0.22** |

The utility rows measure how much value the model's confidence creates when it has to
decide between answering and escalating to a human. exp9a is the first model that is
clearly profitable even when mistakes are ten times as costly as correct answers are
valuable. Calibration error is slightly higher than exp8b's. Post-hoc temperature scaling
(already implemented in the evaluator) would normally close that gap.

### 3.5 The vector-injection test

Each held-out question was scored three ways:

- **text**: the option text in its branch, no injected vector
- **vector**: no option text at all, only the injected embedding
- **both**: the deployed setting

| Test set | Options | both | text | vector | chance |
|---|---|---|---|---|---|
| Banking77 | 4 | 96.7 | 96.7 | **28.0** | 25.0 |
| Banking77 | 16 | 86.7 | 86.7 | **7.0** | 6.2 |
| Banking77 | 71 | 70.0 | 70.3 | **1.7** | 1.4 |
| Zero-shot intents | 50 | 88.0 | 88.0 | **3.0** | 2.0 |

The injection gate, the learned scalar that controls how much of the vector reaches the
model, started at 0.10 and ended at **0.028**.

### 3.6 The rethinking-depth test

Accuracy after each of the six recurrent passes (in-distribution subset):

| Pass | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Accuracy | 82.4 | 82.4 | 82.3 | 82.4 | 82.3 | 82.4 |

---

## 4. Explanation

### 4.1 Why the larger backbone helped so much

Every earlier model in this project trained a 0.4B encoder, and in each case held-out
transfer stayed flat while in-distribution scores rose. The model learned the training
tasks but not the general skill of deciding over unfamiliar options.

Qwen3-Embedding-4B changes two things at once. It has ten times the capacity, and it was
pretrained specifically to judge whether two texts match in meaning, which is what a
System-1 decision is: does this context fit this option better than the others? That
knowledge arrives already trained. After only 46k examples (a fraction of one pass over
the data), held-out accuracy rose on every benchmark.

The typed-decisions yes/no result shows the difference in kind, not just degree. exp8b
always picked the option that paraphrased the question, and scored the same with the
options swapped because it was matching wording, not reading the case. exp9a scores 71.0%
with the options in either order and is right far more often, so it is now using the case
details to decide.

### 4.2 Why option order no longer matters

In exp7 and exp8 all options sat in one sequence, so each option's representation depended
on what came before it. Shuffling the options changed the answer 12–21% of the time. In
exp9 each option is scored in its own branch that cannot see the other options, and every
branch starts at the same position. The score of an option therefore cannot depend on
which other options are present or in what order. This was verified to be exact (maximum
relative difference 2.3×10⁻⁶ against scoring each option alone), and the measured
order-flip rate is 0.0%.

### 4.3 Why vector injection did not work

The idea was that at high option counts each option's text gets truncated, so a vector
summarising the full text, injected at the option's slot, would carry what the text could
not. Three observations explain the result:

1. **The truncation it was designed for never happened on these tasks.** With a
   2,048-token budget, even 71 options get about 20 tokens each, which is more than a
   typical label needs ("card swallowed", "exchange rate"). The text path was never
   starved of information, so the vector had nothing to add.
2. **The vector is a lossy copy of information the model already reads in full.** The
   branch contains the option text, processed by the full 4B backbone with attention to
   the whole context. The pooled embedding is a single-vector summary of that same text,
   computed without the context. Gradient flows to the richer path, and the model turned
   the weaker one down (gate 0.10 → 0.028), exactly as exp7a's gate settled at ~0.02.
3. **The model never learned to read the vector on its own.** Vector-only examples were
   10% of training, and at the end the vector-only mode is at chance even with 4 options.
   The backbone was never pretrained to interpret an embedding vector placed in its input,
   and the gradient pressure to learn that was small because the text path already solved
   the task.

This has now been measured at 0.4B (exp7a/8) and 4.9B (exp9a) with the same outcome. The
mechanism could matter for tasks whose option texts are genuinely too long to fit, but no
benchmark in this project has that property, and it costs an extra backbone pass per
batch for the option embeddings.

### 4.4 Why rethinking passes did not help

The recurrent block re-reads the backbone's output on every pass, but that output is
computed once and never changes. For the questions in these benchmarks (which option fits
this utterance, is this claim supported by this passage), a single read of good features
is already enough, so later passes have nothing new to extract. The training objective
also rewards every pass equally, so the cheapest solution is for passes 2–6 to repeat
pass 1. This matches every earlier depth measurement in the project (exp7b, 7c, 7e, 8),
now at 10× the scale, and costs ~0.5B parameters plus five extra passes of compute.

---

## 5. Caveats

- **Short training.** 1,450 steps (~46k examples). The run was cut from 4,000 steps by the
  credit budget. Training loss was still falling at the end (0.40 at step 510, 0.11 at
  step 1,450), but with only one evaluation there is no held-out learning curve.
- **Reduced validation sets.** To fit the budget, the final evaluation used 100 examples per
  source group, 300 intent and 200 typed-decisions per group. Individual group numbers
  carry roughly ±4–5 points of sampling noise. The aggregate numbers are more stable.
- **Small JevBench samples.** The public tiers have 48, 72 and 111 items, and some question
  types within them have only 6–12. Laya and Jev numbers come from the full tiers.
- **One run, one seed.** No repeat runs, so run-to-run variance is unknown.
- **Calibration measured before temperature scaling.** The fitted per-bucket temperatures
  are in the evaluation output but were not applied to the numbers above.

---

## 6. Run history

The final run succeeded on the fifth A100 session. Each failure was caught early and fixed
before continuing:

| Attempt | What went wrong | Fix |
|---|---|---|
| 1 | Colab's preinstalled `torchao` 0.10 made `peft` refuse to load | uninstall `torchao` during setup |
| 2 | Google Drive's connection dropped while copying the 7.5 GB model | download weights from Hugging Face to local disk; copy data locally with a file-count check; mirror results to Drive with retries |
| 3 | The preflight check read only the last 5,000 characters of output and missed a passing line | checks now run on the full output |
| 4 | Out of memory at step 424, and every following batch failed too, because variables in the training loop still held the failed batch's memory | clear all references and free memory after an out-of-memory error; smaller micro-batches (12k tokens); resume from the step-250 checkpoint |
| 5 | none | 1,450 steps, 0 out-of-memory errors, final evaluation, weights mirrored to Drive, VM released |

---

## 7. What to do next

1. **Remove the vector injection and the rethinking loop.** Neither contributes, and
   together they cost about 0.5B parameters, an extra backbone pass for option embeddings,
   and five extra head passes. A simple scoring head on the branch readouts should match
   exp9a at lower cost.
2. **Spend that compute on the backbone and on training length.** Try Qwen3-Embedding-8B
   and train for many more steps; exp9a saw 14% of one pass over the data, and its
   training loss was still falling. Evaluate every ~1,000 steps to get a held-out curve.
3. **Fix the remaining yes/no weakness.** Add claim-framed yes/no training (both options
   restate the claim, so wording similarity carries no signal) and structured JSON/table
   contexts, which match Jev's decision states and appear in no current training source.
4. **Report calibrated numbers.** Apply the fitted temperatures and re-measure ECE and
   decision utility.

---

## 8. Reproducing

| What | Where |
|---|---|
| Model, packing, head | `s9_model.py`, `s9_head.py` |
| Training | `s9_train.py` |
| Evaluation (incl. modality sweep and depth curve) | `s9_eval.py` |
| Exactness test for branch packing | `test_branch_mask.py` |
| Data rebuild from public sources | `prepare_data.py` |
| Colab workflow | `scripts/colab_exp9_train_gpu.sh`, `scripts/colab_exp9_watch.sh` |
| Trained weights (0.9B trainable parameters, bf16) | `MyDrive/openjev/runs/exp9a/best_bf16.pt`; the frozen base downloads from Hugging Face as `Qwen/Qwen3-Embedding-4B` |
| Full metrics | W&B `open-system-one/exp9a`; `MyDrive/openjev/runs/exp9a/eval_001450.json` |

Evaluate the saved model:

```bash
python experiments/exp9_scale/s9_eval.py --ckpt best_bf16.pt --out results.json
```
