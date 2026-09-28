# Experiment 8 — System-1 decisions, RLCD-style objective, from exp7a

Written at launch (2026-09-24). Results are appended at the bottom.

## The question

Starting from exp7a (the best System-1 checkpoint: zero-shot intents 71.3%,
AG News 75.8%, DAIR Emotion 58.6%, Banking77 56.1% with the routing prompt),
does training on the rebuilt, shortcut-audited, three-question-type corpus with
a calibration-first objective improve **accuracy on the blind-hard subsets**,
**zero-shot transfer**, and **calibration** at the same time?

## Why this and not exp7d/e

exp7d/e chased a rising depth curve on long-context ProofWriter. The model
learned "question contains *not* → False" (it reproduced that heuristic's
accuracy to the decimal) and nothing else. Two separate conclusions:
- Multi-hop deduction over 8K tokens is System-2 work. The product needs
  System-1 decisions plus calibrated escalation.
- Every corpus needs a shortcut audit before training. That audit is now
  `scripts/audit_exp7_data.py`.

## Model — exp7a, exactly

`s1_model.py` rebuilds exp7a's head with matching parameter names: entry
layer, a 2-layer encoder block run **once**, the scorer, and 16 context codes.
The checkpoint loads with `strict=True`. `verify_exp7a.py` replays exp7a's own
epoch-10 eval: val 0.6433 and zero-shot 0.7133 reproduce exactly (head input
= `s0`). Banking77 gives 0.5275 vs 0.5200 (bf16 batching noise).

No recursion, scratchpad, FiLM or LoRA: none has shown a measurable effect
on System-1 tasks, and each is a variable we couldn't explain.

## Objective — RLCD as far as it is public

TypeSafe publishes only RLCD's goal: calibrated decision probabilities. There
is no reward function, estimator, data or code. The public implementation of
the idea (Laya) rewards the reported distribution q with a strictly proper
scoring rule:

    R(q, y) = log q_y + 0.5·spherical(q, y) − RPS(q, y)·[ordinal]

`--estimator exact` (this run) maximizes E[R] using its exact gradient: the
action is the report, and R is differentiable in it. `--estimator grpo` is a
genuinely sampled version: Dirichlet-perturbed reports with a GRPO
group-normalized advantage and the same reward, for an A/B comparison. RPS is
active for the first time: exp7a had no ordinal data, and its logs show
`rps 0.0000` throughout.

## Data (≈331k training rows, see data/exp7_manifest.json)

| Family | Sampling share | Pattern controls |
|---|---|---|
| intent (184 labels, Banking77 held out) | 20% | class-balanced; distractor filter |
| mcq (7 sources, AFLite-filtered) | 25% | HellaSwag ×0.5 |
| bool (BoolQ, MNLI, PAWS, QQP) | 15% | bias_w sampling; QQP ×0.5; yes/no order shuffled |
| score (Yelp, Amazon, SST-5, STS-B) | 15% | descriptive levels, 30% reversed, never reordered |
| diversity (27 label vocabularies) | 25% | class-balanced; native-subset sampling |

Source weights within a family ∝ √size. Modality dropout is 10% vector-only
and 10% text-only.

## Evaluation (every 500 steps; step 0 = exp7a baseline)

42 groups and ~15k examples. Every group reports acc, **acc on the
blind-hard subset**, NLL, Brier, ECE, mean confidence, chance, and RPS /
within-1 for ordinal groups. Results are also aggregated by family, question
type and cardinality bucket, plus decision utility (answer vs escalate at
τ* = (c_wrong − c_esc)/(1 + c_wrong)), order-flip rate, latency, and fitted
per-bucket temperatures.

Checkpoints are selected by the macro accuracy over the in-distribution
groups only. External benchmarks (AG News, DAIR Emotion), Banking77,
typed-decisions and Hard-111 never influence selection.

## Run

`python train.py --run_name exp8a --estimator exact`: 6000 steps × 32
examples, 8-bit AdamW, top backbone LR 1e-5 with layer decay 0.9, head ×5,
3% warmup, cosine to 10%. Measured locally at ~3.8 s/step, 9.5 GB peak on the
RTX 5070 Ti laptop GPU; estimated ~7.5–8 h including evals. W&B project
`open-system-one`.

## Results

### exp8a3 — 6000 steps, exact estimator (2026-09-24/25)

Run history: exp8a slowed 2x at step 60 when the CUDA cache filled the 12 GB
card and the Windows driver spilled into system RAM. Fixed with a PyTorch
memory cap (0.88). exp8a2 then hit an OOM at step ~330 on a rare large batch.
Fixed with smaller micro-batches (8k tokens / 256 options), skipping a step on
OOM instead of crashing, and saving every 250 steps. exp8a3 hit Modern Standby
twice (idle timeout; the GPU was capped at ~33 W) and one unclean reboot after
waking. It resumed from latest.pt, and training now holds the system- and
display-required execution state.

| step | select | blind-hard | held-out¹ | held-out ECE | order flip | utility (3× cost) |
|---|---|---|---|---|---|---|
| 0 (exp7a) | 54.95 | 53.57 | 53.61 | 10.5 | 21.0 | −0.011 |
| 1000 | 64.98 | 63.74 | 53.00 | 16.0 | 17.3 | 0.098 |
| 2000 | 67.15 | 65.93 | **54.99** | 15.2 | 18.0 | 0.140 |
| 3000 | 70.24 | 68.82 | 54.16 | 15.1 | 13.7 | 0.184 |
| **5500 (best.pt)** | **71.64** | **70.55** | 53.46 | 17.6 | 15.3 | 0.230 |
| 6000 | 71.40 | 70.13 | 54.10 | 17.5 | 14.7 | 0.238 |

¹ macro over zero-shot intents, Banking77 (routing), AG News, DAIR Emotion, and
typed-decisions choice/bool/score. None of these are used for selection.

What worked:
- **In-distribution +16.7 points, and the blind-hard subsets gained the same
  amount.** The gains are the task, not the audited shortcuts.
- **PAWS 53.3 → 84.0.** exp7a had learned QQP's "shared words ⇒ paraphrase"
  shortcut; that is now unlearned.
- **Ordinal:** exact accuracy 36 → 60, RPS 0.63 → 0.32, within-one-level 76 → 94%,
  ECE 10.2 → 3.3%. This was the first ordinal training signal the model got.
- **Calibration:** bool ECE 7.6 → 2.5%. The fitted temperature for 2-way choice
  went from 5.0 to 1.08.
- **Order-flip rate 21 → 15%.** Decision utility at 3× error cost went from
  −0.01 to +0.23; at 10× cost from −0.12 to +0.03.

What didn't:
- **Held-out transfer is flat (53.6 → 53.5; peak 55.0 at step 2000) and more
  overconfident** (held-out ECE 10.5 → 17.6). The only held-out gain is
  zero-shot intents (70.6 → ~75). AG News, Emotion and Banking77 did not move.
- **typed-decisions bool = 57.2% from step 2000 on**, which equals the rate at
  which option 2 is correct. At first this was read as "always picks the second
  slot". exp8b's swapped-options check disproved that (see below): the model
  always picks the criterion that *restates the question*.
- typed-decisions choice and score swing ±5 points between evals, more than
  n=500 explains.

### exp8b — descriptive-criteria bool (launched 2026-09-25 02:45)

From exp8a3 best.pt (step 5500). 70% of bool examples are rendered as
descriptive criteria (4 sources × 3–4 phrasings, order shuffled). The audit
shows each criterion text is correct 48–52% of the time, "pick the option
without negation" scores 50.5%, and an options-only classifier scores 50.4%.
Mix: bool .40 / diversity .20 / intent .15 / mcq .15 / score .10. LR 5e-6,
2000 steps. New val groups: bool/<src>@desc and typed/bool@swapped (same items,
options reversed: a position-driven model scores 1 − acc).

Result (2026-09-25, 2000 steps, best.pt = step 2000, select **73.18**, blind-hard 71.64):

| | exp7a | exp8a3 final | exp8b |
|---|---|---|---|
| select / blind-hard | 54.95 / 53.57 | 71.72 / 70.36 | **73.18 / 71.64** |
| bool family | 70.3 | 78.0 | **84.3** |
| bool@desc (boolq/mnli/paws/qqp) | — | 76.3/76.0/69.3/67.7 | **80.7/86.3/88.7/80.7** |
| order flip | 21.0 | 15.3 | **11.7** |
| utility (3× cost) | −0.011 | 0.197 | **0.274** |
| zero-shot intents / Banking77 routing | 70.6 / 55.3 | 73.6 / 55.2 | 75.7 / 55.0 |
| AG News / DAIR Emotion | 74.5 / 58.9 | 73.4 / 59.0 | 73.2 / 57.7 |
| typed choice / bool / score | 23.2 / 59.6 / 33.2 | 27.8 / 57.4 / 27.8 | 27.8 / 57.2 / 31.8 |

- Descriptive-criteria yes/no is fixed **in-distribution**: +4 to +19 points,
  now matching the yes/no-word rendering, with ECE 3.8–6.9%.
- **typed/bool is not fixed, and the swapped check shows why.** typed/bool and
  typed/bool@swapped both score 57.2%. A position-follower would score 42.8%
  on the swapped set. Run directly on all 1600 typed bool items, exp8b picks the
  second-authored criterion 100% of the time, in every workflow, at 66–96%
  confidence. In each case that criterion **paraphrases the question**: "This
  trace requires human review." → "A human should inspect this run." (48% acc,
  96% conf); "…malicious or unauthorised activity" → "The underlying behaviour
  is malicious or unauthorised." (69% acc, 95% conf). Jev's `noul` questions
  state a claim, and the options are claim-false / claim-true. The model
  matches claim to paraphrase, the same "most similar option" skill that
  intent/topic data rewards, and never reads the state. Contexts are ~106
  tokens, so truncation is not the cause.
- Fix direction: train on claim-framed bool, with the question as a proposition
  and BOTH options restating it ("<claim> -- this holds." vs "<claim> -- this is
  not the case."), so that similarity to the question carries zero
  information. Also add structured-state contexts (JSON/tabular), which no
  training source currently has.

