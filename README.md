# Open System-1 (OpenJev)

> An open-source System-1 decision model: calibrated choices over any set of options, without generating text.

Open System-1 is an Apache-2.0 research project building an open alternative to TypeSafe AI's
**Jev**. A System-1 model does not write answers. It receives a situation (the *context*), a
question, and a set of options, and returns a **calibrated probability for each option**,
fast enough to sit inside software as a decision layer.

It supports the three decision types Jev defines:

| Type | Example | Output |
|---|---|---|
| **choice** | "Which team should handle this ticket?" over N teams | probability per option |
| **bool** | "Does this claim hold, given the policy?" | yes / no probability |
| **score** | "How urgent is this, on a 1–5 scale?" | probability per ordinal level |

---

## Current best model: exp9a (4.9B parameters)

Trained on 2026-09-27/28 on a single Colab A100 for 2.4 hours. Full report:
[`experiments/exp9_scale/RESULTS.md`](experiments/exp9_scale/RESULTS.md).

### Held-out benchmarks (never trained on)

| Benchmark | **Open System-1 exp9a** | Laya (Convai) | Jev (TypeSafe) |
|---|---|---|---|
| Banking77, 71-way routing | **73.3** | 42.5 | 87.0 |
| AG News, 4-way | **84.2** | 95.0 | 91.0 |
| DAIR Emotion, 6-way | **63.0** | 59.5 | 48.0 |
| JevBench easy | **97.9** | 94.4 | 100 |
| JevBench original / standard | **91.7** | 72.9 | 99 |
| JevBench hard | **48.6** | 34.1 | 74.1 |

exp9a beats Laya on Banking77, DAIR Emotion and all three JevBench tiers, and beats Jev on DAIR
Emotion. It is behind Jev on Banking77, AG News and JevBench hard, and behind Laya on AG News.

JevBench numbers for exp9a are on the public half of each tier (48 / 72 / 111 items) under the
official protocol (criteria as options, full state as context). Laya and Jev numbers are from the
JevBench v1.2 leaderboard on the full tiers, so that comparison is indicative.

### Progress across experiments

| | exp7a (0.4B) | exp8b (0.4B) | **exp9a (4.9B)** |
|---|---|---|---|
| In-distribution selection score | 54.9 | 73.2 | **81.4** |
| Same, on shortcut-free items only | 53.6 | 71.6 | **80.9** |
| Zero-shot intents (50 options) | 70.6 | 75.7 | **90.0** |
| Banking77 (71-way, held out) | 55.3 | 55.0 | **73.3** |
| AG News | 74.5 | 73.2 | **84.2** |
| typed-decisions: choice / yes-no | 23.2 / 59.6 | 27.8 / 57.2 | **61.0 / 71.0** |
| Answer changes when options are reordered | 21.0% | 11.7% | **0.0%** |
| Answer-or-escalate utility (wrong answer costs 3×) | −0.01 | 0.27 | **0.45** |

---

## Architecture (exp9)

```
 instructions + context ──► shared PREFIX ─────────────┐
                                                       │  each branch sees the prefix
 option 1 ──► [MARK] "Option: …" [EOS]   branch 1 ─────┤  and itself only; every branch
 option 2 ──► [MARK] "Option: …" [EOS]   branch 2 ─────┤  starts at the same position
 …                                                     │
                        Qwen3-Embedding-4B + LoRA (one pass over prefix + all branches)
                                                       │
            option readouts (each branch's [EOS] state) + context summary
                                                       │
                   decision head ──► one logit per option ──► softmax
```

| Component | Detail |
|---|---|
| Backbone | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), frozen, adapted with LoRA (rank 64, all linear layers) |
| Branch packing | A custom attention mask lets every option attend to the shared prefix and to itself, never to other options, and all branches share position ids. An option's score therefore cannot depend on which other options are present or their order. Verified exact against scoring each option alone (relative difference 2.3×10⁻⁶). |
| Readout | each branch's final end-of-text token, the position the embedding model was pretrained to summarise at |
| Decision head | context resampler (128 latents), 6-pass recurrent block with scratchpad, per-option scorer (0.77B parameters) |
| Training objective | RLCD-style: maximise a proper scoring reward (log score + 0.5 × spherical score − ranked probability score for ordinal questions). A proper scoring rule is maximised only by honest probabilities, which is what makes the outputs calibrated. |

Total: 4.92B parameters, of which 0.90B are trained.

---

## What we have learned

These are measured results, not design assumptions. Each is backed by a report in `experiments/`.

**What works**

1. **A strong embedding model as the backbone.** Moving from a 0.4B encoder to Qwen3-Embedding-4B
   is what finally moved held-out accuracy (+5 to +33 points), after it had stayed flat
   through all of experiment 8.
2. **Isolated option branches.** They remove sensitivity to option order completely (0.0% of
   answers change when options are shuffled, down from 12–21%).
3. **Auditing the training data for shortcuts.** Every corpus is checked with "blind" baselines,
   models that see only part of the input. An earlier experiment (exp7e) scored exactly what a
   single word cue in its data predicted and had learned nothing else. The audited data removed
   that failure mode, and the shortcut-free score (80.9) now matches the full score (81.4).
4. **Proper-scoring-rule training.** Ordinal answers are 97% within one level, and the model's
   confidence is useful for deciding when to escalate to a human, even when mistakes are 10× as
   costly as correct answers are valuable.

**What does not work (tested at 0.4B and at 4.9B)**

1. **Injecting a precomputed option embedding into the option's slot.** This was the project's
   original core idea: a fixed-cost vector per option to beat the token-budget limit at high option
   counts. With only the vector available the model scores at chance for every option count, and
   with vector and text together it scores exactly as well as text alone. On these benchmarks,
   option labels are short enough that the text path is never starved, so the vector adds nothing.
   The model learns to switch it off.
2. **Recurrent "rethinking" passes in the decision head.** Accuracy after 1 pass equals accuracy
   after 6 (82.4%). For single-read decisions the backbone's features are already sufficient.
3. **Training on multi-step deduction data (ProofWriter) to make depth useful.** The model learned a
   surface cue instead. That is System-2 work, and System-1 models should escalate it rather than
   attempt it.

**Next**: drop the vector path and the recurrent passes, put the compute into a larger backbone
(Qwen3-Embedding-8B) and much longer training, and add claim-framed yes/no data and structured
(JSON) contexts, which match Jev's decision format.

---

## Training data

About 331k training examples from public datasets, rebuilt and audited in experiment 8:

| Family | Type | Rows | Sources |
|---|---|---|---|
| Intent | choice, 2–128 options | 36k | CLINC150, HWU64, SNIPS, MASSIVE (Banking77 held out) |
| Multiple choice | choice | 131k | RACE, SciQ, CommonsenseQA, OpenBookQA, ARC, HellaSwag |
| Yes/no | bool | 57k | BoolQ, MNLI, PAWS, QQP |
| Ordinal | score | 34k | Yelp, Amazon reviews, SST-5, STS-B |
| Label diversity | choice | 73k | DBpedia, Yahoo Answers, 20 Newsgroups, TREC, GoEmotions, tweet tasks (27 label sets) |

Labels are balanced within every source, texts are deduplicated across splits, and any text that
appears in a held-out benchmark is removed. Shortcut-prone sources are filtered (AFLite) or
reweighted, and every validation item is tagged so accuracy can also be reported on the items
that no shortcut can solve. The corpora are not stored in this repository (about 1.3 GB);
`experiments/exp9_scale/prepare_data.py` rebuilds all of them from their public sources. The
build manifest, filter report and audit results are in `data/`.

---

## Model weights

The exp9a weights are **not in this repository**: the trained part is 1.8 GB (bf16), above
GitHub's file limits. They are stored as a single checkpoint holding the 0.9B trained parameters
(LoRA adapters, decision head, projector). The frozen backbone is downloaded separately from
Hugging Face (`Qwen/Qwen3-Embedding-4B`). A Hugging Face Hub release is planned.

---

## Reproducing

```bash
pip install -r experiments/exp7_hybrid_decision/requirements.txt peft wandb nvidia-ml-py

# rebuild the audited training data (public sources; the filtering step is CPU-heavy)
python experiments/exp9_scale/prepare_data.py

# check that branch packing is exact on your backbone
python experiments/exp9_scale/test_branch_mask.py Qwen/Qwen3-Embedding-0.6B

# local smoke run on the 0.6B backbone (same code path as the 4B run)
python experiments/exp9_scale/s9_train.py --backbone Qwen/Qwen3-Embedding-0.6B --smoke --max_steps 30

# full run on Colab (WSL + colab-cli; see the script header for the steps)
export WANDB_API_KEY=...        # never hard-code it
GPU=A100 HOURS=3 bash scripts/colab_exp9_train_gpu.sh

# evaluate a checkpoint (all benchmarks, modality sweep, depth curve)
python experiments/exp9_scale/s9_eval.py --ckpt best_bf16.pt --out results.json
```

---

## Repository layout

```
experiments/
  exp9_scale/          current model: 4B backbone, branch packing, scaled head   RESULTS.md, NOTES.md
  exp8_s1_rlcd/        0.4B System-1 model, RLCD objective, evaluation suite     NOTES.md
  exp7_hybrid_decision/  original hybrid vector-injection architecture           NOTES.md
  exp1..exp6/          earlier experiments                                       see PROJECT_HISTORY.md
  jevbench_protocol.py official JevBench item rendering
  LAYA_COMPARISON_REPORT.md, PROJECT_HISTORY.md
scripts/               data build / filter / audit, Colab launch and watch scripts
data/                  build manifest, filter and audit reports (corpora are rebuilt, not stored)
snake_laya/            OpenJev-compatible demo server and playground
```

---

## Limitations

- exp9a trained on about 46k examples (14% of one pass over the data) in a credit-limited run.
- Final evaluation used reduced validation sets, so individual groups carry about ±4–5 points of
  sampling noise. JevBench public tiers are small (48 / 72 / 111 items).
- One training run and one seed. Run-to-run variance is not yet measured.
- Multi-step reasoning (the JevBench hard tier) remains well behind Jev.

---

## License

Apache-2.0.
