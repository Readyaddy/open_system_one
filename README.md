# Open System-1 (OpenJev) 🚀
> **An Open-Source, High-Cardinality System-1 Decision Model & Drop-In Replacement for TypeSafe Jev**

Open System-1 (`open_system_one`) is an Apache-2.0 open-source, non-autoregressive **System-1 decision architecture** designed as a fast, calibrated, open replacement for TypeSafe AI's **Jev**.

---

## 🎯 Mission & Background

### What is a "System-1" Model?
Borrowed from psychologist Daniel Kahneman’s framework in *Thinking, Fast and Slow*:
* **System 2** (traditional LLMs like GPT/Claude): Slow, deliberate, autoregressive token generation for complex reasoning.
* **System 1** (Jev & Open System-1): Fast, automatic, parallel decision-making—immediate classification, routing, scoring, and branching without generating prose.

TypeSafe AI released **Jev** as a proprietary System-1 model trained via **Reinforcement Learning for Calibrated Decisions (RLCD)**, supporting three core decision primitives:
* **Choice**: Selecting 1 option from a candidate list of $N$ choices.
* **Score**: Providing an ordinal numerical rating (e.g., severity 1..N).
* **Noul (Boolean)**: True/False determination.

---

## 🚀 Why Open System-1 Exists

While existing open-source attempts (such as Convai's *Laya*) match Jev's API shape, they suffer from a mechanical architectural wall: **the shared sequence token bottleneck**. In Laya, all candidate option texts are packed into a single sequence with a shared token budget (192–256 tokens). For high-cardinality routing tasks like **Banking77** (71–77 options), options get truncated to ~3 tokens each, causing accuracy to collapse to **42.5%**.

**Open System-1** solves this wall through a novel **Hybrid Dual-Path Vector Injection Architecture**:
1. **Low-to-Moderate Cardinality ($N=2..10$)**: Full cross-attention between instructions, option text, and context.
2. **High Cardinality ($N \ge 11$)**: Encodes candidate options independently into 1024-dimensional semantic vectors and **injects them directly into input embedding slots** at Layer 0. This costs **1 token per option** regardless of text length, breaking the truncation wall.

---

## 🏗️ Architecture Blueprint

```
                      ┌──────────────────────────────────────┐
                      │    USER REQUEST (Context & Options)  │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [PATH A: Vector Encoding]                        [PATH B: Text Packing]
Option texts encoded independently              Build packed sequence:
-> Mean-pooled to 1024-d vector (v_i)           [CLS] qtype inst [SEP] [MASK_i] opt_text ... context
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                             VECTOR INJECTION (Layer 0)
                      Add v_i to [MASK_i] token embedding
                                         │
                                         ▼ (s_0 Initial Embeddings)
                        ┌─────────────────────────────────┐
                        │ ModernBERT 24-Layer Transformer │◀──┐
                        └────────────────┬────────────────┘   │ (s_0 Residual
                                         │                    │  Re-injection)
                    Recurrent Loop (k = 1..6) ────────────────┘
                                         │
                      ┌──────────────────┴──────────────────┐
                      ▼                                     ▼
           [DECISION SCORER HEAD]                [HALTING & ESCALATION MLPs]
      Gather [MASK_i] layer-24 vectors        Compute 6-dim feature vector
      -> MLP -> Softmax -> Probabilities      -> halt_head MLP -> Halt at k=1?
                                              -> escalate_head MLP -> Escalate?
```

### Key Technical Specs

* **Backbone**: `answerdotai/ModernBERT-large` (24 layers, $d=1024$, 395M parameters, RoPE positional encoding, unpadded attention, 8192 context window).
* **Vector Injection**: Projects untruncated 1024-d option embeddings directly into `[MASK_i]` token positions at Layer 0.
* **Recurrent Latent Depth ($k=1 \dots 6$)**: Transformer Decision Head loops over option states with **Gated $s_0$ Residual Anchors** (`LayerNorm(s + s0_gate * s0)`) and LayerNorm skip-connections to prevent representation drift across iterations.
* **Adaptive Halting (`halt_head`)**: A lightweight 2-layer MLP inspecting confidence metrics (top prob, top-2 margin, entropy, cardinality) at each depth step. Early exits at $k=1$ for clear inputs, achieving **83.3% compute savings (6x speedup)**.
* **Split-Conformal Escalation (`escalate_head`)**: Calibrated threshold ($\tau = 0.3308$) providing a **distribution-free mathematical guarantee that $\ge 90\%$ of incorrect halted predictions get flagged** for System-2 / human escalation.

---

## 📊 Benchmark Results vs. Baselines (Empirically Measured)

Empirically evaluated on standard public benchmarks (12,215 test items) against closed-source **Jev** (TypeSafe) and open-source **Laya** (Convai):

| Benchmark Dataset | Instruction Phrasing | **Open System-1 (Ours)** | **Laya** *(Convai)* | **Jev** *(TypeSafe)* | Random Chance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AG News** *(4-way zero-shot)* | *"Which category best describes this news article?"* | **75.80%** | **95.0%** | 91.0% | 25.0% |
| **DAIR Emotion** *(6-way zero-shot)* | *"What emotion does this text express?"* | **58.60%** ⭐ | **59.5%** | 48.0% | 16.7% |
| **Banking77** *(71-way Generic Prompt)* | *Randomized generic prompt bank* | **43.66%** | 42.5% | **87.0%** | ~1.4% |
| **Banking77** *(71-way Fixed Routing Prompt)* 🔥 | *"Which category should this be routed to?"* | **56.08%** ⭐ | 42.5% | **87.0%** | ~1.4% |

### 💡 Measured Insights
1. **The Instruction Alignment Effect (+12.4% to +13.1% Measured Boost)**:
   * Tested directly on GPU: Changing from a generic prompt bank (*"Select the best option:"*) to a domain-appropriate routing question (*"Which category should this be routed to?"*) increases zero-shot accuracy on Banking77 from **43.66% $\rightarrow$ 56.08%** (+12.42% absolute boost on v2, and +13.12% on v1).
   * **vs. Laya**: Widens our advantage over Laya (`42.5%`) to **+13.58 percentage points** on 71-way zero-shot routing.
2. **DAIR Emotion (6-way)**: Open System-1 (**58.60%**) **beats Jev (48.0%) by +10.6%** and matches Laya (**59.5%**).

---

## ⚡ Parameter & Efficiency Profile

| Parameter | Value |
| :--- | :--- |
| **Total Model Parameters** | **448.48 Million** |
| **Backbone Parameters** | 394.78 Million |
| **Decision Head Parameters** | 53.70 Million |
| **Halting Network (`HaltingHeads`)** | **794 parameters** (~0.0008M) |
| **Compute Savings with Halting** | **83.3% Saved** (6x speedup at $k=1$) |
| **Escalation Coverage Guarantee** | **$\ge 90\%$ Error Coverage** ($\alpha = 0.10, \tau = 0.3308$) |

---

## 📁 Repository Structure

```
.
├── experiments/
│   └── exp7_hybrid_decision/
│       ├── model.py            # HybridDecisionModel, Vector Injection, Recurrent Block
│       ├── halting.py          # HaltingHeads, 6-dim Feature Extractors, Conformal Thresholds
│       ├── train.py            # TaskMixer, Depth Loss, Training Loop
│       ├── train_halting.py    # Self-Supervised Halting & Escalation Training
│       ├── eval.py             # Benchmark Evaluation Engine (with Fixed Instruction Benchmarks)
│       └── calibrate.py        # Post-hoc Cardinality Bucket Temperature Fitting
├── snake_laya/                 # OpenJev / Jev API Compatible Web Server & Playground
├── scripts/                    # Training, Evaluation & Cloud Launch Scripts
├── ANALYSIS.md                 # Mechanistic Cross-Iteration Analysis
└── README.md
```

---

## 🛠️ Quickstart

### Installation
```bash
git clone https://github.com/Readyaddy/open_system_one.git
cd open_system_one
pip install -r experiments/exp7_hybrid_decision/requirements.txt
```

### Evaluating Checkpoints
```bash
python experiments/exp7_hybrid_decision/eval.py --ckpt checkpoints/exp7c_nomaxsim_v2_best_zeroshot.pt --mode external
```

### Running Halting Training
```bash
python experiments/exp7_hybrid_decision/train_halting.py --resume checkpoints/exp7c_nomaxsim_v2_best_zeroshot.pt --k_max 6 --epochs 6
```

---

## 📜 License
Apache-2.0 License. Free for commercial and research use.
