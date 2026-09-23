"""Generates the figures for the exp7 write-up from the REAL per-epoch
training log and the REAL measured benchmark numbers -- no synthetic or
illustrative data anywhere. Every value below was copied from
`jepa_checkpoints/exp7/train_full.log` (13 completed epochs of the exp7a
run, stopped 2026-09-21) or from the local eval runs against
`exp7_best_zeroshot.pt` (epoch 10).

Usage: python make_figures.py [--outdir figures]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# REAL DATA -- transcribed from train_full.log's per-epoch summary lines.
# --------------------------------------------------------------------------

EPOCHS = list(range(1, 14))
LOSS_TOTAL = [3.4243, 2.8912, 2.1158, 1.8737, 1.7187, 1.6341, 1.5840,
              1.5201, 1.4853, 1.4578, 1.4183, 1.4038, 1.3886]
LOSS_LOG = [2.3522, 1.8961, 1.4458, 1.3279, 1.2376, 1.1683, 1.1356,
            1.0911, 1.0638, 1.0424, 1.0115, 1.0063, 0.9929]
LOSS_SPH = [0.6393, 0.5176, 0.4021, 0.3704, 0.3449, 0.3253, 0.3190,
            0.3086, 0.3008, 0.2942, 0.2867, 0.2835, 0.2821]
LOSS_RPS = [0.7524, 0.7363, 0.4690, 0.3606, 0.3086, 0.3031, 0.2889,
            0.2747, 0.2711, 0.2683, 0.2635, 0.2558, 0.2546]

VAL_ACC = [0.0200, 0.5100, 0.6033, 0.5967, 0.6267, 0.6700, 0.6733,
           0.7100, 0.6433, 0.6433, 0.6700, 0.6833, 0.6933]
ZERO_SHOT = [0.0300, 0.5900, 0.6267, 0.6167, 0.6500, 0.6833, 0.6533,
             0.6833, 0.6833, 0.7133, 0.5933, 0.6900, 0.6533]
BANKING77 = [0.0275, 0.3450, 0.4350, 0.4475, 0.4675, 0.4750, 0.5025,
             0.4975, 0.4550, 0.5200, 0.4625, 0.5125, 0.5025]

INJECT_GATE = [0.0096, 0.0103, 0.0129, 0.0137, 0.0146, 0.0155, 0.0167,
               0.0180, 0.0187, 0.0199, 0.0207, 0.0212, 0.0215]
LR = [5.00e-5, 4.97e-5, 4.86e-5, 4.70e-5, 4.47e-5, 4.19e-5, 3.87e-5,
      3.50e-5, 3.11e-5, 2.71e-5, 2.29e-5, 1.89e-5, 1.50e-5]

# Measured against exp7_best_zeroshot.pt (epoch 10), full test sets.
# Laya/Jev figures are THEIR published numbers, not re-run by us --
# see the write-up's limitations section on why that matters.
BENCH = {
    "AG News\n(4-way)":        {"exp7": 0.744,  "laya": 0.950, "jev": 0.910, "chance": 0.25},
    "DAIR Emotion\n(6-way)":   {"exp7": 0.590,  "laya": 0.595, "jev": 0.480, "chance": 0.167},
    "Banking77 holdout\n(71-way)": {"exp7": 0.5576, "laya": 0.425, "jev": 0.870, "chance": 0.013},
}

# The format-sensitivity result: identical frozen weights, identical option
# set, identical 12,215 holdout examples -- only the instruction text differs.
FORMAT_SENSITIVITY = {
    "Randomized generic\ninstruction bank": 0.4307,
    "Fixed routing question\n(task-appropriate)": 0.5576,
}

C_PRIMARY = "#2563eb"
C_SECOND = "#e11d48"
C_THIRD = "#059669"
C_MUTED = "#94a3b8"
C_ACCENT = "#d97706"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_training_loss(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax1.plot(EPOCHS, LOSS_TOTAL, "o-", color=C_PRIMARY, linewidth=2, markersize=4)
    _style(ax1, "(a) Total training loss", "Epoch", "Loss")
    ax1.set_ylim(0, 3.7)

    ax2.plot(EPOCHS, LOSS_LOG, "o-", color=C_PRIMARY, linewidth=1.6, markersize=3.5,
             label="log score (cross-entropy)")
    ax2.plot(EPOCHS, LOSS_RPS, "s-", color=C_SECOND, linewidth=1.6, markersize=3.5,
             label="RPS (ordinal, score-type only)")
    ax2.plot(EPOCHS, LOSS_SPH, "^-", color=C_THIRD, linewidth=1.6, markersize=3.5,
             label="spherical score")
    _style(ax2, "(b) Loss components", "Epoch", "Component value")
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.set_ylim(0, 2.6)

    fig.tight_layout()
    p = os.path.join(outdir, "fig1_training_loss.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_generalization(outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.plot(EPOCHS, VAL_ACC, "o-", color=C_MUTED, linewidth=1.8, markersize=4,
            label="val_acc (seen intents, in-distribution)")
    ax.plot(EPOCHS, ZERO_SHOT, "o-", color=C_PRIMARY, linewidth=2.2, markersize=4.5,
            label="zero_shot_acc (45 never-trained intents)")
    ax.plot(EPOCHS, BANKING77, "o-", color=C_SECOND, linewidth=2.2, markersize=4.5,
            label="banking77_holdout (71 labels, fully held out)")

    ax.axhline(0.02, color="#cbd5e1", linestyle="--", linewidth=1)
    ax.text(4.2, 0.028, "chance (N=50)", fontsize=7, color="#64748b", va="bottom")

    ax.axvspan(0.5, 1.5, color="#fef3c7", alpha=0.55, zorder=0)
    ax.text(1.75, 0.70, "LR warmup\nspans all of epoch 1", fontsize=7.5, ha="left",
            color="#92400e", linespacing=1.3)

    _style(ax, "Generalization across training (exp7a, 13 epochs)", "Epoch", "Accuracy")
    ax.set_ylim(0, 0.80)
    ax.set_xlim(0.5, 13.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.06))

    fig.tight_layout()
    p = os.path.join(outdir, "fig2_generalization.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gate_and_lr(outdir):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    ax.plot(EPOCHS, INJECT_GATE, "o-", color=C_ACCENT, linewidth=2.2, markersize=4.5)
    ax.axhline(0.01, color="#cbd5e1", linestyle="--", linewidth=1)
    ax.text(13.2, 0.0102, "initialization (0.01)", fontsize=7.5, color="#64748b",
            va="bottom", ha="right")
    _style(ax, "Learned vector-injection gate $g$ over training",
           "Epoch", "$g$ (scalar gate value)")
    ax.set_xlim(0.5, 13.5)
    ax.set_ylim(0.008, 0.023)

    ax2 = ax.twinx()
    ax2.plot(EPOCHS, np.array(LR) * 1e5, color=C_MUTED, linewidth=1.2, linestyle=":")
    ax2.set_ylabel("LR ($\\times 10^{-5}$)", fontsize=8, color="#64748b")
    ax2.tick_params(labelsize=7.5, colors="#64748b")
    ax2.spines[["top"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(outdir, "fig3_inject_gate.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_benchmarks(outdir):
    names = list(BENCH.keys())
    x = np.arange(len(names))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w, [BENCH[n]["exp7"] for n in names], w, label="exp7 (ours, epoch 10)",
           color=C_PRIMARY)
    ax.bar(x, [BENCH[n]["laya"] for n in names], w, label="Laya (published)", color=C_SECOND)
    ax.bar(x + w, [BENCH[n]["jev"] for n in names], w, label="Jev (published)", color=C_MUTED)

    for i, n in enumerate(names):
        for dx, key in [(-w, "exp7"), (0, "laya"), (w, "jev")]:
            v = BENCH[n][key]
            ax.text(i + dx, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.plot([i - 1.5 * w, i + 1.5 * w], [BENCH[n]["chance"]] * 2,
                color="#0f172a", linestyle="--", linewidth=1)

    ax.text(2.42, 0.03, "chance", fontsize=7, color="#0f172a")
    _style(ax, "Zero-shot benchmark comparison (dashed = chance level)", "", "Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    p = os.path.join(outdir, "fig4_benchmarks.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_format_sensitivity(outdir):
    labels = list(FORMAT_SENSITIVITY.keys())
    vals = list(FORMAT_SENSITIVITY.values())

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(labels, vals, width=0.5, color=[C_MUTED, C_PRIMARY])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(1, 0.5576), xytext=(0, 0.4307),
                arrowprops=dict(arrowstyle="->", color=C_SECOND, linewidth=1.8))
    ax.text(0.5, 0.515, "+12.7 points\n(identical weights,\nidentical options)",
            ha="center", fontsize=8.5, color=C_SECOND, fontweight="bold", linespacing=1.4)

    ax.axhline(0.425, color="#0f172a", linestyle="--", linewidth=1.2)
    ax.text(1.38, 0.432, "Laya (0.425)", fontsize=8, color="#0f172a", ha="right")

    _style(ax, "Instruction-phrasing sensitivity (Banking77, n=12,215)", "", "Accuracy")
    ax.set_ylim(0, 0.68)
    ax.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout()
    p = os.path.join(outdir, "fig5_format_sensitivity.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# Cardinality sweep x ablation mode -- measured 2026-09-22 against
# exp7_best_zeroshot.pt (epoch 10), 150 examples per cell. The N=255 cell is
# capped to N=184 by the seen-label pool size after Banking77's removal.
SWEEP_N = [2, 4, 8, 20, 77, 184]
SWEEP = {
    "text_only":   [0.9867, 0.9733, 0.9600, 0.9267, 0.7933, 0.6133],
    "vector_only": [0.5733, 0.2533, 0.1533, 0.0467, 0.0067, 0.0000],
    "both":        [0.9867, 0.9800, 0.9600, 0.9333, 0.7600, 0.5667],
}


def fig_sweep(outdir):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(SWEEP_N))

    ax.plot(x, SWEEP["text_only"], "o-", color=C_SECOND, linewidth=2.4, markersize=6,
            label="text only  (no injected vector, ~Laya)")
    ax.plot(x, SWEEP["both"], "s--", color=C_PRIMARY, linewidth=2.4, markersize=6,
            label="both  (exp7 as designed)")
    ax.plot(x, SWEEP["vector_only"], "^-", color=C_ACCENT, linewidth=2.0, markersize=6,
            label="vector only  (no option text)")

    chance = [1.0 / n for n in SWEEP_N]
    ax.plot(x, chance, ":", color="#0f172a", linewidth=1.4, label="chance (1/N)")

    ax.annotate("vector path sits at chance\nat every cardinality",
                xy=(4, 0.0067), xytext=(2.35, 0.20), fontsize=8.5, color=C_ACCENT,
                fontweight="bold", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, linewidth=1.4))
    ax.annotate("text-only beats both\nby 3.3 / 4.7 pts",
                xy=(5, 0.585), xytext=(3.1, 0.40), fontsize=8.5, color=C_SECOND,
                fontweight="bold", linespacing=1.35, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_SECOND, linewidth=1.4))

    _style(ax, "Cardinality sweep x ablation mode (150 examples/cell)",
           "Option count N", "Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in SWEEP_N])
    ax.set_ylim(-0.03, 1.06)
    ax.legend(fontsize=8, frameon=False, loc="center left")

    fig.tight_layout()
    p = os.path.join(outdir, "fig6_cardinality_sweep.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for fn in (fig_training_loss, fig_generalization, fig_gate_and_lr,
               fig_benchmarks, fig_format_sensitivity, fig_sweep):
        print("wrote", fn(args.outdir))


if __name__ == "__main__":
    main()
