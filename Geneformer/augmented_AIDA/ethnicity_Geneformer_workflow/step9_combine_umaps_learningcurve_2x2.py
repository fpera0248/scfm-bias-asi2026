#!/usr/bin/env python3
"""
Generate learning curve figures for the ethnicity axis.

Structure:
  - 1x2 figure with both classifiers side by side
    Left panel: Random Forest   Right panel: Logistic Regression
  - All four dataset variants as colored lines within each panel
  - Shared single-line legend saved separately

Output:
  figure2_learning_curves_1x2.png
  figure2_learning_curves_legend.png

Run from the ethnicity_scfoundation_workflow directory on Oscar:
    python combine_learning_curves.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────

BASE = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/"
    "ethnicity_scfoundation_workflow"
)
STEP5_DIR = BASE / "step5_outputs_ethnicity"
CSV_PATH  = STEP5_DIR / "step5_learning_curves_ethnicity.csv"
OUTDIR    = BASE / "step9_visualizations_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_1X2    = OUTDIR / "figure2_learning_curves_1x2.png"
OUTPUT_LEGEND = OUTDIR / "figure2_learning_curves_legend.png"

# ── Strategy config ───────────────────────────────────────────────────────

STRATEGIES = [
    {"dataset": "Proportional_2498",          "label": "Proportional", "color": "#4C72B0"},
    {"dataset": "BalancedAugmented_2091Each",  "label": "Augmented",    "color": "#DD8452"},
    {"dataset": "BalancedUpsampled_2091Each",  "label": "Upsampled",    "color": "#55A868"},
    {"dataset": "Downsampled_16Each",          "label": "Downsampled",  "color": "#C44E52"},
]

# ── Plot helper ───────────────────────────────────────────────────────────

def plot_all_strategies(ax, df, model_name, title):
    """All four strategies as colored lines. Solid = validation, dashed = training."""
    df_model = df[df["model"] == model_name]

    for s in STRATEGIES:
        sub = df_model[df_model["dataset"] == s["dataset"]].sort_values("n_train")
        if sub.empty:
            print(f"  WARNING: no data for {s['dataset']} / {model_name}")
            continue

        col = s["color"]

        ax.plot(sub["n_train"], sub["val_f1_mean"],
                "-o", color=col, linewidth=2.2, markersize=5,
                zorder=4, label=s["label"])
        ax.fill_between(sub["n_train"],
                        sub["val_f1_mean"] - sub["val_f1_std"],
                        sub["val_f1_mean"] + sub["val_f1_std"],
                        alpha=0.12, color=col, zorder=3)
        ax.plot(sub["n_train"], sub["train_f1_mean"],
                "--", color=col, alpha=0.45, linewidth=1.8, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Training set size", fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0.30, 1.05)
    ax.tick_params(labelsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Legend ────────────────────────────────────────────────────────────────

def save_legend():
    """Single-line legend: four strategy colors + validation/training line styles."""
    strategy_handles = [
        mlines.Line2D([], [], color=s["color"], linewidth=2.5,
                      marker="o", markersize=7, label=s["label"])
        for s in STRATEGIES
    ]
    style_handles = [
        mlines.Line2D([], [], color="gray", linewidth=2.5,
                      linestyle="-", marker="o", markersize=7,
                      label="Validation (solid)"),
        mlines.Line2D([], [], color="gray", linewidth=2.0,
                      linestyle="--", alpha=0.7,
                      label="Training (dashed)"),
    ]

    all_handles = strategy_handles + style_handles  # 6 total on one line

    fig = plt.figure(figsize=(13, 0.7))
    fig.legend(
        handles=all_handles,
        loc="center",
        ncol=len(all_handles),
        fontsize=13,
        frameon=False,
        handlelength=2.0,
        handleheight=1.0,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0,
    )

    fig.savefig(str(OUTPUT_LEGEND), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved legend: {OUTPUT_LEGEND}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
    print(f"Models found: {df['model'].unique().tolist()}")
    print(f"Datasets found: {df['dataset'].unique().tolist()}")

    # 1x2 figure — RF left, LR right
    fig, (ax_rf, ax_lr) = plt.subplots(1, 2, figsize=(14, 5))

    plot_all_strategies(ax_rf, df, "RandomForest",    "Random Forest")
    plot_all_strategies(ax_lr, df, "LogReg", "Logistic Regression")

    plt.tight_layout(w_pad=2.0)
    fig.savefig(str(OUTPUT_1X2), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_1X2}")

    save_legend()


if __name__ == "__main__":
    main()