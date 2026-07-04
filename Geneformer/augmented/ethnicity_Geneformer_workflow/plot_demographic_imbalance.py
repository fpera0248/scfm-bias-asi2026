#!/usr/bin/env python3
"""
Generate three individual demographic imbalance pie charts for the ILD dataset.
One PNG per axis: Sex, Age, Ethnicity.
Legends handled by shared figure0_demographic_imbalance_legend.png.
All cell counts are exact values from Step 0A logs.

Output:
  step9_visualizations_ethnicity/figure0_imbalance_sex.png
  step9_visualizations_ethnicity/figure0_imbalance_age.png
  step9_visualizations_ethnicity/figure0_imbalance_ethnicity.png
  step9_visualizations_ethnicity/figure0_demographic_imbalance_legend.png

Run from anywhere on Oscar:
    python plot_demographic_imbalance.py
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

BASE   = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/"
    "ethnicity_scfoundation_workflow"
)
OUTDIR = BASE / "step9_visualizations_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Exact cell counts from Step 0A logs ──────────────────────────────────

SEX_DATA = {
    "Male":   309_549,
    "Female": 129_917,
}

AGE_DATA = {
    "10-19":   9_100,
    "20-29":   5_772,
    "30-39":  21_146,
    "40-49":  27_139,
    "50-59": 113_290,
    "60-69": 210_412,
    "70-79":  37_380,
}

ETHNICITY_DATA = {
    "European\nAmerican":  333_602,
    "African\nAmerican":    39_666,
    "Hispanic\nor Latin":   12_787,
    "Asian":                10_054,
    "Native\nAmerican":      2_626,
}

SEX_COLORS = ["#4C72B0", "#DD8452"]
AGE_COLORS = ["#c7e9b4", "#7fcdbb", "#41b6c4",
              "#1d91c0", "#225ea8", "#253494", "#081d58"]
ETH_COLORS = ["#FF8F00", "#7B1FA2", "#388E3C", "#1976D2", "#D32F2F"]


# ── Helper ────────────────────────────────────────────────────────────────

def save_pie(data, colors, title, output_path):
    """Draw and save a single pie chart PNG. No legend, no ratio label."""
    labels  = list(data.keys())
    sizes   = list(data.values())
    max_idx = sizes.index(max(sizes))
    explode = [0.04 if i == max_idx else 0.0 for i in range(len(sizes))]

    def autopct_fn(pct):
        return f"{pct:.1f}%" if pct >= 3.0 else ""

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct=autopct_fn, pctdistance=0.72, startangle=90,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    # No legend — use shared figure0_demographic_imbalance_legend.png
    # No ratio label — mentioned in figure caption

    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_legend():
    """Shared three-row legend: one row per demographic axis."""
    all_entries = [
        ("Sex",
         ["Male", "Female"],
         SEX_COLORS),
        ("Age",
         ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
         AGE_COLORS),
        ("Ethnicity",
         ["European American", "African American", "Hispanic or Latin",
          "Asian", "Native American"],
         ETH_COLORS),
    ]

    fig_l, axes_l = plt.subplots(3, 1, figsize=(12, 2.4))
    for ax_l, (panel_title, labels, colors) in zip(axes_l, all_entries):
        ax_l.set_axis_off()
        handles = [
            mpatches.Patch(facecolor=colors[i], label=labels[i], linewidth=0)
            for i in range(len(labels))
        ]
        ax_l.legend(
            handles=handles,
            title=panel_title,
            title_fontsize=13,
            loc="center",
            ncol=len(labels),
            fontsize=13,
            frameon=False,
            handlelength=1.2,
            handleheight=1.0,
            handletextpad=0.5,
            columnspacing=1.0,
            borderpad=0,
            borderaxespad=0,
        )

    plt.subplots_adjust(hspace=0.0)
    out = OUTDIR / "figure0_demographic_imbalance_legend.png"
    fig_l.savefig(str(out), dpi=300, bbox_inches="tight",
                  facecolor="white", pad_inches=0.03)
    plt.close(fig_l)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    save_pie(SEX_DATA, SEX_COLORS,
             "Sex",
             OUTDIR / "figure0_imbalance_sex.png")

    save_pie(AGE_DATA, AGE_COLORS,
             "Age (decade bins)",
             OUTDIR / "figure0_imbalance_age.png")

    save_pie(ETHNICITY_DATA, ETH_COLORS,
             "Ethnicity",
             OUTDIR / "figure0_imbalance_ethnicity.png")

    save_legend()


if __name__ == "__main__":
    main()