#!/usr/bin/env python3
"""
Generate a clean 1x4 UMAP figure for the ethnicity axis.
Reads scFoundation-embedded h5ad files, computes UMAP, and plots
all four panels in a single row with a separate legend file.

Run from the ethnicity_scfoundation_workflow directory on Oscar:
    python combine_umaps_2x2.py

CHANGES:
  [2026-03-24] Subsamples to UMAP_MAX_PER_GROUP cells per group before UMAP
               so all four panels are visually comparable.
  [2026-03-28] Style overhaul: no axes, arrow compass, strategy names only,
               separate legend file, tight layout.
  [2026-03-28] Changed from 2x2 to 1x4 single-row layout for better fit
               in two-column paper format.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────

BASE = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/"
    "ethnicity_scfoundation_workflow"
)
OUTDIR = BASE / "step9_visualizations_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "h5ad": BASE / "AIDA_Ethnicity_Pilot_Proportional_2498_ETHNICITY_scfoundation.h5ad",
        "label": "Proportional",
    },
    {
        "h5ad": BASE / "AIDA_Ethnicity_Pilot_BalancedAugmented_2091Each_ETHNICITY_scfoundation.h5ad",
        "label": "Augmented",
    },
    {
        "h5ad": BASE / "AIDA_Ethnicity_Pilot_BalancedUpsampled_2091Each_ETHNICITY_scfoundation.h5ad",
        "label": "Upsampled",
    },
    {
        "h5ad": BASE / "AIDA_Ethnicity_Pilot_Downsampled_16Each_ETHNICITY_scfoundation.h5ad",
        "label": "Downsampled",
    },
]

OUTPUT        = OUTDIR / "figure1_ethnicity_umaps_2x2.png"
OUTPUT_LEGEND = OUTDIR / "figure1_ethnicity_umaps_legend.png"

UMAP_MAX_PER_GROUP = 500

# ── Ethnicity styling ─────────────────────────────────────────────────────

CANONICAL_MAP = {
    "asian":             "Asian",
    "european american": "European American",
    "hispanic or latin": "Hispanic or Latin",
    "native american":   "Native American",
    "african american":  "African American",
}

ETH_COLORS = {
    "African American":  "#7B1FA2",
    "Asian":             "#1976D2",
    "European American": "#FF8F00",
    "Hispanic or Latin": "#388E3C",
    "Native American":   "#D32F2F",
}

LEGEND_ORDER = [
    "African American",
    "Asian",
    "European American",
    "Hispanic or Latin",
    "Native American",
]

LEGEND_LABELS = {
    "African American":  "African American",
    "Asian":             "Asian",
    "European American": "European American",
    "Hispanic or Latin": "Hispanic or Latin",
    "Native American":   "Native American",
}

# ── Helpers ───────────────────────────────────────────────────────────────

def load_and_umap(h5ad_path):
    """Load, canonicalize, subsample, compute UMAP."""
    adata = sc.read_h5ad(h5ad_path)

    eth_col = "self_reported_ethnicity"
    raw = adata.obs[eth_col].astype(str).str.strip().str.lower()
    mapped = raw.map(CANONICAL_MAP).fillna("European American")
    adata.obs[eth_col] = mapped

    rng = np.random.default_rng(42)
    keep_idx = []
    for grp in adata.obs[eth_col].unique():
        idx = np.where(adata.obs[eth_col].values == grp)[0]
        n = min(UMAP_MAX_PER_GROUP, len(idx))
        keep_idx.extend(rng.choice(idx, size=n, replace=False).tolist())
    adata = adata[sorted(keep_idx)].copy()
    print(f"  Subsampled to {adata.n_obs} cells (max {UMAP_MAX_PER_GROUP}/group)")

    emb = np.array(adata.obsm["X_scfoundation"], dtype=np.float32).copy()
    emb[np.isnan(emb)] = 0.0
    adata.obsm["X_scfoundation"] = emb

    sc.pp.neighbors(adata, use_rep="X_scfoundation", n_neighbors=15, random_state=42)
    sc.tl.umap(adata, min_dist=0.3, random_state=42)

    return adata


def add_axis_arrows(ax):
    """Draw Nature-style UMAP axis arrows in the bottom-left corner."""
    x0, y0 = 0.04, 0.06
    arm     = 0.16

    arrowprops = dict(
        arrowstyle="-|>",
        color="black",
        lw=1.3,
        mutation_scale=8,
    )

    ax.annotate(
        "", xy=(x0 + arm, y0), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=arrowprops,
    )
    ax.text(
        x0 + arm + 0.02, y0, "UMAP 1",
        transform=ax.transAxes,
        fontsize=7, va="center", ha="left", color="black",
    )

    ax.annotate(
        "", xy=(x0, y0 + arm), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=arrowprops,
    )
    ax.text(
        x0, y0 + arm + 0.02, "UMAP 2",
        transform=ax.transAxes,
        fontsize=7, va="bottom", ha="center", color="black",
    )


def plot_panel(ax, adata, label, show_axis_arrows=False):
    """Plot one UMAP panel with no axis decorations."""
    coords  = adata.obsm["X_umap"]
    eth     = adata.obs["self_reported_ethnicity"].values
    pt_size = 18

    for group in LEGEND_ORDER:
        mask = np.array([e == group for e in eth])
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=pt_size,
            c=[ETH_COLORS[group]],
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )

    # Tight axis limits: clip to 1st-99th percentile
    pad = 0.05
    x, y = coords[:, 0], coords[:, 1]
    x_lo, x_hi = np.percentile(x, 1), np.percentile(x, 99)
    y_lo, y_hi = np.percentile(y, 1), np.percentile(y, 99)
    x_rng = x_hi - x_lo
    y_rng = y_hi - y_lo
    ax.set_xlim(x_lo - pad * x_rng, x_hi + pad * x_rng)
    ax.set_ylim(y_lo - pad * y_rng, y_hi + pad * y_rng)

    ax.set_title(label, fontsize=13, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_axis_arrows:
        add_axis_arrows(ax)


def save_legend():
    """Standalone single-row legend — all five groups on one line."""
    handles = [
        mpatches.Patch(facecolor=ETH_COLORS[g], label=LEGEND_LABELS[g], linewidth=0)
        for g in LEGEND_ORDER
    ]

    fig = plt.figure(figsize=(12, 0.9))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(LEGEND_ORDER),
        fontsize=18,
        frameon=False,
        handlelength=1.4,
        handleheight=1.2,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0,
    )

    fig.savefig(
        str(OUTPUT_LEGEND),
        dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05,
    )
    plt.close(fig)
    print(f"Saved legend: {OUTPUT_LEGEND}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # 1x4 single-row layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        print(f"Loading {ds['h5ad'].name}...")
        adata = load_and_umap(ds["h5ad"])
        # Axis arrows on leftmost panel only (index 0)
        plot_panel(ax, adata, ds["label"], show_axis_arrows=(i == 0))
        del adata

    plt.tight_layout(h_pad=1.0, w_pad=0.5)
    fig.savefig(str(OUTPUT), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved UMAP figure: {OUTPUT}")

    save_legend()


if __name__ == "__main__":
    main()