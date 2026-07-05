#!/usr/bin/env python3
"""
Analyze whether upsampling and downsampling break data distributions
relative to the proportional baseline.

Strategy: load each pilot file to get real cell obs_names, then look up
those cells in the step 0a raw counts file (AIDA_RawCounts_ETHNICITY_900k.h5ad)
which is guaranteed to have raw integer counts in adata.X.

Compares four sampling strategies across:
  1. Library size distribution (total counts per cell)
  2. Gene expression sparsity (fraction of zero-count genes per cell)
  3. Cell type composition (proportion of each cell type)
  4. KS test: each strategy vs proportional baseline

Real cells only — synthetic cells filtered out from augmented.

Output CSVs and PNGs saved to:
  step9_visualizations_ethnicity/distribution_analysis/

Run on Oscar:
    python analyze_distribution_shift.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────

BASE = pathlib.Path(
    "/data/scfoundation/augmentedv4/"
    "ethnicity_scfoundation_workflow"
)
OUTDIR = BASE / "step9_visualizations_ethnicity" / "distribution_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Step 0a output — raw integer counts guaranteed in adata.X
RAW_FILE = BASE / "AIDA_RawCounts_ETHNICITY_900k.h5ad"

# Pilot files — used only to get which cell obs_names belong to each strategy
DATASETS = {
    "Proportional": BASE / "AIDA_Ethnicity_Pilot_Proportional_2498_ETHNICITY.h5ad",
    "Augmented":    BASE / "AIDA_Ethnicity_Pilot_BalancedAugmented_2091Each_ETHNICITY.h5ad",
    "Upsampled":    BASE / "AIDA_Ethnicity_Pilot_BalancedUpsampled_2091Each_ETHNICITY.h5ad",
    "Downsampled":  BASE / "AIDA_Ethnicity_Pilot_Downsampled_16Each_ETHNICITY.h5ad",
}

PALETTE = {
    "Proportional": "#4C72B0",
    "Augmented":    "#DD8452",
    "Upsampled":    "#55A868",
    "Downsampled":  "#C44E52",
}

# ── Helper ────────────────────────────────────────────────────────────────

def get_real_obs_names(path):
    """Return obs_names of real cells only (excludes synthetic)."""
    adata = sc.read_h5ad(path)
    if "source" in adata.obs.columns:
        real = adata[adata.obs["source"] == "real"]
        print(f"  {path.name}: {adata.n_obs} total, {len(real)} real")
        return real.obs_names.tolist(), real.obs
    print(f"  {path.name}: {adata.n_obs} cells (no source column — all real)")
    return adata.obs_names.tolist(), adata.obs


def load_raw_counts_for(obs_names, raw_adata):
    """
    Index the step-0a raw counts AnnData by obs_names.
    Handles duplicated obs_names (upsampled) by using positional iloc.
    """
    # Find which raw indices correspond to these obs_names
    raw_index = pd.Index(raw_adata.obs_names)
    # For upsampled, obs_names contains duplicates — get unique set
    unique_names = list(dict.fromkeys(obs_names))  # preserves order, deduplicates
    valid = [n for n in unique_names if n in raw_index]
    if len(valid) < len(unique_names):
        print(f"    WARNING: {len(unique_names) - len(valid)} obs_names not found in raw file")
    sub = raw_adata[valid].copy()
    X = sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.array(X)
    print(f"    Raw counts shape: {X.shape}  "
          f"mean lib size: {X.sum(axis=1).mean():.0f}")
    return X, sub.obs


# ── Analysis functions ────────────────────────────────────────────────────

def compute_per_cell_stats(X):
    lib_sizes = X.sum(axis=1).flatten()
    sparsity  = (X == 0).mean(axis=1).flatten()
    return lib_sizes, sparsity


def ks_test_vs_baseline(baseline_vals, other_vals, label):
    stat, pval = stats.ks_2samp(baseline_vals, other_vals)
    return {
        "comparison":  label,
        "ks_stat":     round(stat, 4),
        "pval":        pval,
        "significant": pval < 0.05,
    }


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_distributions(data_dict, metric_name, xlabel, log_scale=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in data_dict.items():
        v = np.log1p(vals) if log_scale else vals
        ax.hist(v, bins=60, alpha=0.45, color=PALETTE[name],
                label=name, density=True, edgecolor="none")
    ax.set_xlabel(
        ("log1p(" if log_scale else "") + xlabel + (")" if log_scale else ""),
        fontsize=12,
    )
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of {metric_name}\n(unique real cells only)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTDIR / f"dist_{metric_name.lower().replace(' ', '_')}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_celltype_proportions(prop_dict):
    all_cts = sorted(set(
        ct for p in prop_dict.values() if p is not None for ct in p.index
    ))
    if not all_cts:
        return

    x     = np.arange(len(all_cts))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(12, len(all_cts) * 0.8), 5))
    for i, (name, props) in enumerate(prop_dict.items()):
        if props is None:
            continue
        vals = [props.get(ct, 0.0) for ct in all_cts]
        ax.bar(x + i * width, vals, width, label=name,
               color=PALETTE[name], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(all_cts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of cells", fontsize=12)
    ax.set_title("Cell type composition (unique real cells only)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTDIR / "dist_celltype_proportions.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Load the step-0a raw counts reference once
    if not RAW_FILE.exists():
        print(f"ERROR: raw counts file not found: {RAW_FILE}")
        return
    print(f"Loading raw counts reference: {RAW_FILE.name}")
    raw_adata = sc.read_h5ad(RAW_FILE)
    print(f"  Raw file shape: {raw_adata.shape}")
    print(f"  X range: {raw_adata.X.min():.1f} – {raw_adata.X.max():.1f}")

    print("\nGetting real cell IDs from pilot files...")
    obs_names_dict = {}
    obs_dict       = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping")
            continue
        obs_names_dict[name], obs_dict[name] = get_real_obs_names(path)

    if "Proportional" not in obs_names_dict:
        print("ERROR: Proportional baseline not found.")
        return

    # Look up raw counts for each strategy's unique real cells
    print("\nLooking up raw counts in reference file...")
    X_dict    = {}
    prop_dict = {}
    for name, obs_names in obs_names_dict.items():
        print(f"  {name}:")
        X, obs_sub = load_raw_counts_for(obs_names, raw_adata)
        X_dict[name] = X
        # Cell type proportions from the obs metadata
        for col in ("cell_type", "celltype", "CellType"):
            if col in obs_sub.columns:
                prop_dict[name] = obs_sub[col].value_counts(normalize=True)
                break
        else:
            prop_dict[name] = None

    # ── Library size & sparsity ───────────────────────────────────────────
    print("\nComputing per-cell statistics from raw counts...")
    lib_sizes  = {}
    sparsities = {}
    for name, X in X_dict.items():
        ls, sp = compute_per_cell_stats(X)
        lib_sizes[name]  = ls
        sparsities[name] = sp
        print(f"  {name}: lib size mean={ls.mean():.0f} median={np.median(ls):.0f} | "
              f"sparsity mean={sp.mean():.3f} median={np.median(sp):.3f}")

    plot_distributions(lib_sizes,  "Library Size",
                       "Total counts per cell", log_scale=True)
    plot_distributions(sparsities, "Gene Sparsity",
                       "Fraction of zero-count genes per cell")

    # ── KS tests ─────────────────────────────────────────────────────────
    print("\nRunning KS tests vs Proportional baseline...")
    ks_results = []
    bl_ls = lib_sizes["Proportional"]
    bl_sp = sparsities["Proportional"]

    for name in obs_names_dict:
        if name == "Proportional":
            continue
        ks_results.append(ks_test_vs_baseline(
            bl_ls, lib_sizes[name],
            f"{name} vs Proportional — Library Size"))
        ks_results.append(ks_test_vs_baseline(
            bl_sp, sparsities[name],
            f"{name} vs Proportional — Gene Sparsity"))

    ks_df = pd.DataFrame(ks_results)
    out_csv = OUTDIR / "ks_test_results.csv"
    ks_df.to_csv(out_csv, index=False)
    print("\nKS test results:")
    print(ks_df.to_string(index=False))
    print(f"Saved: {out_csv.name}")

    # ── Cell type composition ─────────────────────────────────────────────
    print("\nCell type proportion shifts vs Proportional:")
    plot_celltype_proportions(prop_dict)
    baseline_props = prop_dict.get("Proportional")
    if baseline_props is not None:
        for name, props in prop_dict.items():
            if name == "Proportional" or props is None:
                continue
            all_cts = baseline_props.index.union(props.index)
            delta = (props.reindex(all_cts).fillna(0) -
                     baseline_props.reindex(all_cts).fillna(0))
            print(f"  {name}: max={delta.abs().max():.4f}  "
                  f"mean={delta.abs().mean():.4f}")

    # ── Summary table ─────────────────────────────────────────────────────
    summary_rows = []
    for name, X in X_dict.items():
        ls, sp = compute_per_cell_stats(X)
        summary_rows.append({
            "Strategy":            name,
            "N unique real cells": X.shape[0],
            "Mean library size":   f"{ls.mean():.0f}",
            "Median library size": f"{np.median(ls):.0f}",
            "Mean sparsity":       f"{sp.mean():.3f}",
            "Median sparsity":     f"{np.median(sp):.3f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUTDIR / "distribution_summary.csv"
    summary_df.to_csv(out_summary, index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"Saved: {out_summary.name}")
    print(f"\nAll outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()