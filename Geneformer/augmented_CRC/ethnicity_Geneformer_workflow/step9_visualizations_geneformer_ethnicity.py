#!/usr/bin/env python3
"""
STEP 9 — Results Visualization (CRC ETHNICITY, Geneformer V2-316M)

Changes vs scFoundation version:
  [GF 1] BASE/OUTDIR   : scfoundation augmentedv4 -> Geneformer augmented
  [GF 2] EMB_KEY       : X_scfoundation -> X_geneformer
  [GF 3] Dataset names : 2091Each/2498/16Each -> 2143Each/2497/48Each
  [GF 4] Step dirs     : updated to geneformer output dirs
  [GF 5] UMAP tag      : "scFoundation Embedding" -> "Geneformer Embedding"
  [GF 6] scIB columns  : human-readable column names from new step3a
  [GF 7] CSV filenames : updated to _geneformer suffix
  [GF 8] Side-by-side UMAP panel added: raw PCA vs Geneformer embedding
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

# ============================================================
# PATHS  [GF 1]
# ============================================================

BASE = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow"
)

OUTDIR     = BASE / "step9_visualizations_ethnicity_geneformer"
STEP3A_DIR = BASE / "benchmark_outputs_geneformer_ethnicity"
STEP4A_DIR = BASE / "step4a_downstream_ethnicity"
STEP4B_DIR = BASE / "step4b_model_robustness_tests_ethnicity_geneformer"
STEP5_DIR  = BASE / "step5_outputs_ethnicity_geneformer"
STEP6_DIR  = BASE / "step6_outputs_ethnicity_geneformer"
STEP7_DIR  = BASE / "step7_representation_diagnostics_ethnicity_geneformer"
STEP8_DIR  = BASE / "step8_eth_conditioned_disease_geneformer"

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FONT SIZES
# ============================================================

plt.rcParams.update({
    "figure.dpi":            150,
    "font.size":             14,
    "axes.titlesize":        16,
    "axes.titleweight":      "bold",
    "axes.labelsize":        14,
    "legend.fontsize":       12,
    "legend.title_fontsize": 13,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
})

# ============================================================
# STYLE & CONSTANTS  [GF 3]
# ============================================================

PALETTE = {
    "Proportional_2497":          "#4C72B0",
    "BalancedAugmented_1880Each": "#DD8452",
    "BalancedUpsampled_1880Each": "#55A868",
    "Downsampled_48Each":         "#C44E52",
}

SHORT_LABELS = {
    "Proportional_2497":          "Proportional\n(1,998 cells)",
    "BalancedAugmented_1880Each": "scDesign3\nAugmented\n(1,504/group)",
    "BalancedUpsampled_1880Each": "Upsampled\n(1,504/group)",
    "Downsampled_48Each":         "Downsampled\n(90/group)",
}

STRATEGY_PALETTE = {
    "Baseline":   "#4C72B0",
    "AR":         "#937860",
    "EOS":        "#55A868",
    "AR+EOS":     "#C44E52",
    "EOS_adv":    "#55A868",
    "AR+EOS_adv": "#C44E52",
}

STRATEGY_DESCRIPTIONS = {
    "Baseline":   "Baseline\n(no reweighting)",
    "AR":         "AR\n(adaptive resampling)",
    "EOS":        "EOS\n(expansive oversampling)",
    "AR+EOS":     "AR + EOS\n(combined)",
    "EOS_adv":    "EOS\n(expansive oversampling)",
    "AR+EOS_adv": "AR + EOS\n(combined)",
}

ETH_COLORS = {
    "asian":             "#1976D2",
    "european american": "#FF8F00",
    "hispanic or latin": "#388E3C",
    "african american":   "#D32F2F",
    "african american":  "#7B1FA2",
    "Asian":             "#1976D2",
    "European American": "#FF8F00",
    "Hispanic or Latin": "#388E3C",
    "African American":   "#D32F2F",
    "African American":  "#7B1FA2",
}

DATASET_ORDER = [
    "Proportional_2497",
    "BalancedAugmented_1880Each",
    "BalancedUpsampled_1880Each",
    "Downsampled_48Each",
]

DATASET_KEYS = [
    "BalancedAugmented_1880Each",
    "Proportional_2497",
    "BalancedUpsampled_1880Each",
    "Downsampled_48Each",
]

# [GF 6] Human-readable column names from updated step3a
SCIB_METRIC_MAP = {
    "NMI":   ("KMeans NMI",       "NMI",   "Cell-Type Cluster NMI",   "Higher = better cell-type label preservation"),
    "ARI":   ("KMeans ARI",       "ARI",   "Cell-Type Cluster ARI",   "Higher = better cluster agreement with annotated labels"),
    "cLISI": ("cLISI",            "cLISI", "cLISI (Cell-Type LISI)",  "Higher = stronger cell-type structure in embedding"),
    "iLISI": ("iLISI",            "iLISI", "iLISI (Integration LISI)","Higher = better ethnicity mixing across groups"),
}

UNDERREP_GROUP = "african american"

# ============================================================
# UMAP CONFIG  [GF 3, GF 5]
# ============================================================

UMAP_DATASETS = {
    "Proportional_2497":          "CRC_Eth_Pilot_Proportional_2497_ETHNICITY",
    "BalancedAugmented_1880Each": "CRC_Eth_Pilot_BalancedAugmented_1880Each_ETHNICITY",
    "BalancedUpsampled_1880Each": "CRC_Eth_Pilot_BalancedUpsampled_1880Each_ETHNICITY",
    "Downsampled_48Each":         "CRC_Eth_Pilot_Downsampled_48Each_ETHNICITY",
}

UMAP_DISPLAY = {
    "Proportional_2497":          "Proportional (1,998 cells — real only)",
    "BalancedAugmented_1880Each": "Balanced Augmented (1,504/group — scDesign3)",
    "BalancedUpsampled_1880Each": "Balanced Upsampled (1,504/group — real only)",
    "Downsampled_48Each":         "Downsampled (90/group — real only)",
}

CANONICAL_ETH_MAP = {
    "asian":             "Asian",
    "european american": "European American",
    "hispanic or latin": "Hispanic or Latin",
    "african american":   "African American",
    "african american":  "African American",
}

ETH_PALETTE_UMAP  = {v: ETH_COLORS[v] for v in CANONICAL_ETH_MAP.values()}
SOURCE_PALETTE_UMAP = {"real": "#2C7BB6", "synthetic": "#D7191C"}
EMB_KEY = "X_geneformer"   # [GF 2]
_CT_PALETTE_CACHE: dict = {}

# ============================================================
# UTILITIES
# ============================================================

def save(fig, name):
    path = OUTDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")

def ds_color(ds):
    return PALETTE.get(ds, "#888888")

def ds_label(ds):
    return SHORT_LABELS.get(ds, ds)

def annotate_bars(ax, bars, values, fmt="{:.3f}", offset_frac=0.02, fontsize=12):
    if not values:
        return
    max_v = max((abs(v) for v in values if pd.notna(v)), default=0.01)
    for bar, val in zip(bars, values):
        if pd.notna(val):
            y = bar.get_height()
            sign_offset = max_v * offset_frac if val >= 0 else -max_v * offset_frac * 4
            ax.text(bar.get_x() + bar.get_width() / 2, y + sign_offset,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=fontsize, color="black", fontweight="bold")

def _safe(s):
    return (s.replace(" ", "_").replace("\n", "_")
             .replace("(", "").replace(")", "")
             .replace(",", "").replace("-", "_")
             .replace("/", "_").lower())

def _single_bar_figure(title, subtitle, xlabel, ylabel, datasets, values,
                       colors, fname, annotate=True, ylim_pad=1.35,
                       hline=None, hline_label=None):
    fig, ax = plt.subplots(figsize=(max(9, len(datasets) * 2.4), 7))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)
    ax.set_title(subtitle, fontsize=13, pad=10, style="italic", color="#444444")
    bars = ax.bar(range(len(datasets)), values, color=colors,
                  edgecolor="white", width=0.6, zorder=3)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds_label(d) for d in datasets], fontsize=13, ha="center")
    ax.tick_params(axis="x", pad=10)
    max_val = max((v for v in values if pd.notna(v)), default=0.05)
    ax.set_ylim(0, max(max_val * ylim_pad, 0.1))
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    if hline is not None:
        ax.axhline(hline, color="green", linestyle="--", linewidth=1.8,
                   alpha=0.7, label=hline_label)
        ax.legend(fontsize=12)
    if annotate:
        annotate_bars(ax, bars, values, fmt="{:.3f}", fontsize=12)
    plt.tight_layout(pad=2.0)
    save(fig, fname)

# ============================================================
# UMAP HELPERS  [GF 5]
# ============================================================

def _umap_params(n_cells):
    return (15, 0.3) if n_cells <= 2500 else (30, 0.5)

def _canonicalize_eth_obs(adata):
    ETH_COL = "self_reported_ethnicity"
    if ETH_COL not in adata.obs.columns:
        for alt in ("ethnicity", "Ethnicity", "ETHNICITY"):
            if alt in adata.obs.columns:
                adata.obs[ETH_COL] = adata.obs[alt]
                break
    if ETH_COL in adata.obs.columns:
        raw    = adata.obs[ETH_COL].astype(str).str.strip().str.lower()
        mapped = raw.map(CANONICAL_ETH_MAP).fillna("European American")
        adata.obs[ETH_COL] = mapped
    return adata

def _run_umap_raw(adata, n_cells=None):
    if n_cells is None:
        n_cells = adata.n_obs
    n_neighbors, min_dist = _umap_params(n_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=42)
    sc.tl.umap(adata, min_dist=min_dist, random_state=42)

def _run_umap_embedding(adata, emb_key=EMB_KEY, n_cells=None):
    if emb_key not in adata.obsm:
        raise RuntimeError(f"Missing {emb_key} in obsm")
    if n_cells is None:
        n_cells = adata.n_obs
    n_neighbors, min_dist = _umap_params(n_cells)
    emb = np.array(adata.obsm[emb_key], dtype=np.float32).copy()
    emb[np.isnan(emb)] = 0.0
    zero_mask = (np.abs(emb).sum(axis=1) == 0)
    if zero_mask.any():
        print(f"   Filtering {zero_mask.sum()} zero-vector cells before UMAP")
        adata = adata[~zero_mask].copy()
        emb   = emb[~zero_mask]
    adata.obsm[emb_key] = emb
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=n_neighbors, random_state=42)
    sc.tl.umap(adata, min_dist=min_dist, random_state=42)
    return adata

def _get_ct_palette(all_celltypes):
    key = tuple(sorted(all_celltypes))
    if key not in _CT_PALETTE_CACHE:
        types  = sorted(all_celltypes)
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(types), 1)))
        _CT_PALETTE_CACHE[key] = {ct: colors[i] for i, ct in enumerate(types)}
    return _CT_PALETTE_CACHE[key]

def _umap_scatter(ax, coords, labels, palette, pt_size, title,
                  subtitle="", legend_outside=False):
    unique = sorted(set(str(l) for l in labels))
    for grp in unique:
        mask  = np.array([str(l) == grp for l in labels])
        color = palette.get(grp, palette.get(grp.lower(), "#AAAAAA"))
        ax.scatter(coords[mask, 0], coords[mask, 1], s=pt_size, c=[color],
                   alpha=0.65, linewidths=0, label=grp, rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_outside:
        ax.legend(fontsize=10, markerscale=2.5, framealpha=0.85,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    else:
        ax.legend(fontsize=11, markerscale=2.5, framealpha=0.8, loc="lower right")

# ============================================================
# [GF 8] SIDE-BY-SIDE UMAP: raw PCA vs Geneformer embedding
# ============================================================

def generate_sidebyside_umap(ds_key, ds_stem, ct_palette):
    """One figure, two panels: raw PCA UMAP (left) vs Geneformer UMAP (right)."""
    raw_path = BASE / f"{ds_stem}.h5ad"
    gf_path  = BASE / f"{ds_stem}_geneformer.h5ad"

    if not raw_path.exists() or not gf_path.exists():
        print(f"  Side-by-side UMAP skipped for {ds_key}: file(s) not found")
        return

    display = UMAP_DISPLAY[ds_key]
    print(f"  [Side-by-side] {ds_key}")

    adata_raw = sc.read_h5ad(raw_path)
    _canonicalize_eth_obs(adata_raw)
    _run_umap_raw(adata_raw, n_cells=adata_raw.n_obs)
    coords_raw = adata_raw.obsm["X_umap"]
    eth_raw    = adata_raw.obs["self_reported_ethnicity"].astype(str).values
    n_raw      = adata_raw.n_obs

    adata_gf = sc.read_h5ad(gf_path)
    _canonicalize_eth_obs(adata_gf)
    try:
        adata_gf = _run_umap_embedding(adata_gf, n_cells=adata_gf.n_obs)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return
    coords_gf = adata_gf.obsm["X_umap"]
    eth_gf    = adata_gf.obs["self_reported_ethnicity"].astype(str).values

    pt_size = max(4, min(25, 50_000 // max(n_raw, adata_gf.n_obs)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Embedding Comparison — {display}", fontsize=16, fontweight="bold", y=1.02)

    _umap_scatter(axes[0], coords_raw, eth_raw, ETH_PALETTE_UMAP, pt_size,
                  "Raw Expression (PCA)", legend_outside=False)
    _umap_scatter(axes[1], coords_gf,  eth_gf,  ETH_PALETTE_UMAP, pt_size,
                  "Geneformer V2-316M Embedding", legend_outside=True)

    plt.tight_layout()
    fname = f"umap_sidebyside_{_safe(ds_key)}_ethnicity.png"
    path = OUTDIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {fname}")

    del adata_raw, adata_gf

# ============================================================
# INDIVIDUAL UMAPs
# ============================================================

def generate_individual_umaps():
    print("\n[UMAPs] Generating individual UMAP PNGs...")

    all_ct = set()
    for ds_key, ds_stem in UMAP_DATASETS.items():
        for suffix in ("", "_geneformer"):
            fpath = BASE / f"{ds_stem}{suffix}.h5ad"
            if fpath.exists():
                try:
                    tmp = sc.read_h5ad(fpath)
                    for col in ("cell_type", "celltype", "CellType"):
                        if col in tmp.obs.columns:
                            all_ct.update(tmp.obs[col].unique())
                            break
                except Exception:
                    pass

    ct_palette = _get_ct_palette(all_ct) if all_ct else {}
    print(f"  Found {len(all_ct)} unique cell types")

    for ds_key, ds_stem in UMAP_DATASETS.items():
        display = UMAP_DISPLAY[ds_key]

        # Raw expression UMAP
        raw_path = BASE / f"{ds_stem}.h5ad"
        if raw_path.exists():
            print(f"\n  [Raw] {ds_key}")
            adata_raw = sc.read_h5ad(raw_path)
            _canonicalize_eth_obs(adata_raw)
            _run_umap_raw(adata_raw, n_cells=adata_raw.n_obs)
            coords = adata_raw.obsm["X_umap"]
            pt_size = max(4, min(25, 50_000 // adata_raw.n_obs))
            ETH_COL = "self_reported_ethnicity"

            if ETH_COL in adata_raw.obs.columns:
                fig, ax = plt.subplots(figsize=(9, 7))
                _umap_scatter(ax, coords, adata_raw.obs[ETH_COL].astype(str).values,
                              ETH_PALETTE_UMAP, pt_size,
                              f"{display}\nColored by Ethnicity (Raw Expression)")
                plt.tight_layout()
                save(fig, f"umap_raw_{_safe(display)}_ethnicity.png")

            for col in ("cell_type", "celltype"):
                if col in adata_raw.obs.columns:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    _umap_scatter(ax, coords, adata_raw.obs[col].astype(str).values,
                                  ct_palette, pt_size,
                                  f"{display}\nColored by Cell Type (Raw Expression)",
                                  legend_outside=True)
                    fig.subplots_adjust(left=0.07, right=0.60, top=0.88, bottom=0.10)
                    save(fig, f"umap_raw_{_safe(display)}_celltype.png")
                    break
            del adata_raw

        # Geneformer embedding UMAP  [GF 5]
        gf_path = BASE / f"{ds_stem}_geneformer.h5ad"
        if gf_path.exists():
            print(f"\n  [Geneformer] {ds_key}")
            adata_gf = sc.read_h5ad(gf_path)
            _canonicalize_eth_obs(adata_gf)
            try:
                adata_gf = _run_umap_embedding(adata_gf, n_cells=adata_gf.n_obs)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                continue
            coords  = adata_gf.obsm["X_umap"]
            pt_size = max(4, min(25, 50_000 // adata_gf.n_obs))
            ETH_COL = "self_reported_ethnicity"

            if ETH_COL in adata_gf.obs.columns:
                fig, ax = plt.subplots(figsize=(9, 7))
                _umap_scatter(ax, coords, adata_gf.obs[ETH_COL].astype(str).values,
                              ETH_PALETTE_UMAP, pt_size,
                              f"{display}\nColored by Ethnicity (Geneformer Embedding)")
                plt.tight_layout()
                save(fig, f"umap_geneformer_{_safe(display)}_ethnicity.png")

            for col in ("cell_type", "celltype"):
                if col in adata_gf.obs.columns:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    _umap_scatter(ax, coords, adata_gf.obs[col].astype(str).values,
                                  ct_palette, pt_size,
                                  f"{display}\nColored by Cell Type (Geneformer Embedding)",
                                  legend_outside=True)
                    fig.subplots_adjust(left=0.07, right=0.60, top=0.88, bottom=0.10)
                    save(fig, f"umap_geneformer_{_safe(display)}_celltype.png")
                    break

            if "source" in adata_gf.obs.columns:
                fig, ax = plt.subplots(figsize=(9, 7))
                counts  = adata_gf.obs["source"].value_counts()
                subtitle = "  ".join(f"{k}: {v:,}" for k, v in counts.items())
                _umap_scatter(ax, coords, adata_gf.obs["source"].astype(str).values,
                              SOURCE_PALETTE_UMAP, pt_size,
                              f"{display}\nReal vs Synthetic (Geneformer Embedding)",
                              subtitle=subtitle)
                plt.tight_layout()
                save(fig, f"umap_geneformer_{_safe(display)}_source.png")

            del adata_gf

        # [GF 8] Side-by-side comparison
        generate_sidebyside_umap(ds_key, ds_stem, ct_palette)

    print("\n  Individual UMAPs complete.")

# ============================================================
# scIB LOADER  [GF 6]
# ============================================================

def load_scib():
    rows = []
    summary_path = STEP3A_DIR / "benchmark_summary_all_modes.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            ds = row.get("dataset", "")
            entry = {"dataset": ds}
            for display, (col, *_) in SCIB_METRIC_MAP.items():
                entry[display] = float(row[col]) if col in row and pd.notna(row[col]) else np.nan
            rows.append(entry)
    else:
        # Fall back to individual bm_dict txt files
        for ds in DATASET_KEYS:
            txt = STEP3A_DIR / f"bm_dict_{ds}.txt"
            if not txt.exists():
                continue
            try:
                text = txt.read_text()
                import re
                match = re.search(r'_results:\s*(.*?)(?=\n\n[a-z_]|\Z)', text, re.DOTALL)
                if not match:
                    continue
                parsed = {}
                for line in match.group(1).splitlines():
                    line = line.strip()
                    if not line or line.startswith("Embedding") or line.startswith("Metric"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            parsed[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
                entry = {"dataset": ds}
                for display, (col, *_) in SCIB_METRIC_MAP.items():
                    entry[display] = parsed.get(col, np.nan)
                rows.append(entry)
            except Exception as e:
                print(f"  Warning parsing {txt.name}: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ============================================================
# FIG 1 — scIB
# ============================================================

def fig1_scib():
    df = load_scib()
    if df.empty:
        print("  Fig 1: No scIB data found — skipping")
        return

    for metric, (_, short, title, subtitle) in SCIB_METRIC_MAP.items():
        sub = df[["dataset", metric]].dropna(subset=[metric])
        sub = sub[sub[metric] != 0].copy()
        if sub.empty:
            continue
        hline  = 5.0 if metric == "iLISI" else None
        hlabel = "Theoretical max (5 groups, fully mixed)" if metric == "iLISI" else None
        _single_bar_figure(
            title=f"Fig 1 — scIB Embedding Quality (Geneformer)\n{title}",
            subtitle=subtitle, xlabel="Dataset Variant", ylabel=short,
            datasets=sub["dataset"].tolist(), values=sub[metric].tolist(),
            colors=[ds_color(d) for d in sub["dataset"]],
            fname=f"step3a_fig1_{metric}.png", hline=hline, hline_label=hlabel,
        )

# ============================================================
# FIG 2 — Disease Prediction  [GF 7]
# ============================================================

def fig2_step4a():
    path = STEP4A_DIR / "step4a_downstream_results_ethnicity_AR_EOS_geneformer.csv"
    if not path.exists():
        print("  Fig 2: Step 4a CSV not found — skipping")
        return

    df  = pd.read_csv(path)
    ref = "Proportional_2497"
    if ref in df["dataset"].unique():
        df = df[df["dataset"] == ref]

    models       = df["model"].unique()
    strategies   = df["strategy"].unique()
    x            = np.arange(len(strategies))
    width        = 0.22
    model_colors = ["#4C72B0", "#DD8452", "#55A868"]
    worst_col    = "disease_worst_ethnicity_acc"

    panels = [
        ("disease_accuracy",  "Overall Disease Prediction Accuracy",         "Accuracy",    "step4a_fig2_overall_accuracy.png"),
        ("disease_macro_f1",  "Macro F1 Score",                              "Macro F1",    "step4a_fig2_macro_f1.png"),
        (worst_col,           "Worst-Ethnicity Disease Prediction Accuracy",  "Accuracy",    "step4a_fig2_worst_ethnicity_accuracy.png"),
    ]

    for metric, panel_title, ylabel, fname in panels:
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.suptitle(f"Fig 2 — Disease Prediction Strategies (Geneformer)\n{panel_title}",
                     fontsize=17, fontweight="bold", y=1.02)
        ax.set_title(f"Dataset: Proportional (1,998 cells)   {ylabel}",
                     fontsize=13, pad=10, style="italic", color="#444444")

        if metric in df.columns:
            for i, (model, color) in enumerate(zip(models, model_colors)):
                sub  = df[df["model"] == model]
                vals = [float(sub[sub["strategy"] == s][metric].values[0])
                        if len(sub[sub["strategy"] == s]) > 0 else np.nan
                        for s in strategies]
                offset = (i - len(models) / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=model,
                              color=color, alpha=0.85, edgecolor="white", zorder=3)
                for bar, val in zip(bars, vals):
                    if pd.notna(val):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.007, f"{val:.3f}",
                                ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xlabel("Fairness Strategy", fontsize=14, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_DESCRIPTIONS.get(s, s) for s in strategies],
                           fontsize=12, ha="center")
        ax.set_ylim(0.3, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.legend(handles=[mpatches.Patch(facecolor=model_colors[i], alpha=0.85, label=m)
                            for i, m in enumerate(models)],
                  title="Classifier", fontsize=12, title_fontsize=13, loc="lower right")
        plt.tight_layout(pad=2.5)
        save(fig, fname)

# ============================================================
# FIG 3 — Model Robustness
# ============================================================

def fig3_step4b():
    path = STEP4B_DIR / "step4b_results_ethnicity_labeled_geneformer.csv"
    if not path.exists():
        print("  Fig 3: Step 4b CSV not found — skipping")
        return

    df      = pd.read_csv(path)
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    models   = df["model"].unique()
    strats   = df["strategy"].unique()

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        positions, data_vals, colors_list = [], [], []
        pos = 0
        group_centers, group_labels = [], []

        for strat in strats:
            group_start = pos
            for model in models:
                vals = sub[(sub["strategy"] == strat) & (sub["model"] == model)]["accuracy"].values
                if len(vals) > 0:
                    positions.append(pos); data_vals.append(vals)
                    colors_list.append(STRATEGY_PALETTE.get(strat, "gray")); pos += 1
            group_centers.append((group_start + pos - 1) / 2)
            group_labels.append(strat); pos += 0.8

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"Fig 3 — Model Robustness (Geneformer)\n"
                     f"Dataset: {ds_label(ds).replace(chr(10), ' ')}",
                     fontsize=17, fontweight="bold", y=1.02)
        if data_vals:
            bp = ax.boxplot(data_vals, positions=positions, widths=0.55,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="black", linewidth=2.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            flierprops=dict(marker="o", markersize=5, alpha=0.5))
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            all_vals = np.concatenate(data_vals)
            span = max(np.nanmax(all_vals) - np.nanmin(all_vals), 0.05)
            ax.set_ylim(max(0, np.nanmin(all_vals) - span * 0.5),
                        min(1.0, np.nanmax(all_vals) + span * 0.5))

        ax.set_xticks(group_centers)
        ax.set_xticklabels([STRATEGY_DESCRIPTIONS.get(s, s) for s in strats],
                           fontsize=13, ha="center")
        ax.set_ylabel("Disease Prediction Accuracy", fontsize=14, labelpad=10)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout(pad=2.5)
        save(fig, f"step4b_fig3_{_safe(ds)}.png")

# ============================================================
# FIG 4 — Learning Curves
# ============================================================

def fig4_learning_curves():
    path = STEP5_DIR / "step5_learning_curves_ethnicity_geneformer.csv"
    if not path.exists():
        print("  Fig 4: Step 5 CSV not found — skipping")
        return

    df           = pd.read_csv(path)
    datasets     = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    model_colors = {"LogReg": "#2196F3", "RandomForest": "#FF5722"}

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.suptitle(f"Fig 4 — Learning Curves (Geneformer)\n"
                     f"Dataset: {ds_label(ds).replace(chr(10), ' ')}",
                     fontsize=17, fontweight="bold", y=1.02)
        for model in df["model"].unique():
            msub = sub[sub["model"] == model].sort_values("n_train")
            if msub.empty:
                continue
            col = model_colors.get(model, "gray")
            ax.plot(msub["n_train"], msub["val_f1_mean"], "-o", color=col,
                    label=f"{model} — validation", linewidth=2.5, markersize=7, zorder=4)
            ax.fill_between(msub["n_train"],
                            msub["val_f1_mean"] - msub["val_f1_std"],
                            msub["val_f1_mean"] + msub["val_f1_std"],
                            alpha=0.15, color=col, zorder=3)
            ax.plot(msub["n_train"], msub["train_f1_mean"], "--", color=col,
                    alpha=0.5, label=f"{model} — train", linewidth=2.0)
        ax.set_xlabel("Training Set Size (cells)", fontsize=14, labelpad=10)
        ax.set_ylabel("Macro-F1 Score", fontsize=14, labelpad=10)
        ax.legend(fontsize=12, loc="lower right")
        ax.yaxis.grid(True, alpha=0.3); ax.xaxis.grid(True, alpha=0.3)
        ax.set_ylim(0.30, 1.05)
        plt.tight_layout(pad=2.5)
        save(fig, f"step5_fig4_learning_curves_{_safe(ds)}.png")

# ============================================================
# FIG 5 — Per-Ethnicity Fairness Deltas
# ============================================================

def fig5_step6():
    path = STEP6_DIR / "step6_per_ethnicity_diagnostics_geneformer.csv"
    if not path.exists():
        print("  Fig 5: Step 6 CSV not found — skipping")
        return

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = [str(i).strip().lower() for i in df.index]

    delta_cols = [(m, c) for m, c in df.columns if "delta" in str(c)]
    if not delta_cols:
        print("  Fig 5: No delta columns — skipping"); return

    metrics    = list(dict.fromkeys([m for m, _ in delta_cols]))
    strategies = list(dict.fromkeys([c.replace("delta_", "").replace("_vs_prop", "") for _, c in delta_cols]))
    groups     = df.index.tolist()

    metric_titles = {
        "silhouette": "Disease Silhouette Score Delta",
        "knn_mixing": "kNN Ethnicity Mixing Delta",
        "accuracy":   "Disease Prediction Accuracy Delta",
        "ari":        "ARI Delta", "nmi": "NMI Delta",
    }

    for eth in groups:
        eth_color = ETH_COLORS.get(eth, "#333333")
        for metric in metrics:
            vals, labels = [], []
            for strat in strategies:
                col_key = (metric, f"delta_{strat}_vs_prop")
                if col_key in df.columns:
                    v = df.loc[eth, col_key]
                    vals.append(float(v) if pd.notna(v) else 0.0)
                    labels.append(ds_label(strat).replace("\n", " "))
            if not vals:
                continue
            colors = ["#DD8452" if v >= 0 else "#C44E52" for v in vals]
            fig, ax = plt.subplots(figsize=(max(9, len(vals) * 2.6), 7))
            fig.suptitle(f"Fig 5 — Per-Ethnicity Fairness Deltas (Geneformer)\n"
                         f"Group: {eth.title()}   Metric: {metric_titles.get(metric, metric)}",
                         fontsize=17, fontweight="bold", y=1.02, color=eth_color)
            bars = ax.bar(range(len(vals)), vals, color=colors,
                          alpha=0.85, edgecolor="white", width=0.55, zorder=3)
            ax.axhline(0, color="black", linewidth=1.8, zorder=4)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=13, ha="center")
            ax.set_ylabel("Delta vs Proportional Baseline", fontsize=14, labelpad=10)
            ax.yaxis.grid(True, alpha=0.3, zorder=0)
            annotate_bars(ax, bars, vals, fmt="{:+.3f}", fontsize=12)
            ax.legend(handles=[
                mpatches.Patch(facecolor="#DD8452", label="Improvement"),
                mpatches.Patch(facecolor="#C44E52", label="Degradation"),
            ], fontsize=12, loc="best")
            plt.tight_layout(pad=2.5)
            save(fig, f"step6_fig5_{_safe(eth)}_{metric}.png")

# ============================================================
# FIG 6 — Representation Quality
# ============================================================

def fig6_step7():
    path = STEP7_DIR / "step7_per_ethnicity_diagnostics_geneformer.csv"
    if not path.exists():
        print("  Fig 6: Step 7 CSV not found — skipping"); return

    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].astype(str).str.strip()

    GROUP_COL = next((c for c in ("self_reported_ethnicity", "ethnicity") if c in df.columns), None)
    if GROUP_COL is None:
        print("  Fig 6: No group column — skipping"); return

    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip().str.lower()
    datasets_in   = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    metrics       = [m for m in ["celltype_purity", "within_ct_eth_mixing", "celltype_macroF1"] if m in df.columns]
    if not datasets_in or not metrics:
        print("  Fig 6: No data — skipping"); return

    metric_info = {
        "celltype_purity":       ("Cell-Type Neighbourhood Purity",    "Purity (0-1)",      "celltype-purity"),
        "within_ct_eth_mixing":  ("Within-Cell-Type Ethnicity Mixing", "Mixing fraction",    "within-ct-eth-mixing"),
        "celltype_macroF1":      ("Cell-Type Linear Probe Macro-F1",   "Macro-F1 (0-1)",     "celltype-macroF1"),
    }

    groups   = sorted(df[GROUP_COL].unique())
    n_groups = len(groups)
    width    = 0.14
    x        = np.arange(len(datasets_in))

    for metric in metrics:
        title, ylabel, fname_stem = metric_info[metric]
        fig, ax = plt.subplots(figsize=(max(16, len(datasets_in) * 4.5), 9))
        fig.suptitle(f"Fig 6 — Representation Quality (Geneformer)\n{title}",
                     fontsize=17, fontweight="bold", y=1.02)

        all_vals = [float(df[(df["dataset"] == ds) & (df[GROUP_COL] == eth)][metric].values[0])
                    for eth in groups for ds in datasets_in
                    if len(df[(df["dataset"] == ds) & (df[GROUP_COL] == eth)]) > 0]
        max_overall = max(all_vals) if all_vals else 1.0

        for i, eth in enumerate(groups):
            vals = [float(df[(df["dataset"] == ds) & (df[GROUP_COL] == eth)][metric].values[0])
                    if len(df[(df["dataset"] == ds) & (df[GROUP_COL] == eth)]) > 0 else np.nan
                    for ds in datasets_in]
            color  = ETH_COLORS.get(eth, "gray")
            offset = (i - n_groups / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=eth.title(),
                            color=color, alpha=0.85, edgecolor="white", zorder=3)
            for bar, val in zip(bars, vals):
                if pd.notna(val) and val >= max_overall * 0.08:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_overall * 0.018,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
        ax.set_xlabel("Dataset Variant", fontsize=14, labelpad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS.get(d, d) for d in datasets_in], fontsize=12, ha="center")
        ax.tick_params(axis="x", pad=12)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if "mixing" in metric:
            ax.axhspan(0.3, 0.7, alpha=0.08, color="green", zorder=1, label="Target (0.3-0.7)")
            ax.set_ylim(0, 1.08)
        else:
            ax.set_ylim(0, min(max_overall * 1.40, 1.08))
        ax.legend(title="Ethnicity Group", fontsize=12, title_fontsize=13,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
        fig.tight_layout(rect=[0, 0, 0.87, 0.93], pad=2.5)
        save(fig, f"step7_fig6_{fname_stem}.png")

# ============================================================
# FIG 7 — Ethnicity-Conditioned Disease Prediction
# ============================================================

def fig7_step8():
    per_eth_path = STEP8_DIR / "step8_per_ethnicity_disease_prediction_geneformer.csv"
    worst_path   = STEP8_DIR / "step8_worst_ethnicity_summary_geneformer.csv"

    if not per_eth_path.exists():
        print("  Fig 7: Step 8 CSV not found — skipping"); return

    per_eth = pd.read_csv(per_eth_path)
    per_eth["ethnicity"] = per_eth["ethnicity"].astype(str).str.strip().str.lower()
    per_eth["dataset"]   = per_eth["dataset"].astype(str).str.strip()
    worst    = pd.read_csv(worst_path) if worst_path.exists() else pd.DataFrame()
    datasets = [d for d in DATASET_ORDER if d in per_eth["dataset"].unique()]
    groups   = sorted(per_eth["ethnicity"].unique())

    # Panel A
    fig, ax = plt.subplots(figsize=(max(13, len(datasets) * 3.2), 8))
    fig.suptitle("Fig 7A — Ethnicity-Conditioned Disease Prediction (Geneformer)\n"
                 "Per-Ethnicity Accuracy by Dataset Variant",
                 fontsize=17, fontweight="bold", y=1.02)
    x = np.arange(len(datasets)); width = 0.18
    for i, eth in enumerate(sorted(groups)):
        sub    = per_eth[per_eth["ethnicity"] == eth].set_index("dataset").reindex(datasets)
        offset = (i - len(groups) / 2 + 0.5) * width
        color  = ETH_COLORS.get(eth, "gray")
        bars   = ax.bar(x + offset, sub["accuracy"], width, label=eth.title(),
                        color=color, alpha=0.85, edgecolor="white", zorder=3)
        for bar, val in zip(bars, sub["accuracy"]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dataset Variant", fontsize=14, labelpad=10)
    ax.set_ylabel("Disease Prediction Accuracy", fontsize=14, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([ds_label(d).replace("\n", " ") for d in datasets], fontsize=13, ha="center")
    ax.set_ylim(0.0, 1.12)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(title="Ethnicity Group", fontsize=12, title_fontsize=13, loc="upper right")
    plt.tight_layout(pad=2.5)
    save(fig, "step8_fig7_per_ethnicity_accuracy.png")

    # Panel B
    if not worst.empty:
        worst["dataset"] = worst["dataset"].astype(str).str.strip()
        ordered_ds    = [d for d in DATASET_ORDER if d in worst["dataset"].values]
        worst_ordered = worst.set_index("dataset").reindex(ordered_ds).reset_index()
        fig, ax = plt.subplots(figsize=(max(9, len(ordered_ds) * 2.6), 7))
        fig.suptitle("Fig 7B — Worst-Performing Ethnicity Group Accuracy (Geneformer)",
                     fontsize=17, fontweight="bold", y=1.02)
        colors = [ds_color(d) for d in worst_ordered["dataset"]]
        bars   = ax.bar(range(len(worst_ordered)), worst_ordered["worst_acc"],
                        color=colors, alpha=0.85, edgecolor="white", width=0.55, zorder=3)
        ax.set_xticks(range(len(worst_ordered)))
        ax.set_xticklabels([ds_label(d).replace("\n", " ") for d in worst_ordered["dataset"]],
                           fontsize=13, ha="center")
        for bar, row in zip(bars, worst_ordered.itertuples()):
            if pd.notna(row.worst_acc):
                ax.text(bar.get_x() + bar.get_width() / 2, row.worst_acc + 0.01,
                        f"{row.worst_acc:.3f}\n({row.worst_ethnicity})",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.12)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout(pad=2.5)
        save(fig, "step8_fig7_worst_ethnicity_accuracy.png")

    # Panel C
    na_rows = per_eth[per_eth["ethnicity"] == UNDERREP_GROUP].copy()
    if "delta_acc_vs_prop" in na_rows.columns and not na_rows.empty:
        na_plot = na_rows[na_rows["dataset"] != "Proportional_2497"].copy()
        na_plot = na_plot.set_index("dataset").reindex(
            [d for d in DATASET_ORDER if d != "Proportional_2497"]).reset_index()
        fig, ax = plt.subplots(figsize=(max(9, len(na_plot) * 2.6), 7))
        fig.suptitle("Fig 7C — African American Accuracy Delta vs Proportional (Geneformer)",
                     fontsize=17, fontweight="bold", y=1.02)
        if na_plot["delta_acc_vs_prop"].notna().any():
            colors = ["#DD8452" if v >= 0 else "#C44E52"
                      for v in na_plot["delta_acc_vs_prop"].fillna(0)]
            bars = ax.bar(range(len(na_plot)), na_plot["delta_acc_vs_prop"].fillna(0),
                          color=colors, alpha=0.85, edgecolor="white", width=0.55, zorder=3)
            ax.axhline(0, color="black", linewidth=2.0, zorder=4)
            ax.set_xticks(range(len(na_plot)))
            ax.set_xticklabels([ds_label(d).replace("\n", " ") for d in na_plot["dataset"]],
                               fontsize=13, ha="center")
            for bar, (_, row) in zip(bars, na_plot.iterrows()):
                val = row["delta_acc_vs_prop"]
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + (0.007 if val >= 0 else -0.03),
                            f"{val:+.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_ylabel("Delta Accuracy vs Proportional_2497 Baseline", fontsize=14, labelpad=10)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout(pad=2.5)
        save(fig, "step8_fig7_native_american_delta.png")

# ============================================================
# FIG 8 — Summary Heatmap
# ============================================================

def fig8_summary_heatmap():
    rows = []

    scib_df = load_scib()
    if not scib_df.empty:
        for _, r in scib_df.iterrows():
            rows.append({"dataset": r["dataset"],
                         "scIB NMI": r.get("NMI", np.nan),
                         "scIB ARI": r.get("ARI", np.nan),
                         "scIB iLISI": r.get("iLISI", np.nan)})

    p = STEP4A_DIR / "step4a_downstream_results_ethnicity_AR_EOS_geneformer.csv"
    if p.exists():
        df4  = pd.read_csv(p)
        base = df4[(df4["strategy"] == "Baseline") & (df4["model"] == "RandomForest")]
        for _, row in base.iterrows():
            rows.append({"dataset": row.get("dataset", ""),
                         "Disease Acc (RF)": row.get("disease_accuracy", np.nan),
                         "Worst-Eth Acc":    row.get("disease_worst_ethnicity_acc", np.nan)})

    p = STEP8_DIR / "step8_worst_ethnicity_summary_geneformer.csv"
    if p.exists():
        for _, row in pd.read_csv(p).iterrows():
            rows.append({"dataset": row["dataset"], "Worst-Eth (S8)": row.get("worst_acc", np.nan)})

    p = STEP8_DIR / "step8_per_ethnicity_disease_prediction_geneformer.csv"
    if p.exists():
        df8b = pd.read_csv(p)
        na   = df8b[df8b["ethnicity"].str.lower().str.strip() == UNDERREP_GROUP]
        for _, row in na.iterrows():
            rows.append({"dataset": row["dataset"],
                         "African Am. Acc":   row.get("accuracy", np.nan),
                         "African Am. Delta": row.get("delta_acc_vs_prop", np.nan)})

    if not rows:
        print("  Fig 8: No data — skipping"); return

    summary = pd.DataFrame(rows).groupby("dataset").mean(numeric_only=True).reset_index()
    ordered = [d for d in DATASET_ORDER if d in summary["dataset"].values]
    summary = summary.set_index("dataset").reindex(ordered).dropna(how="all")
    summary.index = [ds_label(d).replace("\n", " ") for d in summary.index]

    valid_cols = [c for c in summary.columns if summary[c].notna().any() and (summary[c] != 0).any()]
    summary    = summary[valid_cols]
    if summary.empty:
        print("  Fig 8: No valid columns — skipping"); return

    norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(max(13, len(valid_cols) * 2.2), max(5, len(summary) * 1.3 + 2)))
    fig.suptitle("Fig 8 — Summary Heatmap (Geneformer)\n"
                 "Green = best   Red = worst   Raw values shown",
                 fontsize=17, fontweight="bold", y=1.03)
    im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(valid_cols)))
    ax.set_xticklabels(valid_cols, rotation=35, ha="right", fontsize=13)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index, fontsize=14)
    for i in range(len(summary)):
        for j, col in enumerate(valid_cols):
            val = summary.iloc[i][col]
            if pd.notna(val):
                norm_val  = norm.iloc[i][col]
                txt_color = "white" if (norm_val < 0.25 or norm_val > 0.75) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=12, color=txt_color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Normalized Score (0=worst, 1=best)", fontsize=13)
    plt.tight_layout(pad=2.5)
    save(fig, "step9_fig8_summary_heatmap.png")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\nSTEP 9 — Visualization Pipeline (CRC ETHNICITY, Geneformer V2-316M)")
    print(f"   Output: {OUTDIR}\n")

    print("Generating UMAP PNGs...")
    try:
        generate_individual_umaps()
    except Exception as e:
        print(f"  UMAPs failed: {e}")
        import traceback; traceback.print_exc()

    steps = [
        ("Fig 1  — scIB Embedding Quality",          fig1_scib),
        ("Fig 2  — Disease Prediction Strategies",   fig2_step4a),
        ("Fig 3  — Model Robustness",                fig3_step4b),
        ("Fig 4  — Learning Curves",                 fig4_learning_curves),
        ("Fig 5  — Per-Ethnicity Fairness Deltas",   fig5_step6),
        ("Fig 6  — Representation Quality",          fig6_step7),
        ("Fig 7  — Ethnicity-Conditioned Pred.",     fig7_step8),
        ("Fig 8  — Summary Heatmap",                 fig8_summary_heatmap),
    ]

    completed, failed = [], []
    for label, fn in steps:
        print(f"\nGenerating {label}...")
        try:
            fn(); completed.append(label)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(label)

    all_pngs  = sorted(OUTDIR.glob("*.png"))
    umap_pngs = [f for f in all_pngs if "umap_" in f.name]
    fig_pngs  = [f for f in all_pngs if "umap_" not in f.name]

    print(f"\n{'='*60}")
    print("STEP 9 COMPLETE")
    print(f"  Figures : {len(completed)}/{len(steps)} completed")
    if failed:
        print(f"  Failed  : {[f for f in failed]}")
    print(f"  UMAPs   : {len(umap_pngs)} PNGs")
    print(f"  Figures : {len(fig_pngs)} PNGs")
    print(f"\nOutput: {OUTDIR}")


if __name__ == "__main__":
    main()
