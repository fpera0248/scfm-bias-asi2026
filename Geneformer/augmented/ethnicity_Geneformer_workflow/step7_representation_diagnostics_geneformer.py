#!/usr/bin/env python3
"""
STEP 7 — Representation Quality & Fairness Diagnostics (ETHNICITY, Geneformer V2-316M)

Changes vs scFoundation version:
  [GF 1] BASE/OUTDIR  : scfoundation augmentedv4 -> Geneformer augmented
  [GF 2] EMB_KEY      : X_scfoundation -> X_geneformer
  [GF 3] FILES        : updated filenames + cell counts from step3b
  [GF 4] Zero-vector filter added (tokenizer dropout)
  [GF 5] Reference baseline: Proportional_2497
"""

import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = pathlib.Path(
    "/data/Geneformer/augmented/ethnicity_Geneformer_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step7_representation_diagnostics_ethnicity_geneformer"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Ethnicity_Pilot"

FILES = {
    "Proportional_2497":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2497_labeled_geneformer.h5ad",
    "BalancedAugmented_2143Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_2143Each_labeled_geneformer.h5ad",
    "BalancedUpsampled_2143Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_2143Each_labeled_geneformer.h5ad",
    "Downsampled_48Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_48Each_labeled_geneformer.h5ad",
}

EMB_KEY      = "X_geneformer"
ETH_KEY      = "self_reported_ethnicity"
CELL_KEY     = "cell_type"
SOURCE_COL   = "source"
KNN_K        = 15
MIN_CT_SIZE  = 20
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonicalize_ethnicity(series):
    return series.astype(str).str.strip().str.lower()


def load_dataset(label, path):
    log(f"  Loading {label}: {path.name}")
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()

    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing '{EMB_KEY}' in {path.name}")
    if ETH_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{ETH_KEY}' in {path.name}")
    if CELL_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{CELL_KEY}' in {path.name}")

    ad.obs[ETH_KEY] = canonicalize_ethnicity(ad.obs[ETH_KEY])

    if SOURCE_COL in ad.obs.columns:
        ad.obs["is_synthetic"] = (ad.obs[SOURCE_COL] == "synthetic")
    else:
        ad.obs["is_synthetic"] = False

    emb       = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask  = np.isnan(emb).any(axis=1)
    zero_mask = (np.abs(emb).sum(axis=1) == 0)
    bad_mask  = nan_mask | zero_mask
    if bad_mask.any():
        log(f"  Dropping {bad_mask.sum()} cells (NaN or zero embeddings)")
        ad = ad[~bad_mask].copy()

    log(f"  {ad.n_obs:,} cells | Ethnicity dist: "
        f"{ad.obs[ETH_KEY].value_counts().to_dict()}")
    return ad


def celltype_purity(ad):
    emb  = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    cell = ad.obs[CELL_KEY].astype(str).values
    nn   = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb)
    neigh  = nn.kneighbors(return_distance=False)[:, 1:]
    purity = np.mean(cell[neigh] == cell[:, None], axis=1)
    df = pd.DataFrame({ETH_KEY: ad.obs[ETH_KEY].astype(str).values, "purity": purity})
    return df.groupby(ETH_KEY, observed=False)["purity"].mean()


def within_celltype_mixing(ad):
    results = []
    grouped = ad.obs.groupby(CELL_KEY, observed=False).groups

    for ct, sub_idx in grouped.items():
        if len(sub_idx) < MIN_CT_SIZE:
            continue
        pos  = ad.obs.index.get_indexer_for(sub_idx)
        emb  = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[pos]
        eth  = ad.obs.iloc[pos][ETH_KEY].astype(str).values
        nn   = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb)
        neigh  = nn.kneighbors(return_distance=False)[:, 1:]
        mixing = np.mean(eth[neigh] != eth[:, None], axis=1)
        df = pd.DataFrame({ETH_KEY: eth, "mixing": mixing})
        results.append(df.groupby(ETH_KEY, observed=False)["mixing"].mean())

    if not results:
        log("  WARNING: No cell types met MIN_CT_SIZE threshold for mixing calc.")
        return pd.Series(dtype=float)

    return pd.concat(results, axis=1).mean(axis=1)


def celltype_linear_probe(ad):
    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y = ad.obs[CELL_KEY].astype(str).values
    g = ad.obs[ETH_KEY].astype(str).values

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    preds = clf.predict(X)

    df = pd.DataFrame({ETH_KEY: g, "y_true": y, "y_pred": preds})
    scores = {}
    for eth in np.unique(g):
        sub = df[df[ETH_KEY] == eth]
        if sub["y_true"].nunique() < 2:
            continue
        scores[eth] = round(float(f1_score(sub["y_true"], sub["y_pred"], average="macro")), 4)
    return pd.Series(scores)


def main():
    log("=" * 70)
    log("STEP 7 -- Representation Quality & Fairness Diagnostics (ETHNICITY, Geneformer)")
    log("=" * 70)

    datasets = {}
    for label, path in FILES.items():
        if not path.exists():
            log(f"  Skipping {label} -- file not found: {path.name}")
            continue
        datasets[label] = load_dataset(label, path)

    if "Proportional_2497" not in datasets:
        raise RuntimeError("Proportional_2497 dataset required as reference baseline.")

    all_results = []
    for label, ad in datasets.items():
        log(f"\n>> Computing diagnostics for {label}")

        purity = celltype_purity(ad)
        mixing = within_celltype_mixing(ad)
        probe  = celltype_linear_probe(ad)

        df = pd.concat([purity, mixing, probe], axis=1)
        df.columns = ["celltype_purity", "within_ct_eth_mixing", "celltype_macroF1"]
        df = df.reset_index().rename(columns={"index": ETH_KEY})
        df["dataset"] = label
        all_results.append(df)

        log(f"  Purity:  {purity.to_dict()}")
        log(f"  Mixing:  {mixing.to_dict()}")
        log(f"  ProbeF1: {probe.to_dict()}")

    df_all = pd.concat(all_results, ignore_index=True)

    pivot = df_all.pivot_table(
        index=ETH_KEY,
        columns="dataset",
        values=["celltype_purity", "within_ct_eth_mixing", "celltype_macroF1"],
        observed=False,
    )

    ref = "Proportional_2497"
    for metric in ["celltype_purity", "within_ct_eth_mixing", "celltype_macroF1"]:
        for ds in [d for d in datasets if d != ref]:
            col_ds  = (metric, ds)
            col_ref = (metric, ref)
            if col_ds in pivot.columns and col_ref in pivot.columns:
                pivot[(metric, f"delta_{ds}_vs_prop")] = pivot[col_ds] - pivot[col_ref]

    out_csv = OUTDIR / "step7_per_ethnicity_diagnostics_geneformer.csv"
    df_all.to_csv(out_csv, index=False)
    log(f"\nCSV saved -> {out_csv}")

    out_txt = OUTDIR / "step7_summary_ethnicity_geneformer.txt"
    lines = [
        "STEP 7 -- REPRESENTATION QUALITY & FAIRNESS DIAGNOSTICS (ETHNICITY, Geneformer)",
        "=" * 70, "",
        "NOTE: With 5 ethnicity groups, random expected kNN mixing is ~0.80",
        "(4 out of 5 neighbours expected from other groups by chance).",
        "Values well below 0.80 indicate group-specific clustering.",
        "",
        "=" * 70,
        "FULL PIVOT TABLE",
        "=" * 70,
        pivot.to_string(),
        "",
        "=" * 70,
        "DELTA SUMMARY (vs Proportional_2497 baseline)",
        "=" * 70,
    ]

    delta_cols = [c for c in pivot.columns if "delta" in str(c[1])]
    lines.append(pivot[delta_cols].sort_index().to_string() if delta_cols else "No delta columns produced.")

    out_txt.write_text("\n".join(lines))
    log(f"Report saved -> {out_txt}")
    log("\nSTEP 7 COMPLETE")


if __name__ == "__main__":
    main()
