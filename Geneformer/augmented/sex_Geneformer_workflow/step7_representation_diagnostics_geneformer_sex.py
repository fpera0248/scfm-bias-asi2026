#!/usr/bin/env python3
"""
STEP 7 — Representation Quality & Fairness Diagnostics (SEX, Geneformer V2-316M)

Changes vs ethnicity version:
  [SEX 1] BASE/OUTDIR  -> sex_Geneformer_workflow
  [SEX 2] SEX_KEY      -> "sex"
  [SEX 3] FILES        -> sex filenames (1413Each, 586Each, 1999)
  [SEX 4] Reference baseline -> Proportional_1999
  [SEX 5] Random expected kNN mixing ~0.50 with 2 groups
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
    "/oscar/home/fperalta/data/fperalta/Geneformer/augmented/sex_Geneformer_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step7_representation_diagnostics_sex_geneformer"   # [SEX 1]
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Sex_Pilot"

FILES = {                                                                 # [SEX 3]
    "Proportional_1999":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_1999_labeled_geneformer.h5ad",
    "BalancedAugmented_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1413Each_labeled_geneformer.h5ad",
    "BalancedUpsampled_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_labeled_geneformer.h5ad",
    "Downsampled_586Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_586Each_labeled_geneformer.h5ad",
}

EMB_KEY      = "X_geneformer"
SEX_KEY      = "sex"                                                      # [SEX 2]
CELL_KEY     = "cell_type"
SOURCE_COL   = "source"
KNN_K        = 15
MIN_CT_SIZE  = 20
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonicalize_sex(series):
    return series.astype(str).str.strip().str.lower()


def load_dataset(label, path):
    log(f"  Loading {label}: {path.name}")
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()

    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing '{EMB_KEY}' in {path.name}")
    if SEX_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{SEX_KEY}' in {path.name}")
    if CELL_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{CELL_KEY}' in {path.name}")

    ad.obs[SEX_KEY] = canonicalize_sex(ad.obs[SEX_KEY])

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

    log(f"  {ad.n_obs:,} cells | Sex dist: "
        f"{ad.obs[SEX_KEY].value_counts().to_dict()}")
    return ad


def celltype_purity(ad):
    emb  = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    cell = ad.obs[CELL_KEY].astype(str).values
    nn   = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb)
    neigh  = nn.kneighbors(return_distance=False)[:, 1:]
    purity = np.mean(cell[neigh] == cell[:, None], axis=1)
    df = pd.DataFrame({SEX_KEY: ad.obs[SEX_KEY].astype(str).values, "purity": purity})
    return df.groupby(SEX_KEY, observed=False)["purity"].mean()


def within_celltype_mixing(ad):
    results = []
    grouped = ad.obs.groupby(CELL_KEY, observed=False).groups

    for ct, sub_idx in grouped.items():
        if len(sub_idx) < MIN_CT_SIZE:
            continue
        pos  = ad.obs.index.get_indexer_for(sub_idx)
        emb  = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[pos]
        sex  = ad.obs.iloc[pos][SEX_KEY].astype(str).values
        nn   = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb)
        neigh  = nn.kneighbors(return_distance=False)[:, 1:]
        mixing = np.mean(sex[neigh] != sex[:, None], axis=1)
        df = pd.DataFrame({SEX_KEY: sex, "mixing": mixing})
        results.append(df.groupby(SEX_KEY, observed=False)["mixing"].mean())

    if not results:
        log("  WARNING: No cell types met MIN_CT_SIZE threshold for mixing calc.")
        return pd.Series(dtype=float)

    return pd.concat(results, axis=1).mean(axis=1)


def celltype_linear_probe(ad):
    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y = ad.obs[CELL_KEY].astype(str).values
    g = ad.obs[SEX_KEY].astype(str).values

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    preds = clf.predict(X)

    df = pd.DataFrame({SEX_KEY: g, "y_true": y, "y_pred": preds})
    scores = {}
    for grp in np.unique(g):
        sub = df[df[SEX_KEY] == grp]
        if sub["y_true"].nunique() < 2:
            continue
        scores[grp] = round(float(f1_score(sub["y_true"], sub["y_pred"], average="macro")), 4)
    return pd.Series(scores)


def main():
    log("=" * 70)
    log("STEP 7 -- Representation Quality & Fairness Diagnostics (SEX, Geneformer)")
    log("=" * 70)

    datasets = {}
    for label, path in FILES.items():
        if not path.exists():
            log(f"  Skipping {label} -- file not found: {path.name}")
            continue
        datasets[label] = load_dataset(label, path)

    if "Proportional_1999" not in datasets:                              # [SEX 4]
        raise RuntimeError("Proportional_1999 dataset required as reference baseline.")

    all_results = []
    for label, ad in datasets.items():
        log(f"\n>> Computing diagnostics for {label}")

        purity = celltype_purity(ad)
        mixing = within_celltype_mixing(ad)
        probe  = celltype_linear_probe(ad)

        df = pd.concat([purity, mixing, probe], axis=1)
        df.columns = ["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"]
        df = df.reset_index().rename(columns={"index": SEX_KEY})
        df["dataset"] = label
        all_results.append(df)

        log(f"  Purity:  {purity.to_dict()}")
        log(f"  Mixing:  {mixing.to_dict()}")
        log(f"  ProbeF1: {probe.to_dict()}")

    df_all = pd.concat(all_results, ignore_index=True)

    pivot = df_all.pivot_table(
        index=SEX_KEY,
        columns="dataset",
        values=["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"],
        observed=False,
    )

    ref = "Proportional_1999"                                            # [SEX 4]
    for metric in ["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"]:
        for ds in [d for d in datasets if d != ref]:
            col_ds  = (metric, ds)
            col_ref = (metric, ref)
            if col_ds in pivot.columns and col_ref in pivot.columns:
                pivot[(metric, f"delta_{ds}_vs_prop")] = pivot[col_ds] - pivot[col_ref]

    out_csv = OUTDIR / "step7_per_sex_diagnostics_geneformer.csv"
    df_all.to_csv(out_csv, index=False)
    log(f"\nCSV saved -> {out_csv}")

    out_txt = OUTDIR / "step7_summary_sex_geneformer.txt"
    lines = [
        "STEP 7 -- REPRESENTATION QUALITY & FAIRNESS DIAGNOSTICS (SEX, Geneformer)",
        "=" * 70, "",
        "NOTE: With 2 sex groups, random expected kNN mixing is ~0.50",   # [SEX 5]
        "(1 out of 2 neighbours expected from other group by chance).",
        "Values well below 0.50 indicate sex-specific clustering.",
        "",
        "=" * 70,
        "FULL PIVOT TABLE",
        "=" * 70,
        pivot.to_string(),
        "",
        "=" * 70,
        "DELTA SUMMARY (vs Proportional_1999 baseline)",
        "=" * 70,
    ]

    delta_cols = [c for c in pivot.columns if "delta" in str(c[1])]
    lines.append(pivot[delta_cols].sort_index().to_string() if delta_cols else "No delta columns produced.")

    out_txt.write_text("\n".join(lines))
    log(f"Report saved -> {out_txt}")
    log("\nSTEP 7 COMPLETE")


if __name__ == "__main__":
    main()
