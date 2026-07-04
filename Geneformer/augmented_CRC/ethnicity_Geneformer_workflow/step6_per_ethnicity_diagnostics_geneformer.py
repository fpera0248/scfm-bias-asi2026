#!/usr/bin/env python3
"""
STEP 6 — Per-Ethnicity Fairness Diagnostics (CRC ETHNICITY, Geneformer V2-316M)

Changes vs scFoundation version:
  [GF 1] BASE/OUTDIR  : scfoundation augmentedv4 -> Geneformer augmented
  [GF 2] EMB_KEY      : X_scfoundation -> X_geneformer
  [GF 3] FILES        : updated filenames + cell counts from step3b
  [GF 4] Zero-vector filter added (tokenizer dropout)
"""

import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step6_outputs_ethnicity_geneformer"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "CRC_Eth_Pilot"

FILES = {
    "Proportional_2497":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2497_labeled_geneformer.h5ad",
    "BalancedAugmented_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_labeled_geneformer.h5ad",
    "BalancedUpsampled_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_labeled_geneformer.h5ad",
    "Downsampled_48Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_48Each_labeled_geneformer.h5ad",
}

EMB_KEY      = "X_geneformer"
GROUP_KEY    = "self_reported_ethnicity"
DISEASE_COL  = "disease"
CELL_KEY     = "cell_type"
SOURCE_COL   = "source"
KNN_K        = 15
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonicalize_ethnicity(series):
    return series.astype(str).str.strip().str.lower()


def to_binary_disease(x):
    if pd.isna(x):
        return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"


def load_dataset(label, path):
    log(f"  Loading {label}: {path.name}")
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()

    if GROUP_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{GROUP_KEY}' in {path.name}")
    if DISEASE_COL not in ad.obs.columns:
        raise RuntimeError(f"Missing '{DISEASE_COL}' in {path.name}")
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing embedding '{EMB_KEY}' in {path.name}")

    ad.obs[GROUP_KEY]        = canonicalize_ethnicity(ad.obs[GROUP_KEY])
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)

    if SOURCE_COL in ad.obs.columns:
        ad.obs["is_synthetic"] = (ad.obs[SOURCE_COL] == "synthetic")
    else:
        ad.obs["is_synthetic"] = False

    emb      = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask  = np.isnan(emb).any(axis=1)
    zero_mask = (np.abs(emb).sum(axis=1) == 0)
    bad_mask  = nan_mask | zero_mask
    if bad_mask.any():
        log(f"  Dropping {bad_mask.sum()} cells (NaN or zero embeddings)")
        ad = ad[~bad_mask].copy()

    log(f"  {ad.n_obs:,} cells | Ethnicity dist: "
        f"{ad.obs[GROUP_KEY].value_counts().to_dict()}")
    return ad


def silhouette_per_ethnicity(ad):
    real_labeled = (~ad.obs["is_synthetic"]) & pd.notna(ad.obs["disease_binary"])
    ad_r = ad[real_labeled].copy()
    if ad_r.n_obs < 50:
        return pd.DataFrame()

    X = np.array(ad_r.obsm[EMB_KEY], dtype=np.float32)
    y = ad_r.obs["disease_binary"].values
    rows = []
    for eth in sorted(ad_r.obs[GROUP_KEY].unique()):
        m = ad_r.obs[GROUP_KEY] == eth
        if m.sum() < 10:
            continue
        y_s = y[m.values]
        if len(np.unique(y_s)) < 2:
            continue
        sil = silhouette_score(X[m.values], y_s, sample_size=min(3000, m.sum()))
        rows.append({"ethnicity": eth, "silhouette": round(float(sil), 4)})
    return pd.DataFrame(rows)


def knn_mixing_per_ethnicity(ad, k=KNN_K):
    X      = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    groups = ad.obs[GROUP_KEY].astype(str).values
    nn     = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh  = nn.kneighbors(return_distance=False)[:, 1:]
    mixing = np.mean(groups[neigh] != groups[:, None], axis=1)
    df = pd.DataFrame({"ethnicity": groups, "knn_mixing": mixing})
    return df.groupby("ethnicity")["knn_mixing"].mean().reset_index()


def accuracy_per_ethnicity(ad, seed=RANDOM_STATE):
    valid = pd.notna(ad.obs["disease_binary"]) & pd.notna(ad.obs[GROUP_KEY])
    if valid.sum() < 50:
        return {}

    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[valid.values]
    y = ad.obs.loc[valid, "disease_binary"].values
    g = ad.obs.loc[valid, GROUP_KEY].values

    if len(np.unique(y)) < 2:
        return {}

    counts   = pd.Series(y).value_counts()
    stratify = y if (counts >= 2).all() else None

    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=0.2, stratify=stratify, random_state=seed
    )
    clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    per_eth = {}
    for eth in np.unique(g_te):
        m = g_te == eth
        if m.sum() < 5:
            continue
        per_eth[eth] = round(float(accuracy_score(y_te[m], preds[m])), 4)
    return per_eth


def ari_nmi_per_ethnicity(ad):
    if CELL_KEY not in ad.obs.columns:
        return pd.DataFrame()

    real = (~ad.obs["is_synthetic"])
    ad_r = ad[real].copy()
    if ad_r.n_obs < 50:
        return pd.DataFrame()

    sc.pp.neighbors(ad_r, use_rep=EMB_KEY)
    sc.tl.leiden(ad_r, resolution=0.3, flavor="igraph", directed=False,
                 n_iterations=2, key_added="leiden")

    rows = []
    for eth in sorted(ad_r.obs[GROUP_KEY].unique()):
        sub = ad_r[ad_r.obs[GROUP_KEY] == eth]
        if sub.n_obs < 10:
            continue
        y_true = sub.obs[CELL_KEY].astype(str).values
        y_pred = sub.obs["leiden"].astype(str).values
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            continue
        rows.append({
            "ethnicity": eth,
            "ari": round(float(adjusted_rand_score(y_true, y_pred)), 4),
            "nmi": round(float(normalized_mutual_info_score(y_true, y_pred)), 4),
        })
    return pd.DataFrame(rows)


def main():
    log("=" * 70)
    log("STEP 6 -- Per-Ethnicity Fairness Diagnostics (CRC ETHNICITY, Geneformer)")
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
        log(f"\n>> Computing metrics for {label}")

        sil_df = silhouette_per_ethnicity(ad)
        mix_df = knn_mixing_per_ethnicity(ad)
        acc_d  = accuracy_per_ethnicity(ad)
        acc_df = pd.DataFrame([{"ethnicity": e, "accuracy": a} for e, a in acc_d.items()])
        ari_df = ari_nmi_per_ethnicity(ad)

        df = sil_df.copy()
        for other in [mix_df, acc_df, ari_df]:
            if not other.empty:
                df = (df.merge(other, on="ethnicity", how="outer")
                      if not df.empty else other)

        if not df.empty:
            df["dataset"] = label
            all_results.append(df)
            log(f"  Metrics computed: {list(df.columns)}")

    if not all_results:
        raise RuntimeError("No metrics produced -- check input files.")

    df_all = pd.concat(all_results, ignore_index=True)

    metric_cols = [c for c in ["silhouette", "knn_mixing", "accuracy", "ari", "nmi"]
                   if c in df_all.columns]

    pivot = df_all.pivot_table(index="ethnicity", columns="dataset", values=metric_cols)

    ref = "Proportional_2497"
    for ds in [d for d in datasets if d != ref]:
        for metric in metric_cols:
            col_ds  = (metric, ds)
            col_ref = (metric, ref)
            if col_ds in pivot.columns and col_ref in pivot.columns:
                pivot[(metric, f"delta_{ds}_vs_prop")] = pivot[col_ds] - pivot[col_ref]

    out_csv = OUTDIR / "step6_per_ethnicity_diagnostics_geneformer.csv"
    pivot.to_csv(out_csv)
    log(f"\nCSV saved -> {out_csv}")

    out_txt = OUTDIR / "step6_per_ethnicity_diagnostics_report_geneformer.txt"
    lines = [
        "STEP 6 -- PER-ETHNICITY FAIRNESS DIAGNOSTICS (CRC ETHNICITY, Geneformer)",
        "=" * 70, "",
        "INTERPRETATION GUIDE",
        "-" * 40,
        "SILHOUETTE: higher = better disease separation per ethnicity group.",
        "KNN MIXING: 0.3-0.7 target; random expected ~0.80 with 5 groups.",
        "ACCURACY DELTA: positive = group benefited from balancing strategy.",
        "ARI/NMI: negative delta = cell-type structure lost.",
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
    log("\nSTEP 6 COMPLETE")


if __name__ == "__main__":
    main()
