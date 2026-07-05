#!/usr/bin/env python3
"""
STEP 8 — Ethnicity-Conditioned Disease Prediction (CRC ETHNICITY, Geneformer V2-316M)

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

BASE = pathlib.Path(
    "/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step8_eth_conditioned_disease_geneformer"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "CRC_Eth_Pilot"

FILES = {
    "Proportional_2497":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2497_labeled_geneformer.h5ad",
    "BalancedAugmented_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_labeled_geneformer.h5ad",
    "BalancedUpsampled_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_labeled_geneformer.h5ad",
    "Downsampled_48Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_48Each_labeled_geneformer.h5ad",
}

EMB_KEY        = "X_geneformer"
DISEASE_COL    = "disease"
ETH_KEY        = "self_reported_ethnicity"
SOURCE_COL     = "source"
UNDERREP_GROUP = "african american"
MIN_SAMPLES    = 2
TEST_SIZE      = 0.20
RANDOM_STATE   = 42


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

    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing '{EMB_KEY}' in {path.name}")
    if DISEASE_COL not in ad.obs.columns:
        raise RuntimeError(f"Missing '{DISEASE_COL}' in {path.name}")
    if ETH_KEY not in ad.obs.columns:
        raise RuntimeError(f"Missing '{ETH_KEY}' in {path.name}")

    ad.obs[ETH_KEY]          = canonicalize_ethnicity(ad.obs[ETH_KEY])
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)

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


def train_and_evaluate(ad, dataset_name, seed=RANDOM_STATE):
    SMALL_GROUP_THRESHOLD = 30
    MIN_TEST_FROM_SMALL   = 3

    valid = pd.notna(ad.obs["disease_binary"]) & pd.notna(ad.obs[ETH_KEY])
    if valid.sum() < 40:
        log(f"  Too few valid cells ({valid.sum()}) -- skipping {dataset_name}")
        return pd.DataFrame()

    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[valid.values]
    y = ad.obs.loc[valid, "disease_binary"].values
    g = ad.obs.loc[valid, ETH_KEY].values

    if len(np.unique(y)) < 2:
        log(f"  Only one disease class -- skipping {dataset_name}")
        return pd.DataFrame()

    counts   = pd.Series(y).value_counts()
    stratify = y if (counts >= 2).all() else None

    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=TEST_SIZE, stratify=stratify, random_state=seed
    )

    rng = np.random.default_rng(seed)
    all_groups     = set(np.unique(g))
    groups_in_test = set(np.unique(g_te))
    missing        = all_groups - groups_in_test
    small_underrep = {eth for eth in all_groups
                      if (g == eth).sum() < SMALL_GROUP_THRESHOLD
                      and (g_te == eth).sum() < MIN_SAMPLES}
    to_boost = missing | small_underrep

    if to_boost:
        log(f"  Boosting test representation for: {to_boost}")
    for eth in to_boost:
        tr_idx = np.where(g_tr == eth)[0]
        if len(tr_idx) == 0:
            log(f"  WARNING: '{eth}' has no cells in train either -- skipping boost.")
            continue
        n_move = min(MIN_TEST_FROM_SMALL, len(tr_idx))
        move   = rng.choice(tr_idx, size=n_move, replace=False)
        keep   = np.setdiff1d(np.arange(len(g_tr)), move)
        X_te = np.vstack([X_te, X_tr[move]])
        y_te = np.concatenate([y_te, y_tr[move]])
        g_te = np.concatenate([g_te, g_tr[move]])
        X_tr = X_tr[keep]; y_tr = y_tr[keep]; g_tr = g_tr[keep]
        log(f"  Moved {n_move} '{eth}' cells: train->test.")

    clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    rows = []
    for eth in np.unique(g_te):
        m = g_te == eth
        if m.sum() < MIN_SAMPLES:
            log(f"  Skipping eth='{eth}' -- only {m.sum()} test cells")
            continue
        acc  = accuracy_score(y_te[m], preds[m])
        bacc = balanced_accuracy_score(y_te[m], preds[m])
        f1   = f1_score(y_te[m], preds[m], average="macro")
        rows.append({
            "dataset":           dataset_name,
            "ethnicity":         eth,
            "accuracy":          round(float(acc),  4),
            "balanced_accuracy": round(float(bacc), 4),
            "macro_f1":          round(float(f1),   4),
            "n_test_cells":      int(m.sum()),
            "is_underrep":       eth == UNDERREP_GROUP,
            "test_boosted":      eth in to_boost,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        worst_row = df.loc[df["accuracy"].idxmin()]
        log(f"  Worst ethnicity: {worst_row['ethnicity']} | "
            f"acc={worst_row['accuracy']:.4f} | f1={worst_row['macro_f1']:.4f}")
    return df


def main():
    log("=" * 70)
    log("STEP 8 -- Ethnicity-Conditioned Disease Prediction (CRC ETHNICITY, Geneformer)")
    log("=" * 70)

    all_dfs = []
    for label, path in FILES.items():
        if not path.exists():
            log(f"  Skipping {label} -- file not found: {path.name}")
            continue
        log(f"\n>> Evaluating {label}")
        ad = load_dataset(label, path)
        df = train_and_evaluate(ad, label)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No results produced -- check input files.")

    per_eth_df = pd.concat(all_dfs, ignore_index=True)

    ref = "Proportional_2497"
    prop_rows = per_eth_df[per_eth_df["dataset"] == ref][["ethnicity", "accuracy", "macro_f1"]]
    prop_rows = prop_rows.rename(columns={"accuracy": "prop_acc", "macro_f1": "prop_f1"})
    per_eth_df = per_eth_df.merge(prop_rows, on="ethnicity", how="left")
    per_eth_df["delta_acc_vs_prop"] = (per_eth_df["accuracy"] - per_eth_df["prop_acc"]).round(4)
    per_eth_df["delta_f1_vs_prop"]  = (per_eth_df["macro_f1"] - per_eth_df["prop_f1"]).round(4)

    worst_eth_df = (
        per_eth_df.loc[per_eth_df.groupby("dataset")["accuracy"].idxmin()]
        [["dataset", "ethnicity", "accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .rename(columns={"ethnicity": "worst_ethnicity", "accuracy": "worst_acc", "macro_f1": "worst_f1"})
        .reset_index(drop=True)
    )

    underrep_df = (
        per_eth_df[per_eth_df["is_underrep"]]
        .groupby("dataset")[["accuracy", "balanced_accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .mean()
        .reset_index()
    )

    out_per_eth = OUTDIR / "step8_per_ethnicity_disease_prediction_geneformer.csv"
    per_eth_df.to_csv(out_per_eth, index=False)
    log(f"\nPer-ethnicity CSV -> {out_per_eth}")

    out_worst = OUTDIR / "step8_worst_ethnicity_summary_geneformer.csv"
    worst_eth_df.to_csv(out_worst, index=False)
    log(f"Worst-ethnicity CSV -> {out_worst}")

    out_underrep = OUTDIR / "step8_underrepresented_ethnicity_summary_geneformer.csv"
    underrep_df.to_csv(out_underrep, index=False)
    log(f"Under-rep CSV -> {out_underrep}")

    out_txt = OUTDIR / "step8_summary_ethnicity_geneformer.txt"
    lines = [
        "STEP 8 -- ETHNICITY-CONDITIONED DISEASE PREDICTION (CRC ETHNICITY, Geneformer)",
        "=" * 70, "",
        f"Most under-represented group: {UNDERREP_GROUP}",
        "Metrics: accuracy, balanced_accuracy, macro_f1 on 20% held-out test set",
        "Delta = metric(strategy) - metric(Proportional_2497 baseline)",
        "",
        "=" * 70,
        "FULL PER-ETHNICITY RESULTS",
        "=" * 70,
        per_eth_df[["dataset", "ethnicity", "accuracy", "balanced_accuracy",
                    "macro_f1", "delta_acc_vs_prop", "n_test_cells"]].to_string(index=False),
        "",
        "=" * 70,
        "WORST-ETHNICITY PER DATASET",
        "=" * 70,
        worst_eth_df.to_string(index=False),
        "",
        "=" * 70,
        f"UNDER-REPRESENTED GROUP SUMMARY (group = {UNDERREP_GROUP})",
        "=" * 70,
        underrep_df.to_string(index=False),
    ]
    out_txt.write_text("\n".join(lines))
    log(f"Report -> {out_txt}")

    log("\n" + "=" * 70)
    log("WORST-ETHNICITY SUMMARY")
    log("=" * 70)
    print(worst_eth_df.to_string(index=False))

    log("\n" + "=" * 70)
    log(f"UNDER-REPRESENTED GROUP ({UNDERREP_GROUP}) SUMMARY")
    log("=" * 70)
    print(underrep_df.to_string(index=False))

    log("\nSTEP 8 COMPLETE")


if __name__ == "__main__":
    main()
