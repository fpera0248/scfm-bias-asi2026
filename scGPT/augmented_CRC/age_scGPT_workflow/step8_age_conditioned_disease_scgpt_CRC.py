#!/usr/bin/env python3
"""
STEP 8 — Age-Conditioned Disease Prediction (CRC AGE, scGPT)

Changes vs sex version:
  [AGE 1] BASE/OUTDIR    -> age_scGPT_workflow
  [AGE 2] GROUP_KEY      -> "age_bin_10yr"
  [AGE 3] FILES          -> age filenames (1262Each, 25Each, 2495)
  [AGE 4] UNDERREP_GROUP -> "10_19"
  [AGE 5] Reference baseline -> Proportional_2498
  [AGE 6] Output files   -> *_age_geneformer suffix
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
    "/data/scGPT/augmented_CRC/age_scGPT_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step8_age_conditioned_disease_scgpt"   # [AGE 6]
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "CRC_Age_Pilot"

FILES = {                                                          # [AGE 3]
    "Proportional_2498":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2498_labeled_scgpt.h5ad",
    "BalancedAugmented_650Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_650Each_labeled_scgpt.h5ad",
    "BalancedUpsampled_650Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_650Each_labeled_scgpt.h5ad",
    "Downsampled_124Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_124Each_labeled_scgpt.h5ad",
}

EMB_KEY        = "X_scGPT"
DISEASE_COL    = "disease"
AGE_KEY        = "age_bin_10yr"                                   # [AGE 2]
SOURCE_COL     = "source"
UNDERREP_GROUP = "30_39"                                          # [AGE 4]
MIN_SAMPLES    = 2
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group", "development_stage"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_age_col(obs):                                          # [AGE 2]
    for c in AGE_COL_CANDIDATES:
        if c in obs.columns: return c
    raise RuntimeError(f"No age column found. Available: {list(obs.columns)}")


def canonicalize_age(series):
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

    age_col = detect_age_col(ad.obs)
    ad.obs[AGE_KEY]          = canonicalize_age(ad.obs[age_col])
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)

    emb       = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask  = np.isnan(emb).any(axis=1)
    zero_mask = (np.abs(emb).sum(axis=1) == 0)
    bad_mask  = nan_mask | zero_mask
    if bad_mask.any():
        log(f"  Dropping {bad_mask.sum()} cells (NaN or zero embeddings)")
        ad = ad[~bad_mask].copy()

    log(f"  {ad.n_obs:,} cells | Age dist: "
        f"{ad.obs[AGE_KEY].value_counts().sort_index().to_dict()}")
    return ad


def train_and_evaluate(ad, dataset_name, seed=RANDOM_STATE):
    SMALL_GROUP_THRESHOLD = 30
    MIN_TEST_FROM_SMALL   = 3

    valid = pd.notna(ad.obs["disease_binary"]) & pd.notna(ad.obs[AGE_KEY])
    if valid.sum() < 40:
        log(f"  Too few valid cells ({valid.sum()}) -- skipping {dataset_name}")
        return pd.DataFrame()

    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[valid.values]
    y = ad.obs.loc[valid, "disease_binary"].values
    g = ad.obs.loc[valid, AGE_KEY].values

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
    small_underrep = {grp for grp in all_groups
                      if (g == grp).sum() < SMALL_GROUP_THRESHOLD
                      and (g_te == grp).sum() < MIN_SAMPLES}
    to_boost = missing | small_underrep

    if to_boost:
        log(f"  Boosting test representation for: {sorted(to_boost)}")
    for grp in to_boost:
        tr_idx = np.where(g_tr == grp)[0]
        if len(tr_idx) == 0:
            log(f"  WARNING: '{grp}' has no cells in train either -- skipping boost.")
            continue
        n_move = min(MIN_TEST_FROM_SMALL, len(tr_idx))
        move   = rng.choice(tr_idx, size=n_move, replace=False)
        keep   = np.setdiff1d(np.arange(len(g_tr)), move)
        X_te = np.vstack([X_te, X_tr[move]])
        y_te = np.concatenate([y_te, y_tr[move]])
        g_te = np.concatenate([g_te, g_tr[move]])
        X_tr = X_tr[keep]; y_tr = y_tr[keep]; g_tr = g_tr[keep]
        log(f"  Moved {n_move} '{grp}' cells: train->test.")

    clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    rows = []
    for grp in np.unique(g_te):
        m = g_te == grp
        if m.sum() < MIN_SAMPLES:
            log(f"  Skipping age_bin='{grp}' -- only {m.sum()} test cells")
            continue
        acc  = accuracy_score(y_te[m], preds[m])
        bacc = balanced_accuracy_score(y_te[m], preds[m])
        f1   = f1_score(y_te[m], preds[m], average="macro")
        rows.append({
            "dataset":           dataset_name,
            "age_bin":           grp,                             # [AGE 2]
            "accuracy":          round(float(acc),  4),
            "balanced_accuracy": round(float(bacc), 4),
            "macro_f1":          round(float(f1),   4),
            "n_test_cells":      int(m.sum()),
            "is_underrep":       grp == UNDERREP_GROUP,
            "test_boosted":      grp in to_boost,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        worst_row = df.loc[df["accuracy"].idxmin()]
        log(f"  Worst age bin: {worst_row['age_bin']} | "
            f"acc={worst_row['accuracy']:.4f} | f1={worst_row['macro_f1']:.4f}")
    return df


def main():
    log("=" * 70)
    log("STEP 8 -- Age-Conditioned Disease Prediction (CRC AGE, scGPT)")  # [AGE]
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

    per_age_df = pd.concat(all_dfs, ignore_index=True)

    ref = "Proportional_2498"                                     # [AGE 5]
    prop_rows = per_age_df[per_age_df["dataset"] == ref][["age_bin", "accuracy", "macro_f1"]]
    prop_rows = prop_rows.rename(columns={"accuracy": "prop_acc", "macro_f1": "prop_f1"})
    per_age_df = per_age_df.merge(prop_rows, on="age_bin", how="left")
    per_age_df["delta_acc_vs_prop"] = (per_age_df["accuracy"] - per_age_df["prop_acc"]).round(4)
    per_age_df["delta_f1_vs_prop"]  = (per_age_df["macro_f1"] - per_age_df["prop_f1"]).round(4)

    worst_age_df = (
        per_age_df.loc[per_age_df.groupby("dataset")["accuracy"].idxmin()]
        [["dataset", "age_bin", "accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .rename(columns={"age_bin": "worst_age_bin", "accuracy": "worst_acc", "macro_f1": "worst_f1"})
        .reset_index(drop=True)
    )

    underrep_df = (
        per_age_df[per_age_df["is_underrep"]]
        .groupby("dataset")[["accuracy", "balanced_accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .mean()
        .reset_index()
    )

    out_per_age = OUTDIR / "step8_per_age_disease_prediction_scgpt.csv"
    per_age_df.to_csv(out_per_age, index=False)
    log(f"\nPer-age CSV -> {out_per_age}")

    out_worst = OUTDIR / "step8_worst_age_bin_summary_scgpt.csv"
    worst_age_df.to_csv(out_worst, index=False)
    log(f"Worst-age CSV -> {out_worst}")

    out_underrep = OUTDIR / "step8_underrepresented_age_summary_scgpt.csv"
    underrep_df.to_csv(out_underrep, index=False)
    log(f"Under-rep CSV -> {out_underrep}")

    out_txt = OUTDIR / "step8_summary_age_scgpt.txt"
    lines = [
        "STEP 8 -- AGE-CONDITIONED DISEASE PREDICTION (CRC AGE, scGPT)",
        "=" * 70, "",
        f"Most under-represented group: {UNDERREP_GROUP}",
        "Metrics: accuracy, balanced_accuracy, macro_f1 on 20% held-out test set",
        "Delta = metric(strategy) - metric(Proportional_2498 baseline)",
        "",
        "=" * 70,
        "FULL PER-AGE-BIN RESULTS",
        "=" * 70,
        per_age_df[["dataset", "age_bin", "accuracy", "balanced_accuracy",
                    "macro_f1", "delta_acc_vs_prop", "n_test_cells"]].to_string(index=False),
        "",
        "=" * 70,
        "WORST-AGE-BIN PER DATASET",
        "=" * 70,
        worst_age_df.to_string(index=False),
        "",
        "=" * 70,
        f"UNDER-REPRESENTED GROUP SUMMARY (group = {UNDERREP_GROUP})",
        "=" * 70,
        underrep_df.to_string(index=False),
    ]
    out_txt.write_text("\n".join(lines))
    log(f"Report -> {out_txt}")

    log("\n" + "=" * 70)
    log("WORST-AGE-BIN SUMMARY")
    log("=" * 70)
    print(worst_age_df.to_string(index=False))

    log("\n" + "=" * 70)
    log(f"UNDER-REPRESENTED GROUP ({UNDERREP_GROUP}) SUMMARY")
    log("=" * 70)
    print(underrep_df.to_string(index=False))

    log("\nSTEP 8 COMPLETE")


if __name__ == "__main__":
    main()
