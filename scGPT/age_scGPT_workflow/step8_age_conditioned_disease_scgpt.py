#!/usr/bin/env python3
"""
STEP 8 — Sex-Conditioned Disease Prediction (AGE, scGPT)

Changes vs ethnicity scGPT version:
  [SEX 1] BASE/OUTDIR    -> age_scGPT_workflow
  [SEX 2] GROUP_KEY      -> "age_bin_10yr"
  [SEX 3] DATASETS       -> sex filenames
  [SEX 4] UNDERREP_GROUP -> "female"
  [SEX 5] Reference      -> Proportional_2495
"""

import pathlib, time, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

BASE = pathlib.Path("/data/scGPT/age_scGPT_workflow")
LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step8_age_conditioned_disease_scgpt"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Age_Pilot"
FILES = {
    "Proportional_2495":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2495_labeled_scgpt.h5ad",
    "BalancedAugmented_1262Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1262Each_labeled_scgpt.h5ad",
    "BalancedUpsampled_1262Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_labeled_scgpt.h5ad",
    "Downsampled_25Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_25Each_labeled_scgpt.h5ad",
}

EMB_KEY        = "X_scGPT"
DISEASE_COL    = "disease"
AGE_BIN_KEY        = "age_bin_10yr"
SOURCE_COL     = "source"
UNDERREP_GROUP = "20_29"
MIN_SAMPLES    = 2
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def to_binary_disease(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_dataset(label, path):
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique: ad.obs_names_make_unique()
    ad.obs[AGE_BIN_KEY]          = ad.obs[AGE_BIN_KEY].astype(str).str.strip().str.lower()
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)
    emb      = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    bad_mask = np.isnan(emb).any(axis=1) | (np.abs(emb).sum(axis=1) == 0)
    if bad_mask.any(): ad = ad[~bad_mask].copy()
    log(f"  {label}: {ad.n_obs:,} cells | {ad.obs[AGE_BIN_KEY].value_counts().to_dict()}")
    return ad

def train_and_evaluate(ad, dataset_name, seed=RANDOM_STATE):
    SMALL_GROUP_THRESHOLD = 30
    MIN_TEST_FROM_SMALL   = 3

    valid = pd.notna(ad.obs["disease_binary"]) & pd.notna(ad.obs[AGE_BIN_KEY])
    if valid.sum() < 40: return pd.DataFrame()

    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[valid.values]
    y = ad.obs.loc[valid, "disease_binary"].values
    g = ad.obs.loc[valid, AGE_BIN_KEY].values

    if len(np.unique(y)) < 2: return pd.DataFrame()

    counts   = pd.Series(y).value_counts()
    stratify = y if (counts >= 2).all() else None
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=TEST_SIZE, stratify=stratify, random_state=seed)

    rng = np.random.default_rng(seed)
    all_groups     = set(np.unique(g))
    small_underrep = {grp for grp in all_groups
                      if (g == grp).sum() < SMALL_GROUP_THRESHOLD
                      and (g_te == grp).sum() < MIN_SAMPLES}
    to_boost = (all_groups - set(np.unique(g_te))) | small_underrep

    for grp in to_boost:
        tr_idx = np.where(g_tr == grp)[0]
        if len(tr_idx) == 0: continue
        n_move = min(MIN_TEST_FROM_SMALL, len(tr_idx))
        move   = rng.choice(tr_idx, size=n_move, replace=False)
        keep   = np.setdiff1d(np.arange(len(g_tr)), move)
        X_te = np.vstack([X_te, X_tr[move]]); y_te = np.concatenate([y_te, y_tr[move]]); g_te = np.concatenate([g_te, g_tr[move]])
        X_tr = X_tr[keep]; y_tr = y_tr[keep]; g_tr = g_tr[keep]

    clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    rows = []
    for grp in np.unique(g_te):
        m = g_te == grp
        if m.sum() < MIN_SAMPLES: continue
        rows.append({
            "dataset": dataset_name, "age_bin_10yr": grp,
            "accuracy":          round(float(accuracy_score(y_te[m], preds[m])),          4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_te[m], preds[m])), 4),
            "macro_f1":          round(float(f1_score(y_te[m], preds[m], average="macro")), 4),
            "n_test_cells":      int(m.sum()),
            "is_underrep":       grp == UNDERREP_GROUP,
            "test_boosted":      grp in to_boost,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        worst = df.loc[df["accuracy"].idxmin()]
        log(f"  Worst: {worst['age_bin_10yr']} | acc={worst['accuracy']:.4f} | f1={worst['macro_f1']:.4f}")
    return df


def main():
    log("="*70)
    log("STEP 8 -- Sex-Conditioned Disease Prediction (AGE, scGPT)")
    log("="*70)

    all_dfs = []
    for label, path in FILES.items():
        if not path.exists():
            log(f"  Skipping {label}"); continue
        log(f"\n>> Evaluating {label}")
        ad = load_dataset(label, path)
        df = train_and_evaluate(ad, label)
        if not df.empty: all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No results produced.")

    per_age_df = pd.concat(all_dfs, ignore_index=True)

    ref = "Proportional_2495"
    prop_rows = per_age_df[per_age_df["dataset"] == ref][["age_bin_10yr", "accuracy", "macro_f1"]]
    prop_rows = prop_rows.rename(columns={"accuracy": "prop_acc", "macro_f1": "prop_f1"})
    per_age_df = per_age_df.merge(prop_rows, on="age_bin_10yr", how="left")
    per_age_df["delta_acc_vs_prop"] = (per_age_df["accuracy"] - per_age_df["prop_acc"]).round(4)
    per_age_df["delta_f1_vs_prop"]  = (per_age_df["macro_f1"] - per_age_df["prop_f1"]).round(4)

    worst_age_df = (
        per_age_df.loc[per_age_df.groupby("dataset")["accuracy"].idxmin()]
        [["dataset", "age_bin_10yr", "accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .rename(columns={"age_bin_10yr": "worst_age", "accuracy": "worst_acc", "macro_f1": "worst_f1"})
        .reset_index(drop=True)
    )

    underrep_df = (
        per_age_df[per_age_df["is_underrep"]]
        .groupby("dataset")[["accuracy", "balanced_accuracy", "macro_f1", "delta_acc_vs_prop"]]
        .mean().reset_index()
    )

    per_age_df.to_csv(OUTDIR / "step8_per_age_disease_prediction_scgpt.csv",   index=False)
    worst_age_df.to_csv(OUTDIR / "step8_worst_age_summary_scgpt.csv",          index=False)
    underrep_df.to_csv(OUTDIR / "step8_underrepresented_age_summary_scgpt.csv", index=False)

    lines = [
        "STEP 8 -- AGE-CONDITIONED DISEASE PREDICTION (AGE, scGPT)", "="*70, "",
        f"Most under-represented group: {UNDERREP_GROUP}",
        "Delta = metric(strategy) - metric(Proportional_2495 baseline)",
        "", "="*70, "FULL PER-AGE RESULTS", "="*70,
        per_age_df[["dataset","age_bin_10yr","accuracy","balanced_accuracy","macro_f1","delta_acc_vs_prop","n_test_cells"]].to_string(index=False),
        "", "="*70, "WORST-AGE PER DATASET", "="*70,
        worst_age_df.to_string(index=False),
        "", "="*70, f"UNDER-REPRESENTED GROUP ({UNDERREP_GROUP})", "="*70,
        underrep_df.to_string(index=False),
    ]
    (OUTDIR / "step8_summary_scgpt_age.txt").write_text("\n".join(lines))

    print("\n" + "="*70 + "\nWORST-AGE SUMMARY")
    print(worst_age_df.to_string(index=False))
    print("\n" + "="*70 + f"\nUNDER-REPRESENTED GROUP ({UNDERREP_GROUP})")
    print(underrep_df.to_string(index=False))
    log("STEP 8 COMPLETE")

if __name__ == "__main__":
    main()
