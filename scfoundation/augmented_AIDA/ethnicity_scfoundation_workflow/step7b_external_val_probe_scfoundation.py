#!/usr/bin/env python3
"""STEP 7b -- Cell Type Classification (External Validation, scFoundation)

Trains LogReg on each augmented dataset, evaluates per-ethnicity macro F1
on the held-out 12,500-cell validation set. This is the proper generalization
metric, unlike step7's training-accuracy probe.
"""
import pathlib, time, warnings
import numpy as np, pandas as pd, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = pathlib.Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow")
OUTDIR = BASE / "step7b_external_val_scfoundation"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "AIDA_Ethnicity_Pilot"
FILES = {
    "Proportional_2500":          BASE / f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_scfoundation.h5ad",
    "BalancedAugmented_779Each":  BASE / f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_scfoundation.h5ad",
    "BalancedUpsampled_779Each":  BASE / f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_scfoundation.h5ad",
    "Downsampled_92Each":         BASE / f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_scfoundation.h5ad",
}
VAL_FILE = BASE / "AIDA_Ethnicity_External_Validation_12500_scfoundation.h5ad"

EMB_KEY = "X_scfoundation"
ETH_KEY = "self_reported_ethnicity"
CELL_KEY = "cell_type"
SEED = 42

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def canon(s): return s.astype(str).str.strip().str.lower()

def load_emb_h5ad(path, label):
    ad = sc.read_h5ad(path)
    ad.obs[ETH_KEY] = canon(ad.obs[ETH_KEY])
    ad.obs[CELL_KEY] = ad.obs[CELL_KEY].astype(str).str.strip()
    emb = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    bad = np.isnan(emb).any(axis=1) | (np.abs(emb).sum(axis=1) == 0)
    if bad.any():
        ad = ad[~bad].copy()
        log(f"  {label}: dropped {bad.sum()} degenerate cells")
    return ad

def evaluate(train_ad, val_ad, label):
    X_tr = np.array(train_ad.obsm[EMB_KEY], dtype=np.float32)
    y_tr = train_ad.obs[CELL_KEY].values
    X_val = np.array(val_ad.obsm[EMB_KEY], dtype=np.float32)
    y_val = val_ad.obs[CELL_KEY].values
    g_val = val_ad.obs[ETH_KEY].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=3000, solver="saga",
                             class_weight="balanced",
                             n_jobs=-1, random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_val_s)

    # Per-ethnicity scores on validation
    rows = []
    for eth in sorted(np.unique(g_val)):
        mask = g_val == eth
        if mask.sum() < 10:
            continue
        yt = y_val[mask]
        yp = preds[mask]
        if len(np.unique(yt)) < 2:
            continue
        rows.append({
            "dataset": label,
            "ethnicity": eth,
            "n_val_cells": int(mask.sum()),
            "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
        })
    return rows, clf

def main():
    log("="*70)
    log("STEP 7b -- External validation probe (scFoundation)")
    log("="*70)

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation embedding missing: {VAL_FILE}")

    val_ad = load_emb_h5ad(VAL_FILE, "validation")
    log(f"Validation: {val_ad.n_obs:,} cells | {val_ad.obs[ETH_KEY].value_counts().to_dict()}")
    log(f"Validation cell types: {val_ad.obs[CELL_KEY].nunique()}")

    all_rows = []
    for label, path in FILES.items():
        if not path.exists():
            log(f"Skipping {label}: file not found")
            continue
        log(f"\n>> {label}")
        train_ad = load_emb_h5ad(path, label)

        # Filter validation cell types to those present in train
        train_cts = set(train_ad.obs[CELL_KEY].unique())
        val_cts = set(val_ad.obs[CELL_KEY].unique())
        missing = val_cts - train_cts
        if missing:
            log(f"  {len(missing)} val cell types not in train (dropped from val eval): {sorted(missing)[:5]}...")
            val_ad_eval = val_ad[val_ad.obs[CELL_KEY].isin(train_cts)].copy()
        else:
            val_ad_eval = val_ad

        log(f"  Train: {train_ad.n_obs:,} cells, {train_ad.obs[CELL_KEY].nunique()} CTs")
        log(f"  Val (filtered): {val_ad_eval.n_obs:,} cells, {val_ad_eval.obs[CELL_KEY].nunique()} CTs")

        rows, _ = evaluate(train_ad, val_ad_eval, label)
        for r in rows:
            log(f"  {r['ethnicity']}: macro_f1={r['macro_f1']}, acc={r['accuracy']} (n={r['n_val_cells']})")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_out = OUTDIR / "step7b_external_val_per_ethnicity.csv"
    df.to_csv(csv_out, index=False)
    log(f"\nWrote: {csv_out}")

    # Pivot for readability
    if not df.empty:
        pivot = df.pivot_table(index="ethnicity", columns="dataset", values="macro_f1")
        # delta vs Proportional baseline
        if "Proportional_2500" in pivot.columns:
            for ds in [c for c in pivot.columns if c != "Proportional_2500"]:
                pivot[f"delta_{ds}_vs_Prop"] = pivot[ds] - pivot["Proportional_2500"]
        pivot.to_csv(OUTDIR / "step7b_external_val_pivot.csv")
        log("\n" + pivot.to_string())

    log("\nSTEP 7b COMPLETE")

if __name__ == "__main__":
    main()
