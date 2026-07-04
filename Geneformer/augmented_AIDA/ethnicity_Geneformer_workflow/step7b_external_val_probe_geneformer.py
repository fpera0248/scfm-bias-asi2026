#!/usr/bin/env python3
"""
Step 7b -- External validation probe (Geneformer)
Train LogReg on each augmented dataset's embeddings, evaluate on held-out validation set.
Report per-ethnicity macro_f1 and accuracy. True generalization F1, not training F1.
"""

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE = Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow")
EMB_DIR = BASE
OUTPUT_BASE = "AIDA_Ethnicity_Pilot"
EMB_KEY = "X_geneformer"
DEMOGRAPHIC = "self_reported_ethnicity"
OUT_DIR = BASE / "step7b_external_val_geneformer"
OUT_DIR.mkdir(exist_ok=True)

VAL_FILE = EMB_DIR / "AIDA_Ethnicity_External_Validation_12500_geneformer.h5ad"

FILES = {
    "Proportional_2500":         EMB_DIR / f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_geneformer.h5ad",
    "BalancedAugmented_779Each": EMB_DIR / f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_geneformer.h5ad",
    "BalancedUpsampled_779Each": EMB_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_geneformer.h5ad",
    "Downsampled_92Each":        EMB_DIR / f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_geneformer.h5ad",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("STEP 7b -- External validation probe (Geneformer)")
    log("=" * 70)

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file missing: {VAL_FILE}")
    val_adata = ad.read_h5ad(VAL_FILE)
    val_eth = val_adata.obs[DEMOGRAPHIC].astype(str).str.lower().values
    log(f"Validation: {val_adata.n_obs:,} cells | {dict(pd.Series(val_eth).value_counts())}")
    log(f"Validation cell types: {val_adata.obs['cell_type'].nunique()}")

    rows = []

    for name, fpath in FILES.items():
        log(f"\n>> {name}")
        if not fpath.exists():
            log(f"  MISSING: {fpath.name} -- skipping")
            continue

        train = ad.read_h5ad(fpath)
        train_cts = set(train.obs['cell_type'].astype(str).unique())
        val_cts = set(val_adata.obs['cell_type'].astype(str).unique())
        missing = val_cts - train_cts
        if missing:
            log(f"  {len(missing)} val cell types not in train (dropped from val eval): {sorted(missing)[:5]}...")

        val_mask = val_adata.obs['cell_type'].astype(str).isin(train_cts).values
        X_train = np.asarray(train.obsm[EMB_KEY])
        y_train = train.obs['cell_type'].astype(str).values
        X_val = np.asarray(val_adata.obsm[EMB_KEY])[val_mask]
        y_val = val_adata.obs['cell_type'].astype(str).values[val_mask]
        val_eth_f = val_eth[val_mask]

        log(f"  Train: {X_train.shape[0]:,} cells, {len(train_cts)} CTs")
        log(f"  Val (filtered): {X_val.shape[0]:,} cells, {len(set(y_val))} CTs")

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=5000, n_jobs=-1, solver='lbfgs')
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_val_s)

        for eth in sorted(set(val_eth_f)):
            m = val_eth_f == eth
            if m.sum() < 10:
                continue
            f1 = f1_score(y_val[m], y_pred[m], average='macro', zero_division=0)
            acc = accuracy_score(y_val[m], y_pred[m])
            log(f"  {eth}: macro_f1={f1:.4g}, acc={acc:.4g} (n={m.sum()})")
            rows.append({
                'dataset': name,
                'ethnicity': eth,
                'macro_f1': f1,
                'accuracy': acc,
                'n_val': int(m.sum()),
            })

    if not rows:
        raise RuntimeError("No results produced -- check input files exist.")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "step7b_external_val_per_ethnicity.csv"
    df.to_csv(out_csv, index=False)
    log(f"\nWrote: {out_csv}")

    pivot = df.pivot(index='ethnicity', columns='dataset', values='macro_f1')
    if 'Proportional_2500' in pivot.columns:
        for col in list(pivot.columns):
            if col != 'Proportional_2500':
                pivot[f'delta_{col}_vs_Prop'] = pivot[col] - pivot['Proportional_2500']
    out_pivot = OUT_DIR / "step7b_external_val_per_ethnicity_pivot.csv"
    pivot.to_csv(out_pivot)
    log(f"\n{pivot.to_string()}")
    log(f"\nWrote: {out_pivot}")
    log("\nSTEP 7b COMPLETE")


if __name__ == "__main__":
    main()
