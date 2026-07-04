#!/usr/bin/env python3
"""
STEP 7b -- Cell Type Classification (External Validation, scFoundation, AGE)
Trains LogReg on each augmented dataset, evaluates per-age-bin macro F1
on the held-out 10,000-cell validation set.
"""
import pathlib, time, warnings
import numpy as np, pandas as pd, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = pathlib.Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow")
OUTDIR = BASE / "step7b_external_val_scfoundation"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "AIDA_Age_Pilot"

EMB_DIR = BASE / "step2a_embeddings"

FILES = {
    "Proportional_2498":         EMB_DIR / f"{OUTPUT_BASE}_Proportional_2498_AGE_scfoundation.h5ad",
    "BalancedAugmented_747Each": EMB_DIR / f"{OUTPUT_BASE}_BalancedAugmented_747Each_AGE_scfoundation.h5ad",
    "BalancedUpsampled_747Each": EMB_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_747Each_AGE_scfoundation.h5ad",
    "Downsampled_230Each":       EMB_DIR / f"{OUTPUT_BASE}_Downsampled_230Each_AGE_scfoundation.h5ad",
}

VAL_FILE = EMB_DIR / "AIDA_Age_External_Validation_10000_scfoundation.h5ad"
EMB_KEY = "X_scfoundation"
GROUP_KEY = "age_bin_10yr"
CELL_KEY = "cell_type"
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("STEP 7b -- External validation probe (scFoundation, AGE)")
    log("=" * 70)

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file missing: {VAL_FILE}")
    val = sc.read_h5ad(VAL_FILE)
    val_grp = val.obs[GROUP_KEY].astype(str).values
    log(f"Validation: {val.n_obs:,} cells | {dict(pd.Series(val_grp).value_counts())}")
    log(f"Validation cell types: {val.obs[CELL_KEY].nunique()}")

    rows = []
    for name, fpath in FILES.items():
        log(f"\n>> {name}")
        if not fpath.exists():
            log(f"  MISSING: {fpath.name} -- skipping")
            continue

        train = sc.read_h5ad(fpath)
        train_cts = set(train.obs[CELL_KEY].astype(str).unique())
        val_cts = set(val.obs[CELL_KEY].astype(str).unique())
        missing = val_cts - train_cts
        if missing:
            log(f"  {len(missing)} val cell types not in train (dropped from val eval): {sorted(missing)[:5]}...")

        val_mask = val.obs[CELL_KEY].astype(str).isin(train_cts).values
        X_train = np.asarray(train.obsm[EMB_KEY])
        y_train = train.obs[CELL_KEY].astype(str).values
        X_val = np.asarray(val.obsm[EMB_KEY])[val_mask]
        y_val = val.obs[CELL_KEY].astype(str).values[val_mask]
        val_grp_f = val_grp[val_mask]

        log(f"  Train: {X_train.shape[0]:,} cells, {len(train_cts)} CTs")
        log(f"  Val (filtered): {X_val.shape[0]:,} cells, {len(set(y_val))} CTs")

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(max_iter=5000, n_jobs=-1, solver='lbfgs', random_state=SEED)
        clf.fit(scaler.transform(X_train), y_train)
        y_pred = clf.predict(scaler.transform(X_val))

        for grp in sorted(set(val_grp_f)):
            m = val_grp_f == grp
            if m.sum() < 10:
                continue
            f1 = f1_score(y_val[m], y_pred[m], average='macro', zero_division=0)
            acc = accuracy_score(y_val[m], y_pred[m])
            log(f"  {grp}: macro_f1={f1:.4g}, acc={acc:.4g} (n={m.sum()})")
            rows.append({'dataset': name, 'age_bin': grp, 'macro_f1': f1, 'accuracy': acc, 'n_val': int(m.sum())})

    if not rows:
        raise RuntimeError("No results produced -- check input files exist.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "step7b_external_val_per_age.csv", index=False)
    pivot = df.pivot(index='age_bin', columns='dataset', values='macro_f1')
    if 'Proportional_2498' in pivot.columns:
        for col in list(pivot.columns):
            if col != 'Proportional_2498':
                pivot[f'delta_{col}_vs_Prop'] = pivot[col] - pivot['Proportional_2498']
    pivot.to_csv(OUTDIR / "step7b_external_val_per_age_pivot.csv")
    log(f"\n{pivot.to_string()}")
    log("\nSTEP 7b COMPLETE")


if __name__ == "__main__":
    main()
