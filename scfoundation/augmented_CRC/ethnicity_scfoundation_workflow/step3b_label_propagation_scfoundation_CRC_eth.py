#!/usr/bin/env python3
"""STEP 3b — kNN Disease Label Propagation (CRC ETHNICITY, scFoundation)"""

import time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
from sklearn.neighbors import KNeighborsClassifier

BASE    = Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow")
OUTDIR  = BASE / "step3b_labeled"
OUTDIR.mkdir(exist_ok=True)
LOGFILE = OUTDIR / "step3b_label_propagation_log.txt"

OUTPUT_BASE = "CRC_Eth_Pilot"
DATASETS = {
    "BalancedAugmented_1880Each": BASE / "step2a_embeddings" / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETH_scfoundation.h5ad",
    "Proportional_2497":          BASE / "step2a_embeddings" / f"{OUTPUT_BASE}_Proportional_2497_ETH_scfoundation.h5ad",
    "BalancedUpsampled_1880Each": BASE / "step2a_embeddings" / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETH_scfoundation.h5ad",
    "Downsampled_48Each":         BASE / "step2a_embeddings" / f"{OUTPUT_BASE}_Downsampled_48Each_ETH_scfoundation.h5ad",
}

EMB_KEY      = "X_scfoundation"
DISEASE_COL  = "disease"
SOURCE_COL   = "source"
ETHNICITY_COL = "self_reported_ethnicity"
KNN_K = 5; KNN_METRIC = "euclidean"

log_fh = open(LOGFILE, "w")
def log(msg):
    ts = time.strftime("%H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True); print(line, file=log_fh, flush=True)

def has_synthetic(adata):
    return SOURCE_COL in adata.obs.columns and (adata.obs[SOURCE_COL] == "synthetic").any()

def make_output_path(dname):
    return OUTDIR / f"{OUTPUT_BASE}_{dname}_labeled_scfoundation.h5ad"

def propagate_labels(adata, dname):
    emb = np.array(adata.obsm[EMB_KEY], dtype=np.float32, order="C")
    real_mask = (adata.obs[SOURCE_COL] == "real").to_numpy()
    syn_mask  = (adata.obs[SOURCE_COL] == "synthetic").to_numpy()
    log(f"  Real: {real_mask.sum():,}  Synthetic: {syn_mask.sum():,}")
    if ETHNICITY_COL in adata.obs.columns:
        for grp, n in adata.obs[ETHNICITY_COL].value_counts().items(): log(f"    {grp}: {n}")
    disease_vals   = adata.obs[DISEASE_COL].to_numpy()
    real_has_label = real_mask & (~pd.isna(disease_vals))
    X_real = emb[real_has_label]; y_real = disease_vals[real_has_label].astype(str)
    nan_real = np.isnan(X_real).any(axis=1); X_real = X_real[~nan_real]; y_real = y_real[~nan_real]
    if len(X_real) == 0: raise RuntimeError("No real cells with valid embeddings.")
    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC, n_jobs=-1)
    knn.fit(X_real, y_real)
    X_syn = emb[syn_mask].copy(); nan_syn = np.isnan(X_syn).any(axis=1)
    if nan_syn.any():
        col_means = np.nanmean(X_real, axis=0); rows, cols = np.where(np.isnan(X_syn))
        X_syn[rows, cols] = col_means[cols]
    y_syn_pred  = knn.predict(X_syn); y_syn_proba = knn.predict_proba(X_syn); confidence = y_syn_proba.max(axis=1)
    log(f"  Mean confidence: {confidence.mean():.3f}")
    for label, count in pd.Series(y_syn_pred).value_counts().items(): log(f"    [synthetic] {label}: {count}")
    syn_idx = adata.obs.index[syn_mask]
    dc = adata.obs[DISEASE_COL].astype(str); dc.loc[syn_idx] = y_syn_pred; adata.obs[DISEASE_COL] = dc
    cf = pd.Series(np.nan, index=adata.obs.index, dtype=float); cf.loc[syn_idx] = confidence
    adata.obs["knn_label_confidence"] = cf
    return adata

def main():
    warnings.filterwarnings("ignore")
    log("="*70); log("STEP 3b -- kNN Disease Label Propagation (CRC ETHNICITY, scFoundation)"); log("="*70)
    summary_rows = []
    for dname, path in DATASETS.items():
        log(f"\n{'='*70}\nDataset: {dname}")
        if not path.exists(): log(f"  File not found: {path}"); continue
        adata = sc.read_h5ad(path); out_path = make_output_path(dname)
        if not has_synthetic(adata):
            log("  No synthetic cells -- passthrough."); adata.write_h5ad(out_path)
            summary_rows.append(dict(dataset=dname, n_total=adata.n_obs, n_real=adata.n_obs, n_synthetic=0, labels_assigned=0, action="passthrough"))
            continue
        n_syn_before   = int((adata.obs[SOURCE_COL] == "synthetic").sum())
        missing_before = int(((adata.obs[SOURCE_COL] == "synthetic") & adata.obs[DISEASE_COL].isna()).sum())
        t0 = time.time(); adata = propagate_labels(adata, dname); elapsed = time.time() - t0
        missing_after = int(((adata.obs[SOURCE_COL] == "synthetic") & adata.obs[DISEASE_COL].isna()).sum())
        log(f"  Done in {elapsed:.1f}s  |  Labels assigned: {missing_before - missing_after}")
        adata.write_h5ad(out_path)
        summary_rows.append(dict(dataset=dname, n_total=adata.n_obs,
            n_real=int((adata.obs[SOURCE_COL] == "real").sum()),
            n_synthetic=n_syn_before, labels_assigned=missing_before - missing_after, action="knn_propagation"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "step3b_summary.csv", index=False)
    log(f"\n{'='*70}\nSTEP 3b COMPLETE (CRC ETHNICITY, scFoundation)")
    print("\n" + summary_df.to_string(index=False)); log_fh.close()

if __name__ == "__main__":
    main()
