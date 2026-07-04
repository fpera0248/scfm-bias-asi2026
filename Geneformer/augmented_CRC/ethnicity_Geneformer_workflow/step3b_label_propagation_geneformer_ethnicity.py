#!/usr/bin/env python3
"""
STEP 3b — kNN Disease Label Propagation (CRC ETHNICITY, Geneformer V2-316M)

Changes vs step3b scFoundation ethnicity:
  [GF 1]  BASE path -> ethnicity_Geneformer_workflow
  [GF 2]  EMB_KEY   -> X_geneformer
  [GF 3]  DATASETS  -> _geneformer.h5ad filenames, 2143Each / 48Each / 2497
  [GF 4]  make_output_path -> _labeled_geneformer.h5ad suffix

All kNN propagation logic preserved verbatim.
FIX patches 1-4 preserved verbatim.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# PATHS  [GF 1]
# ============================================================

BASE   = Path(
    "/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow"
)
OUTDIR  = BASE / "step3b_labeled"
OUTDIR.mkdir(exist_ok=True)

LOGFILE = OUTDIR / "step3b_label_propagation_log.txt"

OUTPUT_BASE = "CRC_Eth_Pilot"

DATASETS = {                                                     # [GF 3]
    "BalancedAugmented_1880Each": BASE / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETH_geneformer.h5ad",
    "Proportional_2497":          BASE / f"{OUTPUT_BASE}_Proportional_2497_ETH_geneformer.h5ad",
    "BalancedUpsampled_1880Each": BASE / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETH_geneformer.h5ad",
    "Downsampled_48Each":         BASE / f"{OUTPUT_BASE}_Downsampled_48Each_ETH_geneformer.h5ad",
}

# ============================================================
# CONFIG  [GF 2]
# ============================================================

EMB_KEY       = "X_geneformer"
DISEASE_COL   = "disease"
SOURCE_COL    = "source"
ETHNICITY_COL = "self_reported_ethnicity"

KNN_K         = 5
KNN_METRIC    = "euclidean"
RANDOM_STATE  = 42

# ============================================================
# LOGGING
# ============================================================

log_fh = open(LOGFILE, "w")

def log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    print(line, file=log_fh, flush=True)

# ============================================================
# HELPERS
# ============================================================

def has_synthetic(adata) -> bool:
    if SOURCE_COL not in adata.obs.columns:
        return False
    return (adata.obs[SOURCE_COL] == "synthetic").any()


def validate_embedding(adata, fname):
    if EMB_KEY not in adata.obsm:
        raise RuntimeError(
            f"'{EMB_KEY}' not found in {fname}.\n"
            f"Available obsm keys: {list(adata.obsm.keys())}\n"
            "Run step2a first."
        )


def make_output_path(dname: str) -> Path:
    return OUTDIR / f"{OUTPUT_BASE}_{dname}_labeled_geneformer.h5ad"   # [GF 4]


def load_embedding_to_ram(adata) -> np.ndarray:
    """[FIX 3] Force obsm into contiguous RAM array."""
    emb = adata.obsm[EMB_KEY]
    if not isinstance(emb, np.ndarray) or not emb.flags["C_CONTIGUOUS"]:
        log(f"  Materialising {EMB_KEY} into RAM ({emb.shape[0]:,} x {emb.shape[1]} float32) ...")
        t0  = time.time()
        emb = np.array(emb, dtype=np.float32, order="C")
        log(f"  Done in {time.time()-t0:.1f}s  |  {emb.nbytes/1024**2:.1f} MB")
    return emb

# ============================================================
# LABEL PROPAGATION
# ============================================================

def propagate_labels(adata, dname: str) -> sc.AnnData:
    emb = load_embedding_to_ram(adata)   # [FIX 3]

    real_mask = (adata.obs[SOURCE_COL] == "real").to_numpy()
    syn_mask  = (adata.obs[SOURCE_COL] == "synthetic").to_numpy()

    log(f"  Real cells:      {int(real_mask.sum()):,}")
    log(f"  Synthetic cells: {int(syn_mask.sum()):,}")

    if ETHNICITY_COL in adata.obs.columns:
        log(f"  Ethnicity breakdown:")
        for grp, n in adata.obs[ETHNICITY_COL].value_counts().items():
            log(f"    {grp}: {n}")

    disease_vals   = adata.obs[DISEASE_COL].to_numpy()
    real_has_label = real_mask & (~pd.isna(disease_vals))
    n_labeled_real = int(real_has_label.sum())

    if n_labeled_real == 0:
        raise RuntimeError(f"No real cells with disease labels in {dname}.")
    if n_labeled_real < int(real_mask.sum()):
        log(f"  WARNING: {int(real_mask.sum()) - n_labeled_real} real cells missing disease label -- excluded.")

    X_real_raw = emb[real_has_label]
    y_real_raw = disease_vals[real_has_label].astype(str)

    nan_real = np.isnan(X_real_raw).any(axis=1)
    if nan_real.any():
        log(f"  WARNING: {nan_real.sum()} real cells have NaN embeddings -- dropped.")
    X_real = X_real_raw[~nan_real]
    y_real = y_real_raw[~nan_real]

    if len(X_real) == 0:
        raise RuntimeError("No real cells with valid embeddings remain.")

    log(f"  Training kNN (K={KNN_K}, metric={KNN_METRIC}) on {len(X_real):,} real cells ...")
    log(f"  Disease label distribution (training):")
    for label, count in pd.Series(y_real).value_counts().items():
        log(f"    {label}: {count}")

    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC, n_jobs=-1)
    knn.fit(X_real, y_real)

    X_syn     = emb[syn_mask].copy()
    nan_syn   = np.isnan(X_syn).any(axis=1)
    if nan_syn.any():
        log(f"  WARNING: {nan_syn.sum()} synthetic cells have NaN embeddings -- imputing.")
        col_means          = np.nanmean(X_real, axis=0)
        rows, cols         = np.where(np.isnan(X_syn))
        X_syn[rows, cols]  = col_means[cols]

    y_syn_pred  = knn.predict(X_syn)
    y_syn_proba = knn.predict_proba(X_syn)
    confidence  = y_syn_proba.max(axis=1)

    log(f"  Mean confidence: {confidence.mean():.3f}  |  Unanimous ({KNN_K}/{KNN_K}): {(confidence==1.0).mean():.3f}")
    log(f"  Predicted disease distribution (synthetic):")
    for label, count in pd.Series(y_syn_pred).value_counts().items():
        log(f"    {label}: {count}")

    # [FIX 4] Write back via .loc[]
    syn_bool_index  = adata.obs.index[syn_mask]
    disease_col_str = adata.obs[DISEASE_COL].astype(str)
    disease_col_str.loc[syn_bool_index] = y_syn_pred
    adata.obs[DISEASE_COL] = disease_col_str

    conf_col = pd.Series(np.nan, index=adata.obs.index, dtype=float)
    conf_col.loc[syn_bool_index] = confidence
    adata.obs["knn_label_confidence"] = conf_col

    still_missing = adata.obs[DISEASE_COL].isna().sum()
    if still_missing > 0:
        log(f"  WARNING: {still_missing} cells still missing disease label.")
    else:
        log("  All cells now have disease labels.")

    log("  Final disease distribution (all cells):")
    for label, count in adata.obs[DISEASE_COL].value_counts().items():
        log(f"    {label}: {count}")

    log("  Final distribution by source x disease:")
    cross = adata.obs.groupby([SOURCE_COL, DISEASE_COL]).size().reset_index(name="count")
    for _, row in cross.iterrows():
        log(f"    [{row[SOURCE_COL]}] {row[DISEASE_COL]}: {row['count']}")

    return adata

# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    log("=" * 70)
    log("STEP 3b -- kNN Disease Label Propagation (CRC ETHNICITY, Geneformer)")
    log("=" * 70)
    log(f"KNN_K       = {KNN_K}")
    log(f"KNN_METRIC  = {KNN_METRIC}")
    log(f"EMB_KEY     = {EMB_KEY}")
    log(f"Output dir  = {OUTDIR}")

    summary_rows = []

    for dname, path in DATASETS.items():
        log(f"\n{'='*70}")
        log(f"Dataset: {dname}")

        if not path.exists():
            log(f"  File not found: {path}")
            continue

        log(f"  Loading: {path.name}")
        adata = sc.read_h5ad(path)
        log(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")
        log(f"  obsm keys: {list(adata.obsm.keys())}")

        validate_embedding(adata, path.name)

        out_path = make_output_path(dname)

        if not has_synthetic(adata):
            log("  No synthetic cells -- passing through unchanged.")
            log(f"  Writing -> {out_path.name}")
            adata.write_h5ad(out_path)
            summary_rows.append(dict(
                dataset=dname, n_total=adata.n_obs, n_real=adata.n_obs,
                n_synthetic=0, labels_assigned=0, action="passthrough",
            ))
            continue

        n_syn_before   = int((adata.obs[SOURCE_COL] == "synthetic").sum())
        missing_before = int(
            ((adata.obs[SOURCE_COL] == "synthetic") & adata.obs[DISEASE_COL].isna()).sum()
        )
        log(f"  Synthetic cells missing disease label before propagation: {missing_before:,}")

        t0    = time.time()
        adata = propagate_labels(adata, dname)
        elapsed = time.time() - t0

        missing_after = int(
            ((adata.obs[SOURCE_COL] == "synthetic") & adata.obs[DISEASE_COL].isna()).sum()
        )
        log(f"  Completed in {elapsed:.1f}s  |  Labels assigned: {missing_before - missing_after}")
        log(f"  Writing -> {out_path.name}")
        adata.write_h5ad(out_path)

        summary_rows.append(dict(
            dataset=dname,
            n_total=adata.n_obs,
            n_real=int((adata.obs[SOURCE_COL] == "real").sum()),
            n_synthetic=n_syn_before,
            labels_assigned=missing_before - missing_after,
            action="knn_propagation",
        ))

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "step3b_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log(f"\n{'='*70}")
    log("STEP 3b COMPLETE (CRC ETHNICITY, Geneformer)")
    log(f"  Outputs -> {OUTDIR}")
    log(f"  Summary -> {summary_path.name}")
    print("\n" + summary_df.to_string(index=False))

    log_fh.close()


if __name__ == "__main__":
    main()