#!/usr/bin/env python3
"""
STEP 0A — Extract raw counts for ETHNICITY-based augmentation workflow on AIDA.

Reads AIDA Phase 1 v2 PBMC h5ad, cleans demographic columns, splits
into training pool + external validation, writes raw counts in the
format Step 0B expects.

Output:
  AIDA_RawCounts_ETHNICITY.h5ad                       (training pool)
  AIDA_Ethnicity_External_Validation_12500.h5ad       (held-out validation)
"""

import scanpy as sc
import numpy as np
import pandas as pd
import time
import sys

# ============================================================
# CONFIGURATION
# ============================================================
RAW_FILE        = "/oscar/home/fperalta/data/fperalta/scfoundation/aida_phase1_v2.h5ad"
OUTPUT_FILE     = "AIDA_RawCounts_ETHNICITY.h5ad"
VALIDATION_FILE = "AIDA_Ethnicity_External_Validation_12500.h5ad"
GROUP_KEY       = "self_reported_ethnicity"
LOGFILE         = "step0a_ethnicity_log.txt"

VALIDATION_SIZE = 12500
RANDOM_SEED     = 42

REQUIRED_COVARS = {
    "cell_type":               "PBMC cell-type label",
    "self_reported_ethnicity": "Ethnicity of donor",
    "donor_id":                "Unique donor identifier",
    "tissue":                  "Tissue (PBMC)",
    "sex":                     "Sex of donor",
}

OPTIONAL_COVARS = {
    "development_stage":  "Donor age in 'XX-year-old stage' format",
    "Country":            "Country of collection",
    "Annotation_Level1":  "Coarse cell-type annotation",
    "Annotation_Level2":  "Intermediate cell-type annotation",
}

UNKNOWN_VALUES = {
    "unknown", "na", "n/a", "not reported", "", "nan",
    "multiethnic", "na na", "not applicable", "prefer not to say",
}

# ============================================================
# LOGGING
# ============================================================
log_con = open(LOGFILE, "w")

def heartbeat(msg, newline=False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}" + ("\n" if newline else "")
    print(line, file=log_con, flush=True)
    print(line, file=sys.stderr, end="")
    sys.stderr.flush()

heartbeat("\n" + "=" * 70 + "\n", True)
heartbeat("STEP 0A: AIDA ETHNICITY - METADATA EXTRACTION & VALIDATION\n", True)
heartbeat("=" * 70 + "\n", True)

try:
    # ============================================================
    # LOAD AIDA
    # ============================================================
    heartbeat(f"Loading {RAW_FILE} ...", True)
    adata = sc.read_h5ad(RAW_FILE)
    heartbeat(f" Input shape: {adata.shape}", True)
    heartbeat(f" X type: {type(adata.X).__name__}", True)

    # Identify raw count source. CELLxGENE convention puts raw integer
    # counts in adata.raw and normalized values in adata.X, but datasets vary.
    if adata.raw is not None:
        heartbeat(f" adata.raw present with shape {adata.raw.X.shape}", True)
        sample = adata.raw.X[:50, :50]
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        raw_is_int = np.allclose(sample, np.round(sample))
        heartbeat(f" adata.raw integer-valued: {raw_is_int}", True)
        use_raw = raw_is_int
    else:
        use_raw = False
        heartbeat(" adata.raw is None; falling back to adata.X", True)

    if not use_raw:
        sample = adata.X[:50, :50]
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        if not np.allclose(sample, np.round(sample)):
            raise ValueError(
                "Neither adata.raw nor adata.X contains integer counts. "
                "scDesign3 requires raw integer counts."
            )
        heartbeat(" adata.X is integer; using as raw counts", True)

    # ============================================================
    # SANITY CHECKS
    # ============================================================
    if GROUP_KEY not in adata.obs.columns:
        raise ValueError(
            f"Required column '{GROUP_KEY}' not found.\n"
            f"Available: {list(adata.obs.columns)}"
        )

    # ============================================================
    # CLEAN ETHNICITY COLUMN
    # ============================================================
    heartbeat("\nCleaning self_reported_ethnicity column ...", True)
    grp_raw = adata.obs[GROUP_KEY].astype(str).str.lower().str.strip()
    keep = ~grp_raw.isin(UNKNOWN_VALUES) & (grp_raw.str.len() > 0)
    removed = int((~keep).sum())
    heartbeat(f" Removing {removed} cells with unknown ethnicity", True)
    adata = adata[keep].copy()
    adata.obs[GROUP_KEY] = grp_raw[keep].values

    # ============================================================
    # CLEAN CELL_TYPE WHITESPACE
    # AIDA has trailing-whitespace duplicates that scDesign3 would
    # treat as separate categories. Strip them.
    # ============================================================
    if "cell_type" in adata.obs.columns:
        before_n = adata.obs["cell_type"].nunique()
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).str.strip()
        after_n = adata.obs["cell_type"].nunique()
        heartbeat(f"\nCell type cleanup: {before_n} -> {after_n} unique values", True)

    # ============================================================
    # DISTRIBUTION REPORT
    # ============================================================
    heartbeat(f"\n {GROUP_KEY} distribution after cleaning:", True)
    grp_counts = adata.obs[GROUP_KEY].value_counts().sort_index()
    for label, cnt in grp_counts.items():
        heartbeat(f"  {label}: {cnt} cells", True)
    groups_found = sorted(adata.obs[GROUP_KEY].unique())
    heartbeat(f"\n Groups detected ({len(groups_found)}): {groups_found}", True)

    # ============================================================
    # COVARIATE VALIDATION
    # ============================================================
    heartbeat("\nValidating required covariates ...", True)
    for covar in REQUIRED_COVARS:
        if covar in adata.obs.columns:
            n_unique = adata.obs[covar].nunique()
            n_missing = int(adata.obs[covar].isna().sum())
            msg = f" {covar}: {n_unique} unique"
            if n_missing > 0:
                msg += f" ({n_missing} missing)"
            heartbeat(msg, True)
        else:
            heartbeat(f" {covar}: NOT FOUND", True)

    heartbeat("\nValidating optional covariates ...", True)
    for covar in OPTIONAL_COVARS:
        if covar in adata.obs.columns:
            heartbeat(f" {covar}: {adata.obs[covar].nunique()} unique", True)
        else:
            heartbeat(f" - {covar}: not present", True)

    # ============================================================
    # EXTRACT RAW COUNTS
    # ============================================================
    heartbeat("\nExtracting raw count matrix ...", True)
    if use_raw:
        raw_counts = adata.raw.X.copy()
        raw_var = adata.raw.var.copy()
    else:
        raw_counts = adata.X.copy()
        raw_var = adata.var.copy()

    adata_raw = sc.AnnData(
        X=raw_counts,
        obs=adata.obs.copy(),
        var=raw_var,
        uns=adata.uns.copy(),
    )
    heartbeat(f" Raw matrix shape: {adata_raw.shape}", True)
    counts_per_cell = np.asarray(adata_raw.X.sum(axis=1)).flatten()
    heartbeat(" Library size per cell:", True)
    heartbeat(f"  Min:    {counts_per_cell.min():.1f}", True)
    heartbeat(f"  Max:    {counts_per_cell.max():.1f}", True)
    heartbeat(f"  Mean:   {counts_per_cell.mean():.1f}", True)
    heartbeat(f"  Median: {np.median(counts_per_cell):.1f}", True)

    del adata, raw_counts
    import gc; gc.collect()

    # ============================================================
    # STRATIFIED VALIDATION SPLIT
    # ============================================================
    heartbeat(f"\nSplitting validation set ({VALIDATION_SIZE} cells, stratified by ethnicity) ...", True)
    rng = np.random.default_rng(RANDOM_SEED)

    eth_proportions = grp_counts / grp_counts.sum()
    per_group_val = (eth_proportions * VALIDATION_SIZE).round().astype(int)
    diff = VALIDATION_SIZE - per_group_val.sum()
    if diff != 0:
        largest = per_group_val.idxmax()
        per_group_val[largest] += diff

    eth_array = adata_raw.obs[GROUP_KEY].values
    val_indices = []
    for group, n_val in per_group_val.items():
        group_idx = np.where(eth_array == group)[0]
        if len(group_idx) < n_val:
            heartbeat(f" WARNING: {group} has {len(group_idx)} cells, requested {n_val}", True)
            n_val = len(group_idx)
        chosen = rng.choice(group_idx, size=int(n_val), replace=False)
        val_indices.extend(chosen.tolist())
        heartbeat(f"  {group}: holding out {n_val} cells", True)

    val_mask = np.zeros(adata_raw.n_obs, dtype=bool)
    val_mask[val_indices] = True

    adata_val   = adata_raw[val_mask].copy()
    adata_train = adata_raw[~val_mask].copy()

    heartbeat(f"\n Training pool: {adata_train.n_obs} cells", True)
    heartbeat(f" Validation:    {adata_val.n_obs} cells", True)

    # ============================================================
    # WRITE OUTPUTS
    # ============================================================
    heartbeat(f"\nSaving training pool -> {OUTPUT_FILE}", True)
    adata_train.write_h5ad(OUTPUT_FILE, compression="gzip")
    heartbeat(f"Saving validation -> {VALIDATION_FILE}", True)
    adata_val.write_h5ad(VALIDATION_FILE, compression="gzip")

    heartbeat("\n" + "=" * 70 + "\n", True)
    heartbeat("STEP 0A COMPLETE\n", True)
    heartbeat("=" * 70 + "\n", True)
    heartbeat(f" Training:   {OUTPUT_FILE} ({adata_train.n_obs} cells)", True)
    heartbeat(f" Validation: {VALIDATION_FILE} ({adata_val.n_obs} cells)", True)
    heartbeat(f" Groups detected: {groups_found}", True)
    heartbeat("\n Ready for Step 0B\n", True)

except Exception as e:
    heartbeat(f"\n ERROR: {str(e)}", True)
    sys.exit(1)
finally:
    log_con.close()