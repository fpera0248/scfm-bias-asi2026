#!/usr/bin/env python3
"""
STEP 0A — Extract raw counts for ETHNICITY-based augmentation workflow
Geneformer workflow version.

Changes vs. augmentedv4 version:
  [GF 1] BASE path: augmentedv4/ethnicity_scfoundation_workflow
                 -> Geneformer/augmented/ethnicity_Geneformer_workflow
  All other logic preserved verbatim.

See original script header for full documentation.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import time
import sys

# ============================================================
# CONFIGURATION
# ============================================================
RAW_FILE    = "InterstitialLungDisease.h5ad"
OUTPUT_FILE = "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"
GROUP_KEY   = "self_reported_ethnicity"
LOGFILE     = "step0a_ethnicity_log.txt"

REQUIRED_COVARS = {
    "cell_type":                 "Cluster assignment or cell-type label",
    "self_reported_ethnicity":   "Ethnicity of the donor",
    "disease":                   "Disease status (healthy / disease)",
    "sex":                       "Sex of the donor (male / female)",
    "donor_id":                  "Unique donor identifier",
    "tissue":                    "Tissue or sample source",
    "development_stage":         "Age metadata (used as age control covariate in step0b)",
}

OPTIONAL_COVARS = {
    "TobaccoStatus":  "Smoking status",
    "Sample_Source":  "Sample source or collection method",
    "tissue_type":    "Fine-grained tissue type",
}

UNKNOWN_VALUES = {
    "unknown", "na", "n/a", "not reported", "", "nan",
    "multiethnic", "na na", "not applicable", "prefer not to say"
}

# ============================================================
# LOGGING
# ============================================================
log_con = open(LOGFILE, "w")

def heartbeat(msg: str, newline: bool = False) -> None:
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if newline:
        line += "\n"
    print(line, file=log_con, flush=True)
    print(line, file=sys.stderr, end="")
    sys.stderr.flush()

heartbeat("\n" + "=" * 70 + "\n", True)
heartbeat("STEP 0A: ETHNICITY AUGMENTATION (Geneformer) — METADATA EXTRACTION\n", True)
heartbeat("=" * 70 + "\n", True)

try:
    heartbeat(f"Loading {RAW_FILE} ...", True)
    adata = sc.read_h5ad(RAW_FILE)
    heartbeat(f" Input shape: {adata.shape}", True)
    heartbeat(f" X type: {type(adata.X).__name__}", True)

    if adata.raw is None:
        raise ValueError("No `adata.raw` layer found.")
    heartbeat(f" adata.raw exists: shape {adata.raw.X.shape}", True)

    if GROUP_KEY not in adata.obs.columns:
        raise ValueError(
            f"Required column '{GROUP_KEY}' not found.\n"
            f"Available: {list(adata.obs.columns)}"
        )
    heartbeat(f" Column '{GROUP_KEY}' found", True)

    heartbeat("\nCleaning ethnicity column ...", True)
    grp_raw = adata.obs[GROUP_KEY].astype(str).str.lower().str.strip()

    heartbeat(" Raw value counts (before filtering):", True)
    for val, cnt in grp_raw.value_counts().items():
        tag = " [WILL DROP]" if val in UNKNOWN_VALUES or len(val) == 0 else ""
        heartbeat(f"  '{val}': {cnt}{tag}", True)

    # [FIX 001] operator-precedence fix — parentheses around both conditions
    keep = (~grp_raw.isin(UNKNOWN_VALUES)) & (grp_raw.str.len() > 0)

    removed = (~keep).sum()
    if removed > 0:
        heartbeat(f" Removing {removed} cells with unknown/invalid ethnicity", True)

    adata = adata[keep].copy()
    adata.obs[GROUP_KEY] = grp_raw[keep]
    heartbeat(f" Cells remaining: {adata.n_obs}", True)

    groups_found = sorted(adata.obs[GROUP_KEY].unique())
    heartbeat(f" Groups detected ({len(groups_found)}): {groups_found}", True)

    if "african american" not in groups_found:
        heartbeat(" WARNING: 'african american' missing after cleaning.", True)
    else:
        n_aa = (adata.obs[GROUP_KEY] == "african american").sum()
        heartbeat(f" African American confirmed: {n_aa} cells", True)

    heartbeat("\nValidating required covariates ...", True)
    missing_required = []
    for covar, desc in REQUIRED_COVARS.items():
        if covar in adata.obs.columns:
            n_unique  = adata.obs[covar].nunique()
            n_missing = adata.obs[covar].isna().sum()
            suffix    = f" ({n_missing} missing)" if n_missing > 0 else ""
            heartbeat(f" {covar}: {n_unique} unique values{suffix}", True)
        else:
            heartbeat(f" {covar}: NOT FOUND", True)
            missing_required.append(covar)

    if missing_required:
        heartbeat(f"\nWARNING: {len(missing_required)} required covariates missing: {missing_required}", True)

    heartbeat("\nValidating optional covariates ...", True)
    for covar, desc in OPTIONAL_COVARS.items():
        if covar in adata.obs.columns:
            heartbeat(f" {covar}: {adata.obs[covar].nunique()} unique values", True)
        else:
            heartbeat(f" - {covar}: not present (optional)", True)

    heartbeat("\nExtracting raw count matrix from adata.raw ...", True)
    raw_counts = adata.raw.X.copy()
    raw_var    = adata.raw.var.copy()
    adata_raw  = sc.AnnData(
        X   = raw_counts,
        obs = adata.obs.copy(),
        var = raw_var,
        uns = adata.uns.copy(),
    )
    heartbeat(f" Raw matrix shape: {adata_raw.shape}", True)
    heartbeat(f" Raw matrix range: {adata_raw.X.min():.1f} to {adata_raw.X.max():.1f}", True)

    counts_per_cell = np.array(adata_raw.X.sum(axis=1)).flatten()
    heartbeat(f" Library size — mean: {counts_per_cell.mean():.1f}, median: {np.median(counts_per_cell):.1f}", True)

    heartbeat("\nFinal metadata summary:", True)
    heartbeat(f" Total cells: {adata_raw.n_obs}", True)
    heartbeat(f" Total genes: {adata_raw.n_vars}", True)

    heartbeat("\n Ethnicity distribution:", True)
    grp_counts = adata_raw.obs[GROUP_KEY].value_counts().sort_index()
    total_kept = grp_counts.sum()
    for grp_label, cnt in grp_counts.items():
        pct = 100 * cnt / total_kept
        heartbeat(f"  {grp_label}: {cnt} ({pct:.1f}%)", True)

    maj_group = grp_counts.idxmax()
    min_group = grp_counts.idxmin()
    imb_ratio = grp_counts.max() / grp_counts.min()
    heartbeat(f"\n Majority: {maj_group} ({grp_counts[maj_group]} cells)", True)
    heartbeat(f" Minority: {min_group} ({grp_counts[min_group]} cells)", True)
    heartbeat(f" Imbalance ratio: {imb_ratio:.1f}:1", True)

    heartbeat(f"\nSaving to {OUTPUT_FILE}", True)
    adata_raw.write_h5ad(OUTPUT_FILE, compression="gzip")
    heartbeat(" File saved successfully", True)

    heartbeat("\n" + "=" * 70 + "\n", True)
    heartbeat("STEP 0A COMPLETE\n", True)
    heartbeat(f"Output: {OUTPUT_FILE}\n", True)
    heartbeat(f"Groups: {len(groups_found)} ({', '.join(groups_found)})\n", True)

except Exception as e:
    heartbeat(f"\n ERROR: {str(e)}", True)
    sys.exit(1)

finally:
    log_con.close()