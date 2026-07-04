#!/usr/bin/env python3
"""
STEP 0A — Extract raw counts for SEX-based augmentation workflow
Geneformer workflow version.

Changes vs. augmentedv4 version:
  [GF 1] BASE path updated to Geneformer/augmented/sex_Geneformer_workflow
  [GF 2] development_stage added to REQUIRED_COVARS (needed as age control in step0b)
"""

import scanpy as sc
import numpy as np
import time
import sys

RAW_FILE    = "InterstitialLungDisease.h5ad"
OUTPUT_FILE = "InterstitialLungDisease_RawCounts_SEX.h5ad"
SEX_KEY     = "sex"
LOGFILE     = "step0a_sex_log.txt"

REQUIRED_COVARS = {
    "cell_type":               "Cluster assignment or cell-type label",
    "sex":                     "Sex of the donor (male / female)",
    "disease":                 "Disease status",
    "self_reported_ethnicity": "Ethnicity of the donor (control covariate in step0b)",
    "donor_id":                "Unique donor identifier",
    "tissue":                  "Tissue or sample source",
    "development_stage":       "Age metadata (used as age_bin_10yr control in step0b)",
}

OPTIONAL_COVARS = {
    "TobaccoStatus":  "Smoking status",
    "Sample_Source":  "Sample source",
    "tissue_type":    "Fine-grained tissue type",
}

UNKNOWN_VALUES = {"unknown", "na", "n/a", "not reported", "", "nan"}

log_con = open(LOGFILE, "w")

def heartbeat(msg, newline=False):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if newline:
        line += "\n"
    print(line, file=log_con, flush=True)
    print(line, file=sys.stderr, end="")
    sys.stderr.flush()

heartbeat("\n" + "=" * 70 + "\n", True)
heartbeat("STEP 0A: SEX AUGMENTATION (Geneformer) — METADATA EXTRACTION\n", True)
heartbeat("=" * 70 + "\n", True)

try:
    heartbeat(f"Loading {RAW_FILE} ...", True)
    adata = sc.read_h5ad(RAW_FILE)
    heartbeat(f" Input shape: {adata.shape}", True)
    heartbeat(f" X type: {type(adata.X).__name__}", True)

    if adata.raw is None:
        raise ValueError("No `adata.raw` layer found.")
    heartbeat(f" adata.raw exists: shape {adata.raw.X.shape}", True)

    if SEX_KEY not in adata.obs.columns:
        raise ValueError(f"Column '{SEX_KEY}' not found. Available: {list(adata.obs.columns)}")
    heartbeat(f" Sex column '{SEX_KEY}' found", True)

    heartbeat("\nCleaning sex column ...", True)
    sex_raw = adata.obs[SEX_KEY].astype(str).str.lower().str.strip()
    keep    = ~sex_raw.isin(UNKNOWN_VALUES) & sex_raw.isin({"female", "male"})
    removed = (~keep).sum()
    if removed > 0:
        heartbeat(f" Removing {removed} cells with unknown/invalid sex", True)
    adata = adata[keep].copy()
    adata.obs[SEX_KEY] = sex_raw[keep]
    heartbeat(f" Cells remaining: {adata.n_obs}", True)

    heartbeat("\nValidating required covariates ...", True)
    missing_required = []
    for covar, desc in REQUIRED_COVARS.items():
        if covar in adata.obs.columns:
            n_missing = adata.obs[covar].isna().sum()
            suffix    = f" ({n_missing} missing)" if n_missing > 0 else ""
            heartbeat(f" {covar}: {adata.obs[covar].nunique()} unique values{suffix}", True)
        else:
            heartbeat(f" {covar}: NOT FOUND", True)
            missing_required.append(covar)

    if missing_required:
        heartbeat(f"\nWARNING: missing required covariates: {missing_required}", True)

    heartbeat("\nValidating optional covariates ...", True)
    for covar, desc in OPTIONAL_COVARS.items():
        if covar in adata.obs.columns:
            heartbeat(f" {covar}: {adata.obs[covar].nunique()} unique values", True)
        else:
            heartbeat(f" - {covar}: not present (optional)", True)

    heartbeat("\nExtracting raw count matrix from adata.raw ...", True)
    adata_raw = sc.AnnData(
        X   = adata.raw.X.copy(),
        obs = adata.obs.copy(),
        var = adata.raw.var.copy(),
        uns = adata.uns.copy(),
    )
    heartbeat(f" Raw matrix shape: {adata_raw.shape}", True)
    heartbeat(f" Raw matrix range: {adata_raw.X.min():.1f} to {adata_raw.X.max():.1f}", True)

    counts_per_cell = np.array(adata_raw.X.sum(axis=1)).flatten()
    heartbeat(f" Library size — mean: {counts_per_cell.mean():.1f}, median: {np.median(counts_per_cell):.1f}", True)

    heartbeat("\nFinal metadata summary:", True)
    heartbeat(f" Total cells: {adata_raw.n_obs}", True)
    heartbeat(f" Total genes: {adata_raw.n_vars}", True)

    heartbeat("\n Sex distribution:", True)
    for label, cnt in adata_raw.obs[SEX_KEY].value_counts().items():
        pct = 100 * cnt / adata_raw.n_obs
        heartbeat(f"  {label}: {cnt} ({pct:.1f}%)", True)

    maj = adata_raw.obs[SEX_KEY].value_counts().idxmax()
    mn  = adata_raw.obs[SEX_KEY].value_counts().idxmin()
    ratio = adata_raw.obs[SEX_KEY].value_counts().max() / adata_raw.obs[SEX_KEY].value_counts().min()
    heartbeat(f"\n Majority: {maj} | Minority: {mn} | Ratio: {ratio:.1f}:1", True)

    heartbeat(f"\nSaving to {OUTPUT_FILE}", True)
    adata_raw.write_h5ad(OUTPUT_FILE, compression="gzip")
    heartbeat(" File saved successfully", True)

    heartbeat("\n" + "=" * 70 + "\n", True)
    heartbeat("STEP 0A COMPLETE\n", True)
    heartbeat(f"Output: {OUTPUT_FILE}\n", True)

except Exception as e:
    heartbeat(f"\n ERROR: {str(e)}", True)
    sys.exit(1)

finally:
    log_con.close()