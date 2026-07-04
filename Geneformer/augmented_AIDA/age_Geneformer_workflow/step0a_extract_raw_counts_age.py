#!/usr/bin/env python3
"""
STEP 0A — Extract raw counts for AGE-based augmentation workflow
with comprehensive metadata validation & preparation.

Changes vs step0a_extract_raw_counts_ethnicity.py:
  [AGE 1] OUTPUT_FILE  : ...RawCounts_AGE.h5ad          (was ETHNICITY)
  [AGE 2] GROUP_KEY    : "development_stage"             (was "self_reported_ethnicity")
  [AGE 3] LOGFILE      : step0a_age_log.txt              (was step0a_ethnicity_log.txt)
  [AGE 4] UNKNOWN_VALUES extended with age-specific noise values
          ("adult", "child", "infant", "embryonic", "fetal", "newborn")
  [AGE 5] Numeric age extraction + 10-year bin assignment added to obs
          so the output file carries "age_num" and "age_bin_10yr" columns
          that Step 0B can use directly without re-parsing
  [AGE 6] Group distribution summary reports age bins, not ethnicity groups
  [AGE 7] REQUIRED_COVARS updated: development_stage required, ethnicity optional
  [GF 1]  Add n_counts to obs and ensembl_id to var for Geneformer tokenizer

This script:
  - Loads the original ILD AnnData object.
  - Cleans the `development_stage` column (lower-case, trims, drops unknowns).
  - Parses numeric age and assigns 10-year bins (age_bin_10yr).
  - Validates & prepares all metadata columns needed by scDesign3 and Geneformer.
  - Extracts raw (unnormalized) counts from `adata.raw`.
  - Writes a clean raw-counts file ready for the AGE-based
    augmentation step (Step 0B) and Geneformer embedding (Step 2A).

Output:
  AIDA_RawCounts_AGE.h5ad
"""

import scanpy as sc
import numpy as np
import pandas as pd
import time
import sys
import re

# ============================================================
# CONFIGURATION  [AGE 1-3]
# ============================================================
RAW_FILE    = "InterstitialLungDisease.h5ad"
OUTPUT_FILE = "AIDA_RawCounts_AGE.h5ad"   # [AGE 1]
GROUP_KEY   = "development_stage"                             # [AGE 2]
BIN_KEY     = "age_bin_10yr"                                  # [AGE 5] derived column
AGE_NUM_KEY = "age_num"                                       # [AGE 5] parsed numeric age
LOGFILE     = "step0a_age_log.txt"                            # [AGE 3]

# [AGE 5] 10-year bin definitions (underscore labels - no special chars)
AGE_BREAKS = list(range(10, 100, 10))   # [10, 20, 30, ..., 90]
AGE_LABELS = [f"{lo}_{lo+9}" for lo in AGE_BREAKS[:-1]]  # "10_19", "20_29", ..., "80_89"

# [AGE 7] Required covariates that Step 0B will use
REQUIRED_COVARS = {
    "cell_type":          "Cluster assignment or cell-type label",
    "development_stage":  "Age/developmental stage of the donor",
    "disease":            "Disease status (healthy / disease)",
    "donor_id":           "Unique donor identifier",
    "tissue":             "Tissue or sample source",
}

# Optional covariates (Step 0B will skip gracefully if missing)
OPTIONAL_COVARS = {
    "self_reported_ethnicity": "Ethnicity of the donor",
    "sex":                     "Sex of the donor (male / female)",
    "TobaccoStatus":           "Smoking status",
    "Sample_Source":           "Sample source or collection method",
    "tissue_type":             "Fine-grained tissue type",
}

# [AGE 4] Values that should be treated as missing/unknown for development_stage
UNKNOWN_VALUES = {
    "unknown", "na", "n/a", "not reported", "", "nan",
    "not applicable", "adult", "child", "infant",
    "embryonic", "fetal", "newborn",
}

# ============================================================
# LOGGING SETUP
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
heartbeat("STEP 0A: AGE AUGMENTATION - METADATA EXTRACTION & VALIDATION\n", True)
heartbeat("=" * 70 + "\n", True)

try:
    # ============================================================
    # LOAD ORIGINAL DATA
    # ============================================================
    heartbeat(f"Loading {RAW_FILE} ...", True)
    adata = sc.read_h5ad(RAW_FILE)
    heartbeat(f" Input shape: {adata.shape}", True)
    heartbeat(f" X type: {type(adata.X).__name__}", True)
    if hasattr(adata.X, "min"):
        heartbeat(f" X range: {adata.X.min():.3f} -> {adata.X.max():.3f}", True)

    # ============================================================
    # SAFETY CHECKS: RAW LAYER
    # ============================================================
    if adata.raw is None:
        raise ValueError(
            "No `adata.raw` layer found. Raw counts must be available before "
            "running Step 0B (scDesign3)."
        )
    heartbeat(f" adata.raw exists: shape {adata.raw.X.shape}", True)

    # ============================================================
    # SAFETY CHECKS: AGE COLUMN EXISTS  [AGE 2]
    # ============================================================
    if GROUP_KEY not in adata.obs.columns:
        raise ValueError(
            f"Required metadata column '{GROUP_KEY}' not found.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )
    heartbeat(f" Age column '{GROUP_KEY}' found", True)

    # ============================================================
    # CLEAN AGE COLUMN  [AGE 4]
    # ============================================================
    heartbeat("\nCleaning development_stage column ...", True)
    grp_raw = adata.obs[GROUP_KEY].astype(str).str.lower().str.strip()

    keep_unknown = ~grp_raw.isin(UNKNOWN_VALUES) & (grp_raw.str.len() > 0)
    removed_unknown = (~keep_unknown).sum()
    if removed_unknown > 0:
        heartbeat(f" Removing {removed_unknown} cells with unknown/invalid age stage", True)

    adata = adata[keep_unknown].copy()
    adata.obs[GROUP_KEY] = grp_raw[keep_unknown]

    # ============================================================
    # PARSE NUMERIC AGE + ASSIGN 10-YEAR BINS  [AGE 5]
    # ============================================================
    heartbeat("\nParsing numeric age from development_stage ...", True)

    age_text = adata.obs[GROUP_KEY].astype(str)
    age_digits = age_text.str.extract(r"(\d+)", expand=False)
    age_num    = pd.to_numeric(age_digits, errors="coerce")

    in_range = age_num.notna() & (age_num >= AGE_BREAKS[0]) & (age_num < AGE_BREAKS[-1])
    removed_range = (~in_range).sum()
    if removed_range > 0:
        heartbeat(
            f" Removing {removed_range} cells with unparseable or "
            f"out-of-range age (< {AGE_BREAKS[0]} or >= {AGE_BREAKS[-1]})", True
        )

    adata    = adata[in_range].copy()
    age_num  = age_num[in_range]

    adata.obs[AGE_NUM_KEY] = age_num.values.astype(float)

    age_bin = pd.cut(
        adata.obs[AGE_NUM_KEY],
        bins   = AGE_BREAKS,
        labels = AGE_LABELS,
        right  = False,
        include_lowest=True,
    )
    adata.obs[BIN_KEY] = age_bin.astype(str)

    heartbeat(f" Cells remaining after age parsing: {adata.n_obs}", True)

    # [AGE 6] Report age bin distribution
    bin_counts = adata.obs[BIN_KEY].value_counts().sort_index()
    heartbeat(f"\n Age bin distribution ({BIN_KEY}):", True)
    for bin_label, cnt in bin_counts.items():
        heartbeat(f"  {bin_label}: {cnt} cells", True)

    bins_found = sorted(adata.obs[BIN_KEY].unique())
    heartbeat(f"\n Bins detected ({len(bins_found)}): {bins_found}", True)

    # ============================================================
    # VALIDATE & PREPARE REQUIRED COVARIATES
    # ============================================================
    heartbeat("\nValidating required covariates ...", True)
    missing_required = []
    for covar, description in REQUIRED_COVARS.items():
        if covar in adata.obs.columns:
            n_unique  = adata.obs[covar].nunique()
            n_missing = adata.obs[covar].isna().sum()
            heartbeat(f" {covar}: {n_unique} unique values", newline=False)
            if n_missing > 0:
                heartbeat(f" ({n_missing} missing)", True)
            else:
                heartbeat("", True)
        else:
            heartbeat(f" {covar}: NOT FOUND", True)
            missing_required.append(covar)

    if missing_required:
        heartbeat(
            f"\nWARNING: {len(missing_required)} required covariates are missing:", True
        )
        for m in missing_required:
            heartbeat(f" - {m}", True)
        heartbeat(
            "\nStep 0B will still run, but scDesign3 cannot use missing covariates.\n"
            "Add these columns before augmentation if they are critical.", True
        )

    # ============================================================
    # VALIDATE OPTIONAL COVARIATES
    # ============================================================
    heartbeat("\nValidating optional covariates ...", True)
    for covar, description in OPTIONAL_COVARS.items():
        if covar in adata.obs.columns:
            n_unique  = adata.obs[covar].nunique()
            n_missing = adata.obs[covar].isna().sum()
            heartbeat(f" {covar}: {n_unique} unique values", newline=False)
            if n_missing > 0:
                heartbeat(f" ({n_missing} missing)", True)
            else:
                heartbeat("", True)
        else:
            heartbeat(f" - {covar}: not present (optional, will be skipped)", True)

    # ============================================================
    # EXTRACT RAW COUNTS & CREATE NEW AnnData
    # ============================================================
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
    heartbeat(
        f" Raw matrix range: {adata_raw.X.min():.1f} -> {adata_raw.X.max():.1f}", True
    )
    counts_per_cell = np.array(adata_raw.X.sum(axis=1)).flatten()
    heartbeat(" Library size per cell:", True)
    heartbeat(f"  Min:    {counts_per_cell.min():.1f}", True)
    heartbeat(f"  Max:    {counts_per_cell.max():.1f}", True)
    heartbeat(f"  Mean:   {counts_per_cell.mean():.1f}", True)
    heartbeat(f"  Median: {np.median(counts_per_cell):.1f}", True)

    # ============================================================
    # [GF 1] ADD GENEFORMER-REQUIRED METADATA
    # n_counts: per-cell library size (Geneformer tokenizer requires this)
    # ensembl_id: gene IDs in var (Geneformer expects this column name,
    #             var.index already holds Ensembl IDs in this cohort)
    # ============================================================
    heartbeat("\nAdding Geneformer-required metadata ...", True)

    adata_raw.obs["n_counts"] = counts_per_cell.astype(np.int64)
    heartbeat(
        f" Added obs['n_counts']: range {int(counts_per_cell.min())} - "
        f"{int(counts_per_cell.max())}", True
    )

    if "ensembl_id" not in adata_raw.var.columns:
        first_gene = str(adata_raw.var.index[0])
        if not first_gene.startswith("ENSG"):
            raise ValueError(
                f"var.index does not contain Ensembl IDs (first entry: '{first_gene}'). "
                "Geneformer requires Ensembl IDs. Add a symbol-to-Ensembl mapping step "
                "before writing the output file."
            )
        adata_raw.var["ensembl_id"] = adata_raw.var.index.astype(str)
        heartbeat(" Added var['ensembl_id'] from var index", True)
    else:
        heartbeat(" var['ensembl_id'] already exists", True)

    # ============================================================
    # FINAL METADATA SUMMARY
    # ============================================================
    heartbeat("\nFinal metadata summary:", True)
    heartbeat(f" Total cells: {adata_raw.n_obs}", True)
    heartbeat(f" Total genes: {adata_raw.n_vars}", True)

    heartbeat(f"\n Age bin distribution in output ({BIN_KEY}):", True)
    bin_counts_final = adata_raw.obs[BIN_KEY].value_counts().sort_index()
    for bin_label, cnt in bin_counts_final.items():
        heartbeat(f"  {bin_label}: {cnt} cells", True)

    heartbeat("\n Metadata columns in obs:", True)
    for col in sorted(adata_raw.obs.columns):
        n_unique = adata_raw.obs[col].nunique()
        dtype    = adata_raw.obs[col].dtype
        heartbeat(f" - {col} ({dtype}, {n_unique} unique)", True)

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    heartbeat(f"\nSaving raw-counts file -> {OUTPUT_FILE}", True)
    adata_raw.write_h5ad(OUTPUT_FILE, compression="gzip")
    heartbeat(" File saved successfully", True)

    # ============================================================
    # SUCCESS SUMMARY
    # ============================================================
    heartbeat("\n" + "=" * 70 + "\n", True)
    heartbeat("STEP 0A COMPLETE\n", True)
    heartbeat("=" * 70 + "\n", True)
    heartbeat(f" Output : {OUTPUT_FILE}", True)
    heartbeat(f" Log    : {LOGFILE}", True)
    heartbeat(f" Cells  : {adata_raw.n_obs}", True)
    heartbeat(f" Genes  : {adata_raw.n_vars}", True)
    heartbeat(f" Age bins detected: {bins_found}", True)
    heartbeat("\n Ready for Step 0B (scDesign3 AGE-based augmentation)", True)
    heartbeat(" Ready for Step 2A (Geneformer embedding)\n", True)

except Exception as e:
    heartbeat(f"\n ERROR: {str(e)}", True)
    heartbeat("Check the log file for details.", True)
    sys.exit(1)

finally:
    log_con.close()