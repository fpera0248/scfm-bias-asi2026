#!/usr/bin/env python3
"""
STEP 0a — Extract Raw Counts and Prepare Axis-Specific H5ADs (CRC)

Shared across all CRC workflows: runs once, reads the raw Moorman epithelial object
and writes the per-axis RawCounts + external-validation files into every CRC model
workflow directory (the staged step0b + step2a scripts read these).

Input:  ColorectalCancer_Epithelial.h5ad (47,107 cells, 25,344 genes)
        Raw counts in .raw.X; .X is normalized float32.

Outputs per axis (written to all CRC workflow directories):
  SEX: ColorectalCancer_RawCounts_SEX.h5ad + CRC_Sex_External_Validation_<N>.h5ad
  AGE: ColorectalCancer_RawCounts_AGE.h5ad + CRC_Age_External_Validation_<N>.h5ad  (adds age_bin_10yr)
  ETH: ColorectalCancer_RawCounts_ETH.h5ad + CRC_Eth_External_Validation_<N>.h5ad  (excludes unknown ethnicity)

Validation sets are sampled cell-wise (20% held-out), stratified by the axis
demographic column x cell_type, seeded (RANDOM_STATE=42) for reproducibility.
"""

import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ── Paths (mount your data at /data; see CONTAINER.md / README) ────────────────
CRC_BASE   = pathlib.Path("/data/Geneformer/augmented_CRC")
INPUT_H5AD = CRC_BASE / "ColorectalCancer_Epithelial.h5ad"

GF_BASE  = pathlib.Path("/data/Geneformer/augmented_CRC")
SCG_BASE = pathlib.Path("/data/scGPT/augmented_CRC")
SCF_BASE = pathlib.Path("/data/scfoundation/augmented_CRC")

WORKFLOW_DIRS = {
    "sex": [
        GF_BASE  / "sex_Geneformer_workflow",
        SCG_BASE / "sex_scGPT_workflow",
        SCF_BASE / "sex_scfoundation_workflow",
    ],
    "age": [
        GF_BASE  / "age_Geneformer_workflow",
        SCG_BASE / "age_scGPT_workflow",
        SCF_BASE / "age_scfoundation_workflow",
    ],
    "eth": [
        GF_BASE  / "ethnicity_Geneformer_workflow",
        SCG_BASE / "ethnicity_scGPT_workflow",
        SCF_BASE / "ethnicity_scfoundation_workflow",
    ],
}

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
VAL_FRACTION = 0.20

AGE_BREAKS = list(range(10, 90, 10))   # [10,20,30,40,50,60,70,80]
AGE_LABELS = [f"{a}_{a+9}" for a in AGE_BREAKS[:-1]] + ["80_plus"]

ETH_UNKNOWN = {"unknown", "na", "n/a", "not reported", "", "nan"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_age_bin(development_stage_series):
    """Parse 'XX-year-old stage' -> age_bin_10yr string."""
    def _bin(s):
        if pd.isna(s):
            return np.nan
        digits = "".join(c for c in str(s) if c.isdigit())
        if not digits:
            return np.nan
        age = int(digits)
        if age >= 80:
            return "80_plus"
        for i, lower in enumerate(AGE_BREAKS[:-1]):
            upper = AGE_BREAKS[i + 1]
            if lower <= age < upper:
                return AGE_LABELS[i]
        return np.nan
    return development_stage_series.apply(_bin)


def extract_raw_counts(adata):
    """Pull integer counts from .raw.X."""
    log("  Extracting raw counts from .raw ...")
    raw = adata.raw
    X = raw.X
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = sp.csr_matrix(X)
    sample = X.data[:10000] if X.nnz > 0 else np.array([])
    if len(sample) > 0 and not np.allclose(sample, np.round(sample)):
        raise RuntimeError(".raw.X does not appear to contain integer counts.")
    X = X.astype(np.float32)
    log(f"  Raw matrix: {X.shape[0]:,} cells x {X.shape[1]:,} genes  nnz={X.nnz:,}")
    return X, raw.var.copy()


def make_adata_raw(adata_full):
    """Build clean AnnData with raw counts + full obs."""
    X, var = extract_raw_counts(adata_full)
    ad = sc.AnnData(X=X, obs=adata_full.obs.copy(), var=var)
    ad.obs_names = adata_full.obs_names.copy()
    ad.var.index.name = "ensembl_id"
    if "feature_name" not in ad.var.columns and "feature_name" in adata_full.raw.var.columns:
        ad.var["feature_name"] = adata_full.raw.var["feature_name"]
    log(f"  AnnData built: {ad.n_obs:,} cells x {ad.n_vars:,} genes")
    return ad


def sample_validation(ad, strat_col, val_frac=VAL_FRACTION, seed=RANDOM_STATE):
    """Sample val_frac held-out validation cells, stratified by strat_col x cell_type."""
    rng = np.random.default_rng(seed)
    obs = ad.obs.copy()
    obs["_strat"] = obs[strat_col].astype(str) + "__" + obs["cell_type"].astype(str)
    val_mask = np.zeros(ad.n_obs, dtype=bool)
    obs_names_array = np.array(ad.obs_names)
    for grp, grp_df in obs.groupby("_strat"):
        positions = np.where(np.isin(obs_names_array, grp_df.index))[0]
        n_val = max(1, int(len(positions) * val_frac))
        chosen = rng.choice(positions, size=n_val, replace=False)
        val_mask[chosen] = True
    log(f"  Validation split ({strat_col}): {val_mask.sum():,} val / {(~val_mask).sum():,} train")
    return ~val_mask, val_mask


def write_to_dirs(ad, fname, dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / fname
        ad.write_h5ad(out, compression="gzip")
        log(f"  Written -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("STEP 0a -- Extract Raw Counts (CRC Epithelial)")
    log("=" * 70)

    log(f"Loading {INPUT_H5AD.name} ...")
    adata = sc.read_h5ad(INPUT_H5AD)
    log(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    log(f"  cell_type:  {adata.obs['cell_type'].value_counts().to_dict()}")
    log(f"  disease:    {adata.obs['disease'].value_counts().to_dict()}")
    log(f"  sex:        {adata.obs['sex'].value_counts().to_dict()}")
    log(f"  ethnicity:  {adata.obs['self_reported_ethnicity'].value_counts().to_dict()}")

    ad_raw = make_adata_raw(adata)
    del adata

    # Add age_bin_10yr
    log("Adding age_bin_10yr from development_stage ...")
    ad_raw.obs["age_bin_10yr"] = parse_age_bin(ad_raw.obs["development_stage"])
    log(f"  age_bin_10yr dist: {ad_raw.obs['age_bin_10yr'].value_counts().sort_index().to_dict()}")

    eth_lower = ad_raw.obs["self_reported_ethnicity"].str.strip().str.lower()
    ad_raw.obs["_eth_lower"] = eth_lower

    # ── SEX axis ──────────────────────────────────────────────────────────────
    log("\n=== SEX AXIS ===")
    ad_sex = ad_raw[ad_raw.obs["sex"].isin(["female", "male"])].copy()
    log(f"  Cells: {ad_sex.n_obs:,}  |  {ad_sex.obs['sex'].value_counts().to_dict()}")
    train_mask, val_mask = sample_validation(ad_sex, "sex")
    ad_sex_train = ad_sex[train_mask].copy()
    ad_sex_val   = ad_sex[val_mask].copy()
    write_to_dirs(ad_sex_train, "ColorectalCancer_RawCounts_SEX.h5ad", WORKFLOW_DIRS["sex"])
    write_to_dirs(ad_sex_val, f"CRC_Sex_External_Validation_{ad_sex_val.n_obs}.h5ad", WORKFLOW_DIRS["sex"])

    # ── AGE axis ──────────────────────────────────────────────────────────────
    log("\n=== AGE AXIS ===")
    ad_age = ad_raw[ad_raw.obs["age_bin_10yr"].notna()].copy()
    bin_counts = ad_age.obs["age_bin_10yr"].value_counts()
    keep_bins  = bin_counts[bin_counts >= 50].index
    dropped    = bin_counts[bin_counts < 50]
    if len(dropped) > 0:
        log(f"  Dropping age bins with <50 cells: {dropped.to_dict()}")
    ad_age = ad_age[ad_age.obs["age_bin_10yr"].isin(keep_bins)].copy()
    log(f"  Cells: {ad_age.n_obs:,}  |  {ad_age.obs['age_bin_10yr'].value_counts().sort_index().to_dict()}")
    train_mask, val_mask = sample_validation(ad_age, "age_bin_10yr")
    ad_age_train = ad_age[train_mask].copy()
    ad_age_val   = ad_age[val_mask].copy()
    write_to_dirs(ad_age_train, "ColorectalCancer_RawCounts_AGE.h5ad", WORKFLOW_DIRS["age"])
    write_to_dirs(ad_age_val, f"CRC_Age_External_Validation_{ad_age_val.n_obs}.h5ad", WORKFLOW_DIRS["age"])

    # ── ETHNICITY axis ────────────────────────────────────────────────────────
    log("\n=== ETHNICITY AXIS ===")
    ad_eth = ad_raw[~ad_raw.obs["_eth_lower"].isin(ETH_UNKNOWN)].copy()
    log(f"  Cells: {ad_eth.n_obs:,}  |  {ad_eth.obs['self_reported_ethnicity'].value_counts().to_dict()}")
    train_mask, val_mask = sample_validation(ad_eth, "self_reported_ethnicity")
    ad_eth_train = ad_eth[train_mask].copy()
    ad_eth_val   = ad_eth[val_mask].copy()
    write_to_dirs(ad_eth_train, "ColorectalCancer_RawCounts_ETH.h5ad", WORKFLOW_DIRS["eth"])
    write_to_dirs(ad_eth_val, f"CRC_Eth_External_Validation_{ad_eth_val.n_obs}.h5ad", WORKFLOW_DIRS["eth"])

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 0a COMPLETE")
    log(f"  SEX  train: {ad_sex_train.n_obs:,}  val: {ad_sex_val.n_obs:,}")
    log(f"  AGE  train: {ad_age_train.n_obs:,}  val: {ad_age_val.n_obs:,}")
    log(f"  ETH  train: {ad_eth_train.n_obs:,}  val: {ad_eth_val.n_obs:,}")
    summary = pd.DataFrame([
        {"axis": "sex",       "n_train": ad_sex_train.n_obs, "n_val": ad_sex_val.n_obs},
        {"axis": "age",       "n_train": ad_age_train.n_obs, "n_val": ad_age_val.n_obs},
        {"axis": "ethnicity", "n_train": ad_eth_train.n_obs, "n_val": ad_eth_val.n_obs},
    ])
    summary.to_csv(CRC_BASE / "step0a_CRC_summary.csv", index=False)
    log(f"  Summary -> {CRC_BASE / 'step0a_CRC_summary.csv'}")


if __name__ == "__main__":
    main()
