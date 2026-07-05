#!/usr/bin/env python3
"""
STEP 0a -- Extract AIDA raw counts with age bins for AGE workflow.
Reads the 900k ethnicity-subsampled file (same cell pool as ethnicity workflow),
parses age from development_stage, creates 5 decade bins with 60+ folded,
drops age <20, writes RawCounts h5ad + external validation set.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import re

INPUT_H5AD  = "/data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow/AIDA_RawCounts_ETHNICITY_900k.h5ad"
OUT_RAW     = "AIDA_RawCounts_AGE.h5ad"
OUT_VAL     = "AIDA_Age_External_Validation_10000.h5ad"
VAL_N       = 10000   # similar scale to ILD's 10,500 age val
BIN_KEY     = "age_bin_10yr"
AGE_NUM_KEY = "age_num"

def parse_age(s):
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None

print(f"Reading: {INPUT_H5AD}")
ad = sc.read_h5ad(INPUT_H5AD)
print(f"Loaded: {ad.n_obs:,} cells, {ad.n_vars:,} genes")

ad.obs[AGE_NUM_KEY] = ad.obs['development_stage'].astype(str).map(parse_age)
n_unparsable = ad.obs[AGE_NUM_KEY].isna().sum()
if n_unparsable:
    print(f"  Dropping {n_unparsable} unparsable ages")
ad = ad[ad.obs[AGE_NUM_KEY].notna()].copy()

# Drop age <20
n_before = ad.n_obs
ad = ad[ad.obs[AGE_NUM_KEY] >= 20].copy()
print(f"  Dropped {n_before - ad.n_obs} cells with age <20")

# 5 decade bins with 60+ folded
def to_bin(age):
    if age < 30: return "20-29"
    if age < 40: return "30-39"
    if age < 50: return "40-49"
    if age < 60: return "50-59"
    return "60+"

ad.obs[BIN_KEY] = ad.obs[AGE_NUM_KEY].astype(int).map(to_bin)
ad.obs[BIN_KEY] = ad.obs[BIN_KEY].astype("category")

# Strip whitespace on cell_type for downstream consistency
if 'cell_type' in ad.obs.columns:
    ad.obs['cell_type'] = ad.obs['cell_type'].astype(str).str.strip()

print("\nFinal age bin distribution:")
print(ad.obs[BIN_KEY].value_counts().sort_index())

# Stratified validation: VAL_N cells across 5 bins
rng = np.random.default_rng(seed=42)
val_indices = []
per_bin = VAL_N // 5
for b in ["20-29", "30-39", "40-49", "50-59", "60+"]:
    mask = (ad.obs[BIN_KEY] == b).to_numpy()
    candidates = np.where(mask)[0]
    take = min(per_bin, len(candidates))
    val_indices.extend(rng.choice(candidates, size=take, replace=False).tolist())

val_indices = np.array(val_indices)
val_mask = np.zeros(ad.n_obs, dtype=bool)
val_mask[val_indices] = True

ad_val = ad[val_mask].copy()
ad_train = ad[~val_mask].copy()

print(f"\nValidation: {ad_val.n_obs:,} cells (stratified by age bin)")
print(ad_val.obs[BIN_KEY].value_counts().sort_index())
print(f"\nTraining pool: {ad_train.n_obs:,} cells")
print(ad_train.obs[BIN_KEY].value_counts().sort_index())

ad_train.write_h5ad(OUT_RAW, compression="gzip")
ad_val.write_h5ad(OUT_VAL, compression="gzip")
print(f"\nWrote: {OUT_RAW} and {OUT_VAL}")
