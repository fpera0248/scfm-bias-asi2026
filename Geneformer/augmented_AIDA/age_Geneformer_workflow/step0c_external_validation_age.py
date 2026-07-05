#!/usr/bin/env python3
"""STEP 0c — Prepare external validation set (AGE, Geneformer)"""
import scanpy as sc
import numpy as np
import pathlib, time

BASE     = pathlib.Path("/data/Geneformer/augmented_AIDA/age_Geneformer_workflow")
IN_FILE  = BASE / "AIDA_Age_External_Validation_10000.h5ad"
OUT_FILE = BASE / "AIDA_Age_External_Validation_10000.h5ad"
AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group", "development_stage"]

def parse_age_bin(series):
    import re
    def _parse(v):
        v = str(v).strip().lower()
        m = re.search(r'(\d+)', v)
        if not m: return None
        age = int(m.group(1))
        for lo in range(10, 90, 10):
            if lo <= age < lo + 10:
                return f"{lo}_{lo+9}"
        return None
    return series.map(_parse)

print(f"[{time.strftime('%H:%M:%S')}] Loading {IN_FILE}")
adata = sc.read_h5ad(IN_FILE)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

# Ensure age_bin_10yr exists
if "age_bin_10yr" not in adata.obs.columns:
    for c in AGE_COL_CANDIDATES:
        if c in adata.obs.columns:
            adata.obs["age_bin_10yr"] = parse_age_bin(adata.obs[c])
            print(f"  Derived age_bin_10yr from '{c}'")
            break

# Drop unknown age
before = adata.n_obs
adata = adata[adata.obs["age_bin_10yr"].notna()].copy()
print(f"  Dropped {before - adata.n_obs} cells with unknown age (kept {adata.n_obs})")

# Ensure ensembl_id and n_counts
if "ensembl_id" not in adata.var.columns:
    adata.var["ensembl_id"] = adata.var.index.str.replace(r'\.\d+$', '', regex=True)
if "n_counts" not in adata.obs.columns:
    import scipy.sparse as sp
    X = adata.X
    adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).flatten() if sp.issparse(X) else X.sum(axis=1)

print(f"[{time.strftime('%H:%M:%S')}] Age bin distribution:")
print(adata.obs["age_bin_10yr"].value_counts().sort_index())
print(f"[{time.strftime('%H:%M:%S')}] Writing {OUT_FILE}")
adata.write_h5ad(OUT_FILE)
print("Done.")
