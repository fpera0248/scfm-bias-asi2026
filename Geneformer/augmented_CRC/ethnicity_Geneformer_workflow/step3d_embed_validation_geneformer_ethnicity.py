#!/usr/bin/env python3
"""
STEP 3d — Verify External Validation Embedding exists (CRC ETHNICITY)
For CRC, the validation set was embedded during step2a. This step confirms
the output exists and is non-degenerate.
"""
import sys, pathlib
import numpy as np
import scanpy as sc

BASE     = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow")
OUT_FILE = BASE / "CRC_Eth_External_Validation_8572_geneformer.h5ad"
EMB_KEY  = "X_geneformer"
MIN_UNIQUE_ROWS = 10

print(f"STEP 3d -- Verify validation embedding", flush=True)
print(f"  Expected: {OUT_FILE}", flush=True)

if not OUT_FILE.exists():
    print(f"  ERROR: File not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: Embedding key '{EMB_KEY}' missing.", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {ad.n_obs:,}  |  Unique embedding rows: {n_unique:,}", flush=True)
if n_unique <= MIN_UNIQUE_ROWS:
    print(f"  ERROR: Degenerate embedding ({n_unique} unique rows).", flush=True)
    sys.exit(1)

print(f"  OK -- validation embedding valid.", flush=True)

# Symlink to expected location for downstream steps
LINK_TARGET = BASE / "CRC_Eth_External_Validation_8572_geneformer.h5ad"
if not LINK_TARGET.exists():
    import shutil
    shutil.copy(str(OUT_FILE), str(LINK_TARGET))
    print(f"  Copied to: {LINK_TARGET.name}", flush=True)

print("STEP 3d COMPLETE", flush=True)
