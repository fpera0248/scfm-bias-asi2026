#!/usr/bin/env python3
"""
STEP 3d — Verify External Validation Embedding exists (CRC AGE)
Validation was embedded during step2a. This step confirms it and symlinks.
"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE     = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/age_Geneformer_workflow")
EMB_KEY  = "X_geneformer"
MIN_UNIQUE = 10

# check root dir first, then step2a_embeddings subdir
candidates = [
    BASE / "CRC_Age_External_Validation_9402_geneformer.h5ad",
    BASE / "step2a_embeddings" / "CRC_Age_External_Validation_9402_geneformer.h5ad",
]

OUT_FILE = None
for c in candidates:
    if c.exists():
        OUT_FILE = c
        break

print(f"STEP 3d -- Verify CRC AGE validation embedding", flush=True)

if OUT_FILE is None:
    print(f"  ERROR: Validation embedding not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: {EMB_KEY} missing from {OUT_FILE.name}", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {ad.n_obs:,}  |  Unique embedding rows: {n_unique:,}", flush=True)
if n_unique <= MIN_UNIQUE:
    print(f"  ERROR: Degenerate embedding.", flush=True)
    sys.exit(1)

# ensure it exists in root dir for downstream scripts
root_copy = BASE / "CRC_Age_External_Validation_9402_geneformer.h5ad"
if not root_copy.exists():
    shutil.copy(str(OUT_FILE), str(root_copy))
    print(f"  Copied to root: {root_copy.name}", flush=True)

print("  OK -- validation embedding valid.", flush=True)
print("STEP 3d COMPLETE", flush=True)
