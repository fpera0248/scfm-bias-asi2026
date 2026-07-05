#!/usr/bin/env python3
"""STEP 3d — Verify External Validation Embedding (CRC ETHNICITY, scFoundation)"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE    = pathlib.Path("/data/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow")
EMB_KEY = "X_scfoundation"

candidates = [
    BASE / "step2a_embeddings" / "CRC_Eth_External_Validation_8572_scfoundation.h5ad",
    BASE / "step2a_embeddings" / "CRC_Eth_External_Validation_8572_scfoundation.h5ad",
]

print("STEP 3d -- Verify CRC ETH scFoundation validation embedding", flush=True)

OUT_FILE = None
for c in candidates:
    if c.exists():
        OUT_FILE = c
        break

if OUT_FILE is None:
    print("  ERROR: Validation embedding not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: {EMB_KEY} missing", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {ad.n_obs:,}  |  Unique embedding rows: {n_unique:,}", flush=True)
if n_unique <= 10:
    print("  ERROR: Degenerate embedding.", flush=True)
    sys.exit(1)

print("  OK -- validation embedding valid.", flush=True)
print("STEP 3d COMPLETE", flush=True)
