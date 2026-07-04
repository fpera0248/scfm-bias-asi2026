#!/usr/bin/env python3
import sys, os
import anndata as ad
import numpy as np
import scipy.sparse as sp

if len(sys.argv) < 2:
    print("Usage: validate_augmented_h5ad.py <h5ad> [min_synth_lib=500]")
    sys.exit(2)

path = sys.argv[1]
min_synth_lib = int(sys.argv[2]) if len(sys.argv) > 2 else 500

if not os.path.exists(path):
    print(f"FAIL: file not found: {path}")
    sys.exit(1)

a = ad.read_h5ad(path)
if 'source' not in a.obs.columns:
    print(f"OK: no 'source' column, not augmented data")
    sys.exit(0)

if sp.issparse(a.X):
    libsize = np.array(a.X.sum(axis=1)).flatten()
else:
    libsize = a.X.sum(axis=1)

sources = a.obs['source'].astype(str).values
synth_mask = sources == 'synthetic'
real_mask = sources == 'real'

if synth_mask.sum() == 0:
    print(f"OK: no synthetic cells")
    sys.exit(0)

synth_lib = float(np.median(libsize[synth_mask]))
real_lib = float(np.median(libsize[real_mask])) if real_mask.sum() > 0 else 0

print(f"{path}")
print(f"  shape: {a.shape}")
print(f"  real n={int(real_mask.sum())} median lib={real_lib:.0f}")
print(f"  synth n={int(synth_mask.sum())} median lib={synth_lib:.0f}")

if synth_lib < min_synth_lib:
    print(f"FAIL: synth lib {synth_lib:.0f} < {min_synth_lib}")
    sys.exit(1)

print("OK")
sys.exit(0)
