#!/usr/bin/env python3
"""
STEP 2a — Embed datasets with scFoundation (ETHNICITY)

Changes vs sex version:
  [ETH 1] BASE/INDIR    -> ethnicity_scfoundation_workflow
  [ETH 2] OUTPUT_BASE   -> ILD_Ethnicity_Pilot
  [ETH 3] DATASETS      -> ethnicity filenames (2143Each, 48Each, 2497)
  [ETH 4] VALIDATION    -> ILD_Ethnicity_External_Validation_12500
  [ETH 5] EMB_KEY       -> X_scfoundation (same)
"""

import sys
import scanpy as sc
import numpy as np
import torch
from modelgenerator.tasks import Embed
from tqdm import tqdm
import pathlib
import time
import scipy.sparse as sp

print(f"Python: {sys.executable}", flush=True)
print(f"Script: STEP 2a -- scFoundation ETHNICITY embedding", flush=True)

BASE   = pathlib.Path("/data/scfoundation/augmentedv4/ethnicity_scfoundation_workflow")
INDIR  = BASE
OUTDIR = BASE
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Ethnicity_Pilot"

DATASETS = {                                                                        # [ETH 3]
    "Proportional_2497": (
        f"{OUTPUT_BASE}_Proportional_2497_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_Proportional_2497_ETHNICITY_scfoundation.h5ad",
    ),
    "BalancedAugmented_2143Each": (
        f"{OUTPUT_BASE}_BalancedAugmented_2143Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_BalancedAugmented_2143Each_ETHNICITY_scfoundation.h5ad",
    ),
    "BalancedUpsampled_2143Each": (
        f"{OUTPUT_BASE}_BalancedUpsampled_2143Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_BalancedUpsampled_2143Each_ETHNICITY_scfoundation.h5ad",
    ),
    "Downsampled_48Each": (
        f"{OUTPUT_BASE}_Downsampled_48Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_Downsampled_48Each_ETHNICITY_scfoundation.h5ad",
    ),
    "ExternalValidation_12500": (                                                   # [ETH 4]
        "ILD_Ethnicity_External_Validation_12500.h5ad",
        "ILD_Ethnicity_External_Validation_12500_scfoundation.h5ad",
    ),
}

SEQ_LEN         = 15000
EMB_KEY         = "X_scfoundation"
DTYPE           = torch.float16
MIN_UNIQUE_ROWS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    free_mem   = torch.cuda.mem_get_info()[0] / 1024**3
    BATCH_SIZE = 64 if free_mem > 20 else 32 if free_mem > 10 else 16
else:
    BATCH_SIZE = 8

print(f"\nDevice    : {DEVICE}", flush=True)
print(f"Batch size: {BATCH_SIZE}", flush=True)

cfg   = {"model.backbone": "scfoundation"}
model = Embed.from_config(cfg).to(DEVICE).eval()
assert cfg["model.backbone"] == "scfoundation", "Backbone mismatch"
print("scFoundation backbone confirmed", flush=True)


def _pad_batch(dense_batch: np.ndarray) -> np.ndarray:
    n_cells = dense_batch.shape[0]
    n_genes = dense_batch.shape[1]
    take    = min(n_genes, SEQ_LEN)
    if n_genes > SEQ_LEN:
        part_idx   = np.argpartition(dense_batch, -take, axis=1)[:, -take:]
        part_vals  = np.take_along_axis(dense_batch, part_idx, axis=1)
        sort_ord   = np.argsort(part_vals, axis=1)[:, ::-1]
        sorted_top = np.take_along_axis(part_vals, sort_ord, axis=1)
    else:
        sort_ord   = np.argsort(dense_batch, axis=1)[:, ::-1]
        sorted_top = np.take_along_axis(dense_batch, sort_ord, axis=1)
    out = np.zeros((n_cells, SEQ_LEN), dtype=np.int32)
    out[:, :take] = sorted_top.astype(np.int32)
    zero_rows         = out[:, 0] == 0
    out[zero_rows, 0] = 1
    return out


def is_degenerate_embedding(out_path):
    try:
        ad = sc.read_h5ad(out_path)
        if EMB_KEY not in ad.obsm: return True
        n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
        if n_unique <= MIN_UNIQUE_ROWS:
            print(f"   Existing embedding degenerate ({n_unique} unique rows) -- regenerating.")
            return True
        print(f"   Existing embedding valid ({n_unique} unique rows) -- skipping.")
        return False
    except Exception as e:
        print(f"   Could not validate ({e}) -- regenerating.")
        return True


def embed_dataset(adata: sc.AnnData) -> np.ndarray:
    n_cells   = adata.n_obs
    n_batches = int(np.ceil(n_cells / BATCH_SIZE))

    print(f"   Loading sparse matrix into RAM ...", flush=True)
    t_load = time.time()
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    else:
        X = X.copy()
    load_sec = time.time() - t_load
    nnz_gb   = X.data.nbytes / 1024**3
    print(f"   Matrix loaded in {load_sec:.1f}s  |  nnz={X.nnz:,}  ({nnz_gb:.2f} GB sparse data)", flush=True)

    embeddings = np.zeros((n_cells, 768), dtype=np.float32)
    print(f"   Batches: {n_batches}  (BATCH_SIZE={BATCH_SIZE}, cells={n_cells:,})", flush=True)

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="   Batch embed", file=sys.stdout):
            s = b * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_cells)
            batch_dense = X[s:e].toarray()
            batch_pad   = _pad_batch(batch_dense)
            input_ids   = torch.tensor(batch_pad, device=DEVICE, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=DTYPE):
                out = model({"input_ids": input_ids})
                emb = out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            embeddings[s:e] = emb
            del input_ids, out, emb, batch_dense, batch_pad
            if DEVICE == "cuda" and b % 50 == 0:
                torch.cuda.empty_cache()

    return embeddings


total_start = time.time()

for label, (in_fname, out_fname) in DATASETS.items():
    in_path  = INDIR  / in_fname
    out_path = OUTDIR / out_fname

    if not in_path.exists():
        print(f"\nSkipping {label} -- input not found: {in_fname}", flush=True)
        continue

    if out_path.exists():
        print(f"\n{'='*70}", flush=True)
        print(f"  {label} -- output already exists: {out_fname}", flush=True)
        if not is_degenerate_embedding(out_path):
            continue

    print(f"\n{'='*70}", flush=True)
    print(f"Embedding: {label}", flush=True)
    print(f"   Input : {in_path}", flush=True)
    print(f"   Output: {out_path}", flush=True)

    adata = sc.read_h5ad(in_path)
    print(f"   Cells : {adata.n_obs:,}", flush=True)
    print(f"   Genes : {adata.n_vars:,}", flush=True)

    mean_counts = float(np.asarray(adata.X.sum(axis=1)).mean())
    print(f"   Mean counts/cell: {mean_counts:.1f}", flush=True)
    if mean_counts == 0:
        print(f"   ERROR: Count matrix empty -- skipping.", flush=True)
        continue

    t0         = time.time()
    embeddings = embed_dataset(adata)
    elapsed    = time.time() - t0
    print(f"\n   Embed runtime: {elapsed/60:.1f} min  ({elapsed/adata.n_obs:.3f} s/cell)", flush=True)

    n_unique = len(np.unique(embeddings, axis=0))
    print(f"   Unique embedding rows: {n_unique:,}", flush=True)
    if n_unique <= MIN_UNIQUE_ROWS:
        print(f"   ERROR: Degenerate output -- NOT saving.", flush=True)
        continue

    adata.obsm[EMB_KEY] = embeddings
    adata.write(out_path)
    print(f"   Saved -> {out_fname}", flush=True)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}", flush=True)
print(f"STEP 2a COMPLETE (ETHNICITY, scFoundation)", flush=True)
print(f"   Total runtime: {total_elapsed/60:.1f} min", flush=True)
for label, (_, out_fname) in DATASETS.items():
    status = "OK" if (OUTDIR / out_fname).exists() else "MISSING"
    print(f"  [{status}] {out_fname}  [{label}]", flush=True)
