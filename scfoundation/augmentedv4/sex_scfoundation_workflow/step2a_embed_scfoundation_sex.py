#!/usr/bin/env python3
"""
STEP 2a — Embed datasets with scFoundation (SEX)
FIXED v2 — eliminates 25-hr vectorization bottleneck.

ROOT CAUSE of slow vectorization:
    adata.X[i] on an HDF5-backed AnnData triggers one HDF5 read per row.
    At 200k cells this is 200k sequential disk reads → ~24 hrs.

FIX (two-part):
    1. Load the full sparse matrix into RAM in one shot with adata.X[:].
       For a 200k × 31k sparse matrix this is typically <2 GB in CSR format.
    2. Fuse vectorize+embed into a single batch loop — never build the full
       (n_obs × SEQ_LEN) padded array; create and discard it per batch.

Expected speedup: vectorization goes from ~24 hrs → ~2–5 min.
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
print(f"Script: STEP 2a FIXED v2 — fused batch embed, RAM-loaded matrix", flush=True)

# ============================================================
# PATHS  ← edit these to match your run
# ============================================================

BASE = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow"
)
INDIR  = BASE
OUTDIR = BASE
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Sex_Pilot"

# ── Datasets to embed ────────────────────────────────────────
# Map:  label → (input_filename, output_filename)
# Already-embedded files will be skipped automatically via the
# is_degenerate_embedding() check below.
DATASETS = {
    "Proportional_1999": (
        f"{OUTPUT_BASE}_Proportional_1999_SEX.h5ad",
        f"{OUTPUT_BASE}_Proportional_1999_SEX_scfoundation.h5ad",
    ),
    "BalancedAugmented_1413Each": (
        f"{OUTPUT_BASE}_BalancedAugmented_1413Each_SEX.h5ad",
        f"{OUTPUT_BASE}_BalancedAugmented_1413Each_SEX_scfoundation.h5ad",
    ),
    "BalancedUpsampled_1413Each": (
        f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_SEX.h5ad",
        f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_SEX_scfoundation.h5ad",
    ),
    "Downsampled_586Each": (
        f"{OUTPUT_BASE}_Downsampled_586Each_SEX.h5ad",
        f"{OUTPUT_BASE}_Downsampled_586Each_SEX_scfoundation.h5ad",
    ),
    "ExternalValidation_5000": (
        "ILD_Sex_External_Validation_5000.h5ad",
        "ILD_Sex_External_Validation_5000_scfoundation.h5ad",
    ),
}

# ============================================================
# CONFIG
# ============================================================

SEQ_LEN         = 15000   # scFoundation mae_encoder_max_seq_len
EMB_KEY         = "X_scfoundation"
DTYPE           = torch.float16
MIN_UNIQUE_ROWS = 10

# ============================================================
# DEVICE + BATCH SIZE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    free_mem   = torch.cuda.mem_get_info()[0] / 1024**3
    BATCH_SIZE = 64 if free_mem > 20 else 32 if free_mem > 10 else 16
else:
    BATCH_SIZE = 8

print(f"\n🧠 Device    : {DEVICE}", flush=True)
print(f"🧠 Batch size: {BATCH_SIZE}", flush=True)

# ============================================================
# LOAD MODEL
# ============================================================

cfg   = {"model.backbone": "scfoundation"}
model = Embed.from_config(cfg).to(DEVICE).eval()
assert cfg["model.backbone"] == "scfoundation", "❌ Backbone mismatch"
print("✅ scFoundation backbone confirmed", flush=True)

# ============================================================
# HELPERS
# ============================================================

def _pad_batch(dense_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised pad for shape (batch, n_genes) → (batch, SEQ_LEN).

    Sort each cell's gene values descending so highest counts are
    at the front, then truncate/zero-pad to SEQ_LEN.
    Any all-zero cell gets position-0 set to 1 to prevent src_len=0
    crashing inside the scFoundation transformer.
    """
    n_cells  = dense_batch.shape[0]
    n_genes  = dense_batch.shape[1]
    take     = min(n_genes, SEQ_LEN)

    # Descending sort — only need top `take` values
    # np.argpartition + sort is faster than full argsort for large n_genes
    if n_genes > SEQ_LEN:
        # Partial sort: get indices of top `take` values, then sort those
        part_idx  = np.argpartition(dense_batch, -take, axis=1)[:, -take:]
        part_vals = np.take_along_axis(dense_batch, part_idx, axis=1)
        sort_ord  = np.argsort(part_vals, axis=1)[:, ::-1]
        sorted_top = np.take_along_axis(part_vals, sort_ord, axis=1)
    else:
        sort_ord   = np.argsort(dense_batch, axis=1)[:, ::-1]
        sorted_top = np.take_along_axis(dense_batch, sort_ord, axis=1)

    out = np.zeros((n_cells, SEQ_LEN), dtype=np.int32)
    out[:, :take] = sorted_top.astype(np.int32)

    # Safety: all-zero rows → set first token to 1
    zero_rows        = out[:, 0] == 0
    out[zero_rows, 0] = 1

    return out


def is_degenerate_embedding(out_path):
    try:
        ad      = sc.read_h5ad(out_path)
        if EMB_KEY not in ad.obsm:
            return True
        n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
        if n_unique <= MIN_UNIQUE_ROWS:
            print(f"   ⚠️  Existing embedding degenerate ({n_unique} unique rows) — regenerating.")
            return True
        print(f"   ✅ Existing embedding valid ({n_unique} unique rows) — skipping.")
        return False
    except Exception as e:
        print(f"   ⚠️  Could not validate ({e}) — regenerating.")
        return True

# ============================================================
# CORE EMBED FUNCTION
# ============================================================

def embed_dataset(adata: sc.AnnData) -> np.ndarray:
    """
    Embed all cells using scFoundation.

    Step 1: Load the full sparse matrix into RAM in one shot.
            This replaces 200k individual HDF5 row-reads with a single
            contiguous read — the root cause of the 24-hr bottleneck.

    Step 2: Fused batch loop: slice → dense → pad → GPU → embed.
            Peak memory per batch: BATCH_SIZE × SEQ_LEN × 4 bytes ≈ 4 MB.
    """
    n_cells   = adata.n_obs
    n_batches = int(np.ceil(n_cells / BATCH_SIZE))

    # ── Step 1: pull full matrix into RAM as CSR ──────────────────────────
    print(f"   Loading sparse matrix into RAM ...", flush=True)
    t_load = time.time()

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    else:
        # Even if already CSR, force materialisation from HDF5
        X = X.copy()

    load_sec = time.time() - t_load
    nnz_gb   = X.data.nbytes / 1024**3
    print(f"   Matrix loaded in {load_sec:.1f}s  |  nnz={X.nnz:,}  ({nnz_gb:.2f} GB sparse data)", flush=True)

    # ── Step 2: fused vectorize + embed ──────────────────────────────────
    embeddings = np.zeros((n_cells, 768), dtype=np.float32)
    print(f"   Batches: {n_batches}  (BATCH_SIZE={BATCH_SIZE}, cells={n_cells:,})", flush=True)

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="   Batch embed", file=sys.stdout):
            s = b * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_cells)

            # dense-convert this batch only (fast CSR row slice)
            batch_dense = X[s:e].toarray()          # (bs, n_genes) float32/64

            # vectorised pad
            batch_pad = _pad_batch(batch_dense)      # (bs, SEQ_LEN) int32

            input_ids = torch.tensor(batch_pad, device=DEVICE, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=DTYPE):
                out = model({"input_ids": input_ids})
                emb = out.last_hidden_state.mean(dim=1).float().cpu().numpy()

            embeddings[s:e] = emb

            del input_ids, out, emb, batch_dense, batch_pad
            if DEVICE == "cuda" and b % 50 == 0:
                torch.cuda.empty_cache()

    return embeddings

# ============================================================
# MAIN
# ============================================================

total_start = time.time()

for label, (in_fname, out_fname) in DATASETS.items():
    in_path  = INDIR  / in_fname
    out_path = OUTDIR / out_fname

    if not in_path.exists():
        print(f"\n⚠️  Skipping {label} — input not found: {in_fname}", flush=True)
        print(f"   Looked in: {in_path}", flush=True)
        continue

    if out_path.exists():
        print(f"\n{'='*70}", flush=True)
        print(f"⏭️  {label} — output already exists: {out_fname}", flush=True)
        if not is_degenerate_embedding(out_path):
            continue

    print(f"\n{'='*70}", flush=True)
    print(f"🚀 Embedding: {label}", flush=True)
    print(f"   Input : {in_path}", flush=True)
    print(f"   Output: {out_path}", flush=True)

    adata = sc.read_h5ad(in_path)
    print(f"   Cells : {adata.n_obs:,}", flush=True)
    print(f"   Genes : {adata.n_vars:,}", flush=True)

    mean_counts = float(np.asarray(adata.X.sum(axis=1)).mean())
    print(f"   Mean counts/cell: {mean_counts:.1f}", flush=True)
    if mean_counts == 0:
        print(f"   ❌ Count matrix is empty — skipping.", flush=True)
        continue

    t0         = time.time()
    embeddings = embed_dataset(adata)
    elapsed    = time.time() - t0

    print(f"\n   ⏱  Embed runtime: {elapsed/60:.1f} min  ({elapsed/adata.n_obs:.3f} s/cell)", flush=True)

    n_unique = len(np.unique(embeddings, axis=0))
    print(f"   Unique embedding rows: {n_unique:,}", flush=True)
    if n_unique <= MIN_UNIQUE_ROWS:
        print(f"   ❌ Degenerate output — NOT saving. Check input file.", flush=True)
        continue

    adata.obsm[EMB_KEY] = embeddings
    adata.write(out_path)
    print(f"   💾 Saved → {out_fname}", flush=True)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}", flush=True)
print(f"🎉 STEP 2a COMPLETE", flush=True)
print(f"   Total runtime: {total_elapsed/60:.1f} min", flush=True)
for label, (_, out_fname) in DATASETS.items():
    print(f"  • {out_fname}  [{label}]", flush=True)