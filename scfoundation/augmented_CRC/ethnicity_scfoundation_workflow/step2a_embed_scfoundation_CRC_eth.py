#!/usr/bin/env python3
"""STEP 2a — scFoundation Cell Embedding (CRC AGE)"""
import sys, time
import scanpy as sc
import numpy as np
import torch
from modelgenerator.tasks import Embed
from tqdm import tqdm
import pathlib
import scipy.sparse as sp

BASE   = pathlib.Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow")
OUTDIR = BASE / "step2a_embeddings"
OUTDIR.mkdir(exist_ok=True)
RAW_COUNTS_PATH = BASE / "ColorectalCancer_RawCounts_ETH.h5ad"
OUTPUT_BASE     = "CRC_Eth_Pilot"

DATASETS = {
    "Proportional_2497":           (BASE / f"{OUTPUT_BASE}_Proportional_2497_ETHNICITY.h5ad",           f"{OUTPUT_BASE}_Proportional_2497_ETH_scfoundation.h5ad"),
    "BalancedAugmented_1880Each":  (BASE / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETHNICITY.h5ad",  f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETH_scfoundation.h5ad"),
    "BalancedUpsampled_1880Each":  (BASE / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETHNICITY.h5ad",  f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETH_scfoundation.h5ad"),
    "Downsampled_48Each":          (BASE / f"{OUTPUT_BASE}_Downsampled_48Each_ETHNICITY.h5ad",           f"{OUTPUT_BASE}_Downsampled_48Each_ETH_scfoundation.h5ad"),
    "ExternalValidation_8572":     (BASE / "CRC_Eth_External_Validation_8572.h5ad",                "CRC_Eth_External_Validation_8572_scfoundation.h5ad"),
}

SEQ_LEN = 15000; EMB_KEY = "X_scfoundation"; DTYPE = torch.float16; MIN_UNIQUE_ROWS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    free_mem   = torch.cuda.mem_get_info()[0] / 1024**3
    BATCH_SIZE = 64 if free_mem > 20 else 32 if free_mem > 10 else 16
else:
    BATCH_SIZE = 8

print(f"Device: {DEVICE}  Batch: {BATCH_SIZE}", flush=True)
cfg   = {"model.backbone": "scfoundation"}
model = Embed.from_config(cfg).to(DEVICE).eval()
assert cfg["model.backbone"] == "scfoundation"
print("scFoundation backbone confirmed", flush=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _pad_batch(dense_batch):
    n_cells = dense_batch.shape[0]; n_genes = dense_batch.shape[1]; take = min(n_genes, SEQ_LEN)
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
    out[out[:, 0] == 0, 0] = 1
    return out

def is_degenerate_embedding(out_path):
    try:
        ad = sc.read_h5ad(out_path)
        if EMB_KEY not in ad.obsm: return True
        n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
        if n_unique <= MIN_UNIQUE_ROWS: return True
        log(f"  OK: existing embedding valid ({n_unique} unique rows) -- skipping.")
        return False
    except: return True

def embed_dataset(adata):
    n_cells = adata.n_obs; n_batches = int(np.ceil(n_cells / BATCH_SIZE))
    log(f"  Loading sparse matrix into RAM...")
    X = adata.X
    if not sp.issparse(X): X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix): X = X.tocsr()
    else: X = X.copy()
    embeddings = np.zeros((n_cells, 768), dtype=np.float32)
    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="  Batch embed", file=sys.stdout):
            s = b * BATCH_SIZE; e = min(s + BATCH_SIZE, n_cells)
            batch_dense = X[s:e].toarray()
            batch_pad   = _pad_batch(batch_dense)
            input_ids   = torch.tensor(batch_pad, device=DEVICE, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=DTYPE):
                out = model({"input_ids": input_ids})
                emb = out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            embeddings[s:e] = emb
            del input_ids, out, emb, batch_dense, batch_pad
            if DEVICE == "cuda" and b % 50 == 0: torch.cuda.empty_cache()
    return embeddings

log("Loading raw counts...")
raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
log(f"  {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes")

for label, (in_path, out_fname) in DATASETS.items():
    out_path = OUTDIR / out_fname
    if not in_path.exists(): log(f"\nSkipping {label} -- not found"); continue
    if out_path.exists() and not is_degenerate_embedding(out_path): continue
    log(f"\n{'='*70}\nEmbedding: {label}")
    adata = sc.read_h5ad(in_path)
    log(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")
    mean_counts = float(np.asarray(adata.X.sum(axis=1)).mean())
    if mean_counts == 0: log("  ERROR: empty count matrix -- skipping."); continue
    t0         = time.time()
    embeddings = embed_dataset(adata)
    elapsed    = time.time() - t0
    log(f"  Runtime: {elapsed/60:.1f} min  ({elapsed/adata.n_obs:.3f} s/cell)")
    n_unique = len(np.unique(embeddings, axis=0))
    log(f"  Unique rows: {n_unique:,}")
    if n_unique <= MIN_UNIQUE_ROWS: log("  ERROR: degenerate -- NOT saving."); continue
    adata.obsm[EMB_KEY] = embeddings
    adata.write(out_path)
    log(f"  Saved -> {out_fname}")

log("STEP 2a CRC ETH scFoundation COMPLETE")
for label, (_, out_fname) in DATASETS.items():
    status = "OK" if (OUTDIR/out_fname).exists() else "MISSING"
    log(f"  [{status}] {out_fname}")
