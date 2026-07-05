#!/usr/bin/env python3
"""
STEP 2a — Embed datasets with Geneformer V2-316M (ETHNICITY)

Changes vs step2a_age_embed_geneformer.py:
  [ETH 1]  BASE path -> ethnicity_Geneformer_workflow
  [ETH 2]  OUTPUT_BASE -> AIDA_Ethnicity_Pilot
  [ETH 3]  DATASETS -> ethnicity filenames (2143Each, 48Each, Proportional_2500)
  [ETH 4]  Print strings updated for ethnicity context

All embedding logic (tokenize, EmbExtractor, re-sort, degenerate check)
preserved verbatim from step2a_age_embed_geneformer.py.

Checkpoint path: /data/geneformer/Geneformer-V2-316M
"""

import sys
import os
import shutil
import scanpy as sc
import numpy as np
import pandas as pd
import pathlib
import time

print(f"Python: {sys.executable}", flush=True)
print(f"Script: STEP 2a ETHNICITY -- Geneformer V2-316M embed", flush=True)

from geneformer import TranscriptomeTokenizer, EmbExtractor

# ============================================================
# PATHS  [ETH 1]
# ============================================================

BASE = pathlib.Path(
    "/data/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow"
)
INDIR  = BASE
OUTDIR = BASE
OUTDIR.mkdir(exist_ok=True)

TOKENIZE_DIR = BASE / "tokenized_datasets"
TOKENIZE_DIR.mkdir(exist_ok=True)

GENEFORMER_MODEL = pathlib.Path(
    "/data/Geneformer/geneformer_repo/Geneformer-V2-316M"
)

OUTPUT_BASE = "AIDA_Ethnicity_Pilot"  # [ETH 2]
RAW_COUNTS_PATH = BASE / "AIDA_RawCounts_ETHNICITY_900k.h5ad"

# ============================================================
# DATASETS  [ETH 3]
# Filenames match step0b output:
#   TARGET_PER_BIN = 2143  |  DOWN_TARGET = 48  |  Proportional = 2497
# ============================================================

DATASETS = {
    "Proportional_2500": (
        f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_geneformer.h5ad",
    ),
    "Full_BalancedAugmented": (
        f"{OUTPUT_BASE}_Full_BalancedAugmented_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_Full_BalancedAugmented_ETHNICITY_geneformer.h5ad",
    ),
    "BalancedAugmented_779Each": (
        f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_geneformer.h5ad",
    ),
    "BalancedUpsampled_779Each": (
        f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_geneformer.h5ad",
    ),
    "Downsampled_92Each": (
        f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY.h5ad",
        f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_geneformer.h5ad",
    ),
    "ExternalValidation_12500": (
        "AIDA_Ethnicity_External_Validation_12500.h5ad",
        "AIDA_Ethnicity_External_Validation_12500_geneformer.h5ad",
    ),
}

# ============================================================
# CONFIG
# ============================================================

EMB_KEY         = "X_geneformer"
EMB_DIM         = 1152        # V2-316M hidden size
EMB_LAYER       = -1          # second-to-last layer per Nature paper
FORWARD_BATCH   = 8
NPROC           = 4
MIN_UNIQUE_ROWS = 10

# ============================================================
# INPUT PREP HELPERS
# ============================================================

def ensure_ensembl_id(adata, label):
    candidates = ["ensembl_id", "gene_id", "ensembl", "gene_ids"]
    col = None
    for c in candidates:
        if c in adata.var.columns:
            col = c
            break
    if col is None and adata.var.index.str.startswith("ENSG").any():
        adata.var["ensembl_id"] = adata.var.index.str.split(".").str[0]
        print(f"  [GF 5] ensembl_id set from var index (stripped version suffix)", flush=True)
        return adata
    if col is None:
        raise RuntimeError(
            f"[GF 5] No Ensembl ID column found in {label}.\n"
            f"  var columns: {list(adata.var.columns)}\n"
            f"  var index sample: {list(adata.var.index[:5])}\n"
            f"  Add 'ensembl_id' to adata.var before running."
        )
    adata.var["ensembl_id"] = adata.var[col].astype(str).str.split(".").str[0]
    print(f"  [GF 5] ensembl_id set from '{col}' (stripped version suffix)", flush=True)
    return adata


def ensure_n_counts(adata, label):
    if "n_counts" not in adata.obs.columns:
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            n_counts = np.asarray(X.sum(axis=1)).ravel()
        else:
            n_counts = X.sum(axis=1)
        adata.obs["n_counts"] = n_counts.astype(np.float32)
        print(f"  [GF 5] n_counts computed from X (mean={n_counts.mean():.0f})", flush=True)
    else:
        print(f"  [GF 5] n_counts already present", flush=True)
    return adata


def ensure_obs_id(adata):
    adata.obs["obs_id"] = adata.obs.index.astype(str)
    return adata

# ============================================================
# REAL-CELL COUNT SWAP
# ============================================================

def swap_real_cell_counts(adata, raw_adata):
    import scipy.sparse as sp
    if "source" not in adata.obs.columns:
        return adata
    real_mask     = (adata.obs["source"] == "real").values
    real_barcodes = adata.obs.index[real_mask]
    available     = real_barcodes[real_barcodes.isin(raw_adata.obs.index)]
    if len(available) == 0:
        print(f"  [SWAP] No real barcodes in raw counts -- skipping.", flush=True)
        return adata
    adata_genes  = np.array([g.split(".")[0] for g in adata.var.index])
    raw_genes    = np.array([g.split(".")[0] for g in raw_adata.var.index])
    raw_gene_map = {g: i for i, g in enumerate(raw_genes)}
    col_map      = np.array([raw_gene_map.get(g, -1) for g in adata_genes])
    valid_cols   = col_map >= 0
    raw_sub = raw_adata[available, :]
    raw_X   = raw_sub.X
    if sp.issparse(raw_X):
        raw_X = raw_X.toarray().astype(np.float32)
    else:
        raw_X = raw_X.astype(np.float32)
    aligned                = np.zeros((len(available), adata.n_vars), dtype=np.float32)
    aligned[:, valid_cols] = raw_X[:, col_map[valid_cols]]
    X = adata.X
    if sp.issparse(X):
        X = X.toarray().astype(np.float32)
    else:
        X = X.astype(np.float32).copy()
    avail_pos    = np.where(adata.obs.index.isin(available))[0]
    X[avail_pos] = aligned
    adata.X      = sp.csr_matrix(X)
    print(f"  [SWAP] {len(available):,}/{real_mask.sum():,} real cells -> full-transcriptome counts", flush=True)
    return adata

# ============================================================
# TOKENIZE ONE DATASET
# ============================================================

def tokenize_dataset(adata, label, tok_dir):
    tmp_h5ad     = tok_dir / f"{label}_tmp.h5ad"
    dataset_name = label

    adata.write_h5ad(tmp_h5ad)
    print(f"  Wrote temp h5ad: {tmp_h5ad}", flush=True)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"obs_id": "obs_id", "cell_type": "cell_type"},
        nproc=NPROC,
        model_version="V2",
    )
    tk.tokenize_data(
        str(tok_dir),
        str(tok_dir),
        dataset_name,
        file_format="h5ad",
    )

    dataset_path = tok_dir / f"{dataset_name}.dataset"
    if not dataset_path.exists():
        raise RuntimeError(f"Tokenization failed: {dataset_path} not found")

    tmp_h5ad.unlink()
    print(f"  Tokenized -> {dataset_path}", flush=True)
    return dataset_path

# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(dataset_path, emb_out_dir, label):
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        emb_layer=EMB_LAYER,
        emb_label=["obs_id"],
        forward_batch_size=FORWARD_BATCH,
        nproc=NPROC,
        model_version="V2",
        max_ncells=None,
    )

    emb_out_dir.mkdir(exist_ok=True)
    embs = embex.extract_embs(
        str(GENEFORMER_MODEL),
        str(dataset_path),
        str(emb_out_dir),
        label,
    )
    return embs

# ============================================================
# DEGENERATE CHECK
# ============================================================

def is_degenerate_embedding(out_path):
    try:
        ad = sc.read_h5ad(out_path)
        if EMB_KEY not in ad.obsm:
            return True
        n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
        if n_unique <= MIN_UNIQUE_ROWS:
            print(f"  WARNING: existing embedding degenerate ({n_unique} unique rows) -- regenerating.", flush=True)
            return True
        print(f"  OK: existing embedding valid ({n_unique} unique rows) -- skipping.", flush=True)
        return False
    except Exception as e:
        print(f"  WARNING: could not validate ({e}) -- regenerating.", flush=True)
        return True

# ============================================================
# MAIN
# ============================================================

if not GENEFORMER_MODEL.exists():
    raise RuntimeError(
        f"Geneformer model not found: {GENEFORMER_MODEL}\n"
        f"Clone from: https://huggingface.co/ctheodoris/Geneformer"
    )

total_start = time.time()

print(f"Loading raw counts for real-cell swap...", flush=True)
raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
print(f"  Raw counts: {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes", flush=True)

for label, (in_fname, out_fname) in DATASETS.items():
    in_path  = INDIR  / in_fname
    out_path = OUTDIR / out_fname

    if not in_path.exists():
        print(f"\nSkipping {label} -- input not found: {in_fname}", flush=True)
        continue

    if out_path.exists():
        print(f"\n{'='*70}", flush=True)
        print(f"SKIP {label} -- output already exists", flush=True)
        if not is_degenerate_embedding(out_path):
            continue

    print(f"\n{'='*70}", flush=True)
    print(f"Embedding: {label}", flush=True)
    print(f"  Input : {in_path}", flush=True)
    print(f"  Output: {out_path}", flush=True)

    adata = sc.read_h5ad(in_path)
    print(f"  Cells : {adata.n_obs:,}", flush=True)
    print(f"  Genes : {adata.n_vars:,}", flush=True)

    adata = swap_real_cell_counts(adata, raw_adata)
    adata = ensure_ensembl_id(adata, label)
    adata = ensure_n_counts(adata, label)
    adata = ensure_obs_id(adata)

    # Subset to Geneformer vocab before tokenizing to prevent cell dropout
    import pickle
    gf_dir = os.path.dirname(__import__("geneformer").__file__)
    with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as _f:
        _tok = pickle.load(_f)
    _gf_ids = set(_tok.keys()) - {"<cls>", "<eos>", "<mask>", "<pad>"}
    _our_ids = np.array([g.split(".")[0] for g in adata.var.index])
    _keep = np.isin(_our_ids, list(_gf_ids))
    adata = adata[:, _keep].copy()
    adata.var.index = _our_ids[_keep]
    adata.var["ensembl_id"] = adata.var.index  # re-sync after subset
    print(f"  Subset to Geneformer vocab: {_keep.sum():,} / {len(_keep):,} genes retained", flush=True)

    tok_dir     = TOKENIZE_DIR / label
    tok_dir.mkdir(exist_ok=True)
    emb_out_dir = TOKENIZE_DIR / f"{label}_embs"

    t0 = time.time()

    print(f"  Tokenizing...", flush=True)
    dataset_path = tokenize_dataset(adata, label, tok_dir)

    print(f"  Extracting embeddings (layer={EMB_LAYER}, dim={EMB_DIM})...", flush=True)
    emb_df = extract_embeddings(dataset_path, emb_out_dir, label)

    elapsed = time.time() - t0
    print(f"  Embed runtime: {elapsed/60:.1f} min  ({elapsed/adata.n_obs:.3f} s/cell)", flush=True)

    # Re-sort to original cell order; fill dropped cells with zeros
    emb_df = emb_df.set_index("obs_id") if "obs_id" in emb_df.columns else emb_df
    obs_ids = adata.obs["obs_id"].values
    missing = [i for i in obs_ids if i not in emb_df.index]
    if missing:
        print(f"  WARNING: {len(missing)} cells dropped by tokenizer -- filling with zeros.", flush=True)
    emb_matrix = np.zeros((len(obs_ids), emb_df.shape[1]), dtype=np.float32)
    present_mask = np.array([i in emb_df.index for i in obs_ids])
    emb_matrix[present_mask] = emb_df.loc[obs_ids[present_mask]].values.astype(np.float32)

    n_unique = len(np.unique(emb_matrix, axis=0))
    print(f"  Unique embedding rows: {n_unique:,}", flush=True)
    if n_unique <= MIN_UNIQUE_ROWS:
        print(f"  ERROR: degenerate output -- NOT saving.", flush=True)
        continue

    adata.obsm[EMB_KEY] = emb_matrix
    adata.write_h5ad(out_path)
    print(f"  Saved -> {out_fname}", flush=True)

    shutil.rmtree(tok_dir, ignore_errors=True)
    shutil.rmtree(emb_out_dir, ignore_errors=True)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}", flush=True)
print(f"STEP 2a ETHNICITY GENEFORMER COMPLETE", flush=True)
print(f"  Total runtime: {total_elapsed/60:.1f} min", flush=True)
for label, (_, out_fname) in DATASETS.items():
    out_path = OUTDIR / out_fname
    status   = "OK" if out_path.exists() else "MISSING"
    print(f"  [{status}] {out_fname}", flush=True)