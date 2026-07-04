#!/usr/bin/env python3
"""
STEP 3d — Embed External Validation Set with Geneformer V2-316M (ETHNICITY)

Changes vs step3d scFoundation ethnicity:
  [GF 1]  BASE path -> ethnicity_Geneformer_workflow
  [GF 2]  EMB_KEY   -> X_geneformer
  [GF 3]  OUT_FILE  -> ILD_Ethnicity_External_Validation_12500_geneformer.h5ad
  [GF 4]  Embedding logic: scFoundation (modelgenerator) -> Geneformer V2-316M
          (TranscriptomeTokenizer + EmbExtractor, same approach as step2a)

ILD_Ethnicity_External_Validation_12500.h5ad already exists from step0c.
No need to rerun step3c.
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
print(f"Script: STEP 3d ETHNICITY -- Geneformer V2-316M external validation embed", flush=True)

from geneformer import TranscriptomeTokenizer, EmbExtractor

# ============================================================
# PATHS  [GF 1]
# ============================================================

BASE = pathlib.Path(
    "/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"
)

IN_FILE  = BASE / "ILD_Ethnicity_External_Validation_12500.h5ad"
OUT_FILE = BASE / "ILD_Ethnicity_External_Validation_12500_geneformer.h5ad"   # [GF 3]

TOKENIZE_DIR = BASE / "tokenized_datasets_validation"
TOKENIZE_DIR.mkdir(exist_ok=True)

GENEFORMER_MODEL = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M"
)

# ============================================================
# CONFIG  [GF 2]
# ============================================================

EMB_KEY         = "X_geneformer"
EMB_DIM         = 1152
EMB_LAYER       = -1
FORWARD_BATCH   = 8
NPROC           = 4
MIN_UNIQUE_ROWS = 10
LABEL           = "ExternalValidation"

# ============================================================
# INPUT PREP HELPERS
# ============================================================

def ensure_ensembl_id(adata):
    candidates = ["ensembl_id", "gene_id", "ensembl", "gene_ids"]
    col = None
    for c in candidates:
        if c in adata.var.columns:
            col = c
            break
    if col is None and adata.var.index.str.startswith("ENSG").any():
        adata.var["ensembl_id"] = adata.var.index.str.split(".").str[0]
        print(f"  ensembl_id set from var index", flush=True)
        return adata
    if col is None:
        raise RuntimeError(
            f"No Ensembl ID column found.\n"
            f"  var columns: {list(adata.var.columns)}\n"
            f"  var index sample: {list(adata.var.index[:5])}"
        )
    adata.var["ensembl_id"] = adata.var[col].astype(str).str.split(".").str[0]
    print(f"  ensembl_id set from '{col}'", flush=True)
    return adata


def ensure_n_counts(adata):
    if "n_counts" not in adata.obs.columns:
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            n_counts = np.asarray(X.sum(axis=1)).ravel()
        else:
            n_counts = X.sum(axis=1)
        adata.obs["n_counts"] = n_counts.astype(np.float32)
        print(f"  n_counts computed from X (mean={n_counts.mean():.0f})", flush=True)
    else:
        print(f"  n_counts already present", flush=True)
    return adata


def ensure_obs_id(adata):
    adata.obs["obs_id"] = adata.obs.index.astype(str)
    return adata

# ============================================================
# TOKENIZE
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
        max_ncells=None,
        emb_layer=EMB_LAYER,
        emb_label=["obs_id"],
        forward_batch_size=FORWARD_BATCH,
        nproc=NPROC,
        model_version="V2",
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

if not IN_FILE.exists():
    print(f"\nERROR: input not found: {IN_FILE}", flush=True)
    print(f"ILD_Ethnicity_External_Validation_12500.h5ad should exist from step0c.", flush=True)
    sys.exit(1)

if OUT_FILE.exists():
    print(f"\n{'='*70}", flush=True)
    print(f"SKIP -- output already exists: {OUT_FILE.name}", flush=True)
    if not is_degenerate_embedding(OUT_FILE):
        print("Nothing to do.", flush=True)
        sys.exit(0)

print(f"\n{'='*70}", flush=True)
print(f"Embedding: external validation set", flush=True)
print(f"  Input : {IN_FILE}", flush=True)
print(f"  Output: {OUT_FILE}", flush=True)

total_start = time.time()

adata = sc.read_h5ad(IN_FILE)
print(f"  Cells : {adata.n_obs:,}", flush=True)
print(f"  Genes : {adata.n_vars:,}", flush=True)

adata = ensure_ensembl_id(adata)
adata = ensure_n_counts(adata)
adata = ensure_obs_id(adata)

tok_dir     = TOKENIZE_DIR / LABEL
tok_dir.mkdir(exist_ok=True)
emb_out_dir = TOKENIZE_DIR / f"{LABEL}_embs"

t0 = time.time()

print(f"  Tokenizing...", flush=True)
dataset_path = tokenize_dataset(adata, LABEL, tok_dir)

print(f"  Extracting embeddings (layer={EMB_LAYER}, dim={EMB_DIM})...", flush=True)
emb_df = extract_embeddings(dataset_path, emb_out_dir, LABEL)

elapsed = time.time() - t0
print(f"  Embed runtime: {elapsed/60:.1f} min  ({elapsed/adata.n_obs:.3f} s/cell)", flush=True)

emb_df     = emb_df.set_index("obs_id") if "obs_id" in emb_df.columns else emb_df
n_before   = adata.n_obs
adata      = adata[adata.obs["obs_id"].isin(emb_df.index)].copy()
if adata.n_obs < n_before:
    print(f"  WARNING: {n_before - adata.n_obs} cells dropped (tokenizer dropout)", flush=True)
emb_matrix = emb_df.loc[adata.obs["obs_id"]].values.astype(np.float32)

n_unique = len(np.unique(emb_matrix, axis=0))
print(f"  Unique embedding rows: {n_unique:,}", flush=True)
if n_unique <= MIN_UNIQUE_ROWS:
    print(f"  ERROR: degenerate output -- NOT saving.", flush=True)
    sys.exit(1)

adata.obsm[EMB_KEY] = emb_matrix
adata.write_h5ad(OUT_FILE)
print(f"  Saved -> {OUT_FILE.name}", flush=True)

shutil.rmtree(tok_dir, ignore_errors=True)
shutil.rmtree(emb_out_dir, ignore_errors=True)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}", flush=True)
print(f"STEP 3d ETHNICITY GENEFORMER COMPLETE", flush=True)
print(f"  Total runtime: {total_elapsed/60:.1f} min", flush=True)
print(f"  Output: {OUT_FILE}", flush=True)