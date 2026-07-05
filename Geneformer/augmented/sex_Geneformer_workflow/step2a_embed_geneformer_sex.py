#!/usr/bin/env python3
"""
STEP 2a — Embed datasets with Geneformer V2-316M (SEX)

Changes vs ethnicity version:
  [SEX 1] BASE path      : ethnicity_Geneformer_workflow -> sex_Geneformer_workflow
  [SEX 2] OUTPUT_BASE    : ILD_Ethnicity_Pilot           -> ILD_Sex_Pilot
  [SEX 3] DATASETS       : sex filenames (1413Each, 586Each, Proportional_1999)
  [SEX 4] RAW_COUNTS     : _ETHNICITY.h5ad              -> _SEX.h5ad
  [SEX 5] VALIDATION     : ILD_Sex_External_Validation_5000.h5ad added
  [SEX 6] Print strings updated for sex context

All embedding logic preserved verbatim from ethnicity version.
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
print(f"Script: STEP 2a SEX -- Geneformer V2-316M embed", flush=True)

from geneformer import TranscriptomeTokenizer, EmbExtractor

# ============================================================
# PATHS  [SEX 1]
# ============================================================

BASE = pathlib.Path(
    "/data/Geneformer/augmented/sex_Geneformer_workflow"
)
INDIR  = BASE
OUTDIR = BASE
OUTDIR.mkdir(exist_ok=True)

TOKENIZE_DIR = BASE / "tokenized_datasets"
TOKENIZE_DIR.mkdir(exist_ok=True)

GENEFORMER_MODEL = pathlib.Path(
    "/data/Geneformer/geneformer_repo/Geneformer-V2-316M"
)

OUTPUT_BASE     = "ILD_Sex_Pilot"                                    # [SEX 2]
RAW_COUNTS_PATH = BASE / "InterstitialLungDisease_RawCounts_SEX.h5ad"  # [SEX 4]

# ============================================================
# DATASETS  [SEX 3]
# Filenames match step0b output:
#   TARGET_PER_BIN = 1413  |  DOWN_TARGET = 586  |  Proportional = 1999
# ============================================================

DATASETS = {
    "Proportional_1999": (
        f"{OUTPUT_BASE}_Proportional_1999_SEX.h5ad",
        f"{OUTPUT_BASE}_Proportional_1999_SEX_geneformer.h5ad",
    ),
    "BalancedAugmented_1413Each": (
        f"{OUTPUT_BASE}_BalancedAugmented_1413Each_SEX.h5ad",
        f"{OUTPUT_BASE}_BalancedAugmented_1413Each_SEX_geneformer.h5ad",
    ),
    "BalancedUpsampled_1413Each": (
        f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_SEX.h5ad",
        f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_SEX_geneformer.h5ad",
    ),
    "Downsampled_586Each": (
        f"{OUTPUT_BASE}_Downsampled_586Each_SEX.h5ad",
        f"{OUTPUT_BASE}_Downsampled_586Each_SEX_geneformer.h5ad",
    ),
    "ExternalValidation_5000": (                                     # [SEX 5]
        "ILD_Sex_External_Validation_5000.h5ad",
        "ILD_Sex_External_Validation_5000_geneformer.h5ad",
    ),
}

# ============================================================
# CONFIG
# ============================================================

EMB_KEY         = "X_geneformer"
EMB_DIM         = 1152
EMB_LAYER       = -1
FORWARD_BATCH   = 8
NPROC           = 4
MIN_UNIQUE_ROWS = 10

# ============================================================
# HELPERS (preserved verbatim from ethnicity version)
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
            f"  var index sample: {list(adata.var.index[:5])}"
        )
    adata.var["ensembl_id"] = adata.var[col].astype(str).str.split(".").str[0]
    print(f"  [GF 5] ensembl_id set from '{col}' (stripped version suffix)", flush=True)
    return adata


def ensure_n_counts(adata, label):
    if "n_counts" not in adata.obs.columns:
        import scipy.sparse as sp
        X = adata.X
        n_counts = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        adata.obs["n_counts"] = n_counts.astype(np.float32)
        print(f"  [GF 5] n_counts computed from X (mean={n_counts.mean():.0f})", flush=True)
    else:
        print(f"  [GF 5] n_counts already present", flush=True)
    return adata


def ensure_obs_id(adata):
    adata.obs["obs_id"] = adata.obs.index.astype(str)
    return adata


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
    tk.tokenize_data(str(tok_dir), str(tok_dir), dataset_name, file_format="h5ad")
    dataset_path = tok_dir / f"{dataset_name}.dataset"
    if not dataset_path.exists():
        raise RuntimeError(f"Tokenization failed: {dataset_path} not found")
    tmp_h5ad.unlink()
    print(f"  Tokenized -> {dataset_path}", flush=True)
    return dataset_path


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
    return embex.extract_embs(
        str(GENEFORMER_MODEL),
        str(dataset_path),
        str(emb_out_dir),
        label,
    )


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
    raise RuntimeError(f"Geneformer model not found: {GENEFORMER_MODEL}")

total_start = time.time()

print(f"Loading raw counts for real-cell swap...", flush=True)
raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
print(f"  Raw counts: {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes", flush=True)

import pickle
gf_dir  = os.path.dirname(__import__("geneformer").__file__)
with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as _f:
    _tok = pickle.load(_f)
GF_IDS = set(_tok.keys()) - {"<cls>", "<eos>", "<mask>", "<pad>"}

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

    adata = sc.read_h5ad(in_path)
    print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}", flush=True)

    adata = swap_real_cell_counts(adata, raw_adata)
    adata = ensure_ensembl_id(adata, label)
    adata = ensure_n_counts(adata, label)
    adata = ensure_obs_id(adata)

    our_ids = np.array([g.split(".")[0] for g in adata.var.index])
    keep    = np.isin(our_ids, list(GF_IDS))
    adata   = adata[:, keep].copy()
    adata.var.index          = our_ids[keep]
    adata.var["ensembl_id"]  = adata.var.index
    print(f"  Vocab subset: {keep.sum():,} / {len(keep):,} genes retained", flush=True)

    tok_dir     = TOKENIZE_DIR / label
    tok_dir.mkdir(exist_ok=True)
    emb_out_dir = TOKENIZE_DIR / f"{label}_embs"

    t0 = time.time()
    print(f"  Tokenizing...", flush=True)
    dataset_path = tokenize_dataset(adata, label, tok_dir)
    print(f"  Extracting embeddings...", flush=True)
    emb_df = extract_embeddings(dataset_path, emb_out_dir, label)
    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed/60:.1f} min ({elapsed/adata.n_obs:.3f} s/cell)", flush=True)

    emb_df   = emb_df.set_index("obs_id") if "obs_id" in emb_df.columns else emb_df
    obs_ids  = adata.obs["obs_id"].values
    missing  = [i for i in obs_ids if i not in emb_df.index]
    if missing:
        print(f"  WARNING: {len(missing)} cells dropped by tokenizer -- filling with zeros.", flush=True)

    emb_matrix   = np.zeros((len(obs_ids), emb_df.shape[1]), dtype=np.float32)
    present_mask = np.array([i in emb_df.index for i in obs_ids])
    emb_matrix[present_mask] = emb_df.loc[obs_ids[present_mask]].values.astype(np.float32)

    n_unique = len(np.unique(emb_matrix, axis=0))
    print(f"  Unique embedding rows: {n_unique:,}", flush=True)
    if n_unique <= MIN_UNIQUE_ROWS:
        print(f"  ERROR: degenerate output -- NOT saving.", flush=True)
        continue

    adata.obsm[EMB_KEY] = emb_matrix

    # Restore disease/sex/ethnicity from source h5ad (tokenizer corrupts categorical encoding)
    print(f"  Restoring obs labels from source...", flush=True)
    src_obs = sc.read_h5ad(in_path).obs
    for col in ["disease", "sex", "self_reported_ethnicity", "cell_type", "donor_id"]:
        if col in src_obs.columns:
            adata.obs[col] = src_obs.loc[adata.obs.index, col].astype(str).values

    adata.write_h5ad(out_path)
    print(f"  Saved -> {out_fname}", flush=True)

    shutil.rmtree(tok_dir, ignore_errors=True)
    shutil.rmtree(emb_out_dir, ignore_errors=True)

total_elapsed = time.time() - total_start
print(f"\n{'='*70}", flush=True)
print(f"STEP 2a SEX GENEFORMER COMPLETE", flush=True)
print(f"  Total runtime: {total_elapsed/60:.1f} min", flush=True)
for label, (_, out_fname) in DATASETS.items():
    out_path = OUTDIR / out_fname
    status   = "OK" if out_path.exists() else "MISSING"
    print(f"  [{status}] {out_fname}", flush=True)
