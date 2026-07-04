#!/usr/bin/env python3
"""STEP 2a — Embed AGE datasets with Geneformer V2-316M"""
import sys, os, shutil, time, pathlib
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONNOUSERSITE"] = "1"

print(f"Python: {sys.executable}", flush=True)
print(f"Script: STEP 2a AGE -- Geneformer V2-316M embed", flush=True)

from geneformer import TranscriptomeTokenizer, EmbExtractor

BASE         = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow")
TOKENIZE_DIR = BASE / "tokenized_datasets"
TOKENIZE_DIR.mkdir(exist_ok=True)

GENEFORMER_MODEL = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M")
OUTPUT_BASE      = "ILD_Age_Pilot"
RAW_COUNTS_PATH  = BASE / "InterstitialLungDisease_RawCounts_AGE.h5ad"

DATASETS = {
    "Proportional_2495": (
        f"{OUTPUT_BASE}_Proportional_2495_AGE.h5ad",
        f"{OUTPUT_BASE}_Proportional_2495_AGE_geneformer.h5ad",
    ),
    "BalancedAugmented_1262Each": (
        f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE.h5ad",
        f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE_geneformer.h5ad",
    ),
    "BalancedUpsampled_1262Each": (
        f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE.h5ad",
        f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE_geneformer.h5ad",
    ),
    "Downsampled_25Each": (
        f"{OUTPUT_BASE}_Downsampled_25Each_AGE.h5ad",
        f"{OUTPUT_BASE}_Downsampled_25Each_AGE_geneformer.h5ad",
    ),
    "ExternalValidation_10500": (
        "ILD_Age_External_Validation_10500.h5ad",
        "ILD_Age_External_Validation_10500_geneformer.h5ad",
    ),
}

EMB_KEY       = "X_geneformer"
EMB_DIM       = 1152
EMB_LAYER     = -1
FORWARD_BATCH = 8
NPROC         = 4
MIN_UNIQUE_ROWS = 10

# ── helpers ───────────────────────────────────────────────────────────────────

def ensure_ensembl_id(adata, label):
    candidates = ["ensembl_id","gene_id","ensembl","gene_ids"]
    col = None
    for c in candidates:
        if c in adata.var.columns: col = c; break
    if col is not None:
        adata.var["ensembl_id"] = adata.var[col].astype(str).str.replace(r'\.\d+$','',regex=True)
        print(f"  [GF 5] ensembl_id from var['{col}']", flush=True)
    else:
        adata.var["ensembl_id"] = adata.var.index.astype(str).str.replace(r'\.\d+$','',regex=True)
        print(f"  [GF 5] ensembl_id from var.index", flush=True)
    return adata

def ensure_n_counts(adata, label):
    if "n_counts" not in adata.obs.columns:
        X = adata.X
        adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).flatten() if sp.issparse(X) else X.sum(axis=1)
        print(f"  [GF 5] n_counts computed from X", flush=True)
    else:
        print(f"  [GF 5] n_counts already present", flush=True)
    return adata

def ensure_obs_id(adata):
    adata.obs["obs_id"] = adata.obs.index.astype(str)
    return adata

def swap_real_cell_counts(adata, raw_adata):
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
    if sp.issparse(raw_X): raw_X = raw_X.toarray().astype(np.float32)
    else: raw_X = raw_X.astype(np.float32)
    aligned                = np.zeros((len(available), adata.n_vars), dtype=np.float32)
    aligned[:, valid_cols] = raw_X[:, col_map[valid_cols]]
    X = adata.X
    if sp.issparse(X): X = X.toarray().astype(np.float32)
    else: X = X.astype(np.float32).copy()
    avail_pos    = np.where(adata.obs.index.isin(available))[0]
    X[avail_pos] = aligned
    adata.X      = sp.csr_matrix(X)
    print(f"  [SWAP] {len(available):,}/{real_mask.sum():,} real cells -> full-transcriptome counts", flush=True)
    return adata

def is_degenerate_embedding(path):
    try:
        ad  = sc.read_h5ad(path)
        emb = ad.obsm.get(EMB_KEY)
        if emb is None: return True
        return len(np.unique(emb, axis=0)) < MIN_UNIQUE_ROWS
    except Exception:
        return True

def tokenize_dataset(adata, label, tok_dir):
    tmp_h5ad     = tok_dir / f"{label}_tmp.h5ad"
    dataset_name = label
    adata.write_h5ad(tmp_h5ad)
    print(f"  Wrote temp h5ad: {tmp_h5ad}", flush=True)
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"obs_id": "obs_id", "cell_type": "cell_type",
                               "age_bin_10yr": "age_bin_10yr"},
        nproc=NPROC, model_version="V2",
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
        model_type="Pretrained", num_classes=0,
        emb_mode="cell", cell_emb_style="mean_pool",
        emb_layer=EMB_LAYER, emb_label=["obs_id"],
        forward_batch_size=FORWARD_BATCH, nproc=NPROC,
        model_version="V2", max_ncells=None,
    )
    emb_out_dir.mkdir(exist_ok=True)
    emb_df = embex.extract_embs(
        str(GENEFORMER_MODEL),
        str(dataset_path),
        str(emb_out_dir),
        label,
    )
    return emb_df

# ── load GF vocab ─────────────────────────────────────────────────────────────
print("Loading Geneformer token dictionary...", flush=True)
import pickle
gf_dir = os.path.dirname(__import__("geneformer").__file__)
with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as _f:
    _tok = pickle.load(_f)
GF_IDS = set(_tok.keys()) - {"<cls>", "<eos>", "<mask>", "<pad>"}
print(f"  Vocab size: {len(GF_IDS):,} genes", flush=True)

# ── load raw counts once ──────────────────────────────────────────────────────
print(f"Loading raw counts from {RAW_COUNTS_PATH}...", flush=True)
raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
print(f"  Raw: {raw_adata.n_obs:,} cells x {raw_adata.n_vars:,} genes", flush=True)

# ── main loop ─────────────────────────────────────────────────────────────────
for label, (in_fname, out_fname) in DATASETS.items():
    in_path  = BASE / in_fname
    out_path = BASE / out_fname

    if not in_path.exists():
        print(f"\nSkipping {label} -- input not found: {in_fname}", flush=True)
        continue
    if out_path.exists() and not is_degenerate_embedding(out_path):
        print(f"\nSKIP {label} -- output already exists", flush=True)
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
    adata.var.index         = our_ids[keep]
    adata.var["ensembl_id"] = adata.var.index
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

    emb_df  = emb_df.set_index("obs_id") if "obs_id" in emb_df.columns else emb_df
    obs_ids = adata.obs["obs_id"].values
    missing = [i for i in obs_ids if i not in emb_df.index]
    if missing:
        print(f"  WARNING: {len(missing)} cells missing from embedding -- filling with zeros", flush=True)
        emb_dim = emb_df.shape[1]
        zero_df = pd.DataFrame(
            np.zeros((len(missing), emb_dim), dtype=np.float32),
            index=missing,
            columns=emb_df.columns,
        )
        emb_df = pd.concat([emb_df, zero_df])

    emb_mat = emb_df.loc[obs_ids].values.astype(np.float32)
    adata.obsm[EMB_KEY] = emb_mat

    n_unique = len(np.unique(emb_mat, axis=0))
    print(f"  Unique embedding rows: {n_unique} / {adata.n_obs}", flush=True)
    if n_unique < MIN_UNIQUE_ROWS:
        print(f"  WARNING: degenerate embedding ({n_unique} unique rows)", flush=True)

    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}", flush=True)
    shutil.rmtree(tok_dir, ignore_errors=True)
    shutil.rmtree(emb_out_dir, ignore_errors=True)

print("\nSTEP 2a AGE complete.", flush=True)
