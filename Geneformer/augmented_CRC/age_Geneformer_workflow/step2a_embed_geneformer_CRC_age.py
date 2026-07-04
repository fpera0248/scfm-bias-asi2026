#!/usr/bin/env python3
"""STEP 2a — Embed datasets with Geneformer V2-316M (CRC AGE)"""
import sys, os, shutil, time, pickle
import scanpy as sc
import numpy as np
import pathlib
import scipy.sparse as sp
from geneformer import TranscriptomeTokenizer, EmbExtractor

BASE   = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/age_Geneformer_workflow")
OUTDIR = BASE
OUTDIR.mkdir(exist_ok=True)
TOKENIZE_DIR = BASE / "tokenized_datasets"
TOKENIZE_DIR.mkdir(exist_ok=True)
GENEFORMER_MODEL = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M")
RAW_COUNTS_PATH  = BASE / "ColorectalCancer_RawCounts_AGE.h5ad"
OUTPUT_BASE      = "CRC_Age_Pilot"

DATASETS = {
    "Proportional_2498":          (f"{OUTPUT_BASE}_Proportional_2498_AGE.h5ad",          f"{OUTPUT_BASE}_Proportional_2498_AGE_geneformer.h5ad"),
    "BalancedAugmented_650Each":  (f"{OUTPUT_BASE}_BalancedAugmented_650Each_AGE.h5ad",  f"{OUTPUT_BASE}_BalancedAugmented_650Each_AGE_geneformer.h5ad"),
    "BalancedUpsampled_650Each":  (f"{OUTPUT_BASE}_BalancedUpsampled_650Each_AGE.h5ad",  f"{OUTPUT_BASE}_BalancedUpsampled_650Each_AGE_geneformer.h5ad"),
    "Downsampled_124Each":         (f"{OUTPUT_BASE}_Downsampled_124Each_AGE.h5ad",          f"{OUTPUT_BASE}_Downsampled_124Each_AGE_geneformer.h5ad"),
    "ExternalValidation_9402":    ("CRC_Age_External_Validation_9402.h5ad",               "CRC_Age_External_Validation_9402_geneformer.h5ad"),
}

EMB_KEY = "X_geneformer"; EMB_LAYER = -1; FORWARD_BATCH = 8; NPROC = 4; MIN_UNIQUE_ROWS = 10

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_ensembl_id(adata, label):
    candidates = ["ensembl_id", "gene_id", "ensembl", "gene_ids"]
    for c in candidates:
        if c in adata.var.columns:
            adata.var["ensembl_id"] = adata.var[c].astype(str).str.split(".").str[0]
            return adata
    if adata.var.index.str.startswith("ENSG").any():
        adata.var["ensembl_id"] = adata.var.index.str.split(".").str[0]
        return adata
    raise RuntimeError(f"No Ensembl ID column found in {label}.")

def ensure_n_counts(adata, label):
    if "n_counts" not in adata.obs.columns:
        X = adata.X
        n_counts = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        adata.obs["n_counts"] = n_counts.astype(np.float32)
    return adata

def ensure_obs_id(adata):
    adata.obs["obs_id"] = adata.obs.index.astype(str); return adata

def swap_real_cell_counts(adata, raw_adata):
    if "source" not in adata.obs.columns: return adata
    real_mask     = (adata.obs["source"] == "real").values
    real_barcodes = adata.obs.index[real_mask]
    available     = real_barcodes[real_barcodes.isin(raw_adata.obs.index)]
    if len(available) == 0: return adata
    adata_genes = np.array([g.split(".")[0] for g in adata.var.index])
    raw_genes   = np.array([g.split(".")[0] for g in raw_adata.var.index])
    raw_gene_map = {g: i for i, g in enumerate(raw_genes)}
    col_map     = np.array([raw_gene_map.get(g, -1) for g in adata_genes])
    valid_cols  = col_map >= 0
    raw_sub = raw_adata[available, :]
    raw_X   = raw_sub.X.toarray().astype(np.float32) if sp.issparse(raw_sub.X) else raw_sub.X.astype(np.float32)
    aligned = np.zeros((len(available), adata.n_vars), dtype=np.float32)
    aligned[:, valid_cols] = raw_X[:, col_map[valid_cols]]
    X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else adata.X.astype(np.float32).copy()
    avail_pos    = np.where(adata.obs.index.isin(available))[0]
    X[avail_pos] = aligned
    adata.X      = sp.csr_matrix(X)
    log(f"  [SWAP] {len(available):,}/{real_mask.sum():,} real cells -> full-transcriptome counts")
    return adata

def tokenize_dataset(adata, label, tok_dir):
    tmp_h5ad = tok_dir / f"{label}_tmp.h5ad"
    adata.write_h5ad(tmp_h5ad)
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"obs_id": "obs_id", "cell_type": "cell_type"},
        nproc=NPROC, model_version="V2")
    tk.tokenize_data(str(tok_dir), str(tok_dir), label, file_format="h5ad")
    dataset_path = tok_dir / f"{label}.dataset"
    if not dataset_path.exists(): raise RuntimeError(f"Tokenization failed: {dataset_path}")
    tmp_h5ad.unlink()
    return dataset_path

def extract_embeddings(dataset_path, emb_out_dir, label):
    embex = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cell",
        cell_emb_style="mean_pool", emb_layer=EMB_LAYER, emb_label=["obs_id"],
        forward_batch_size=FORWARD_BATCH, nproc=NPROC, model_version="V2", max_ncells=None)
    emb_out_dir.mkdir(exist_ok=True)
    return embex.extract_embs(str(GENEFORMER_MODEL), str(dataset_path), str(emb_out_dir), label)

def is_degenerate_embedding(out_path):
    try:
        ad = sc.read_h5ad(out_path)
        if EMB_KEY not in ad.obsm: return True
        n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
        if n_unique <= MIN_UNIQUE_ROWS: return True
        log(f"  OK: existing embedding valid ({n_unique} unique rows) -- skipping.")
        return False
    except: return True

if not GENEFORMER_MODEL.exists(): raise RuntimeError(f"Model not found: {GENEFORMER_MODEL}")
log("Loading raw counts for real-cell swap...")
raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
log(f"  Raw counts: {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes")

gf_dir = os.path.dirname(__import__("geneformer").__file__)
with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as _f:
    _tok = pickle.load(_f)
GF_IDS = set(_tok.keys()) - {"<cls>", "<eos>", "<mask>", "<pad>"}

for label, (in_fname, out_fname) in DATASETS.items():
    in_path  = BASE / in_fname
    out_path = OUTDIR / out_fname
    if not in_path.exists(): log(f"\nSkipping {label} -- input not found: {in_fname}"); continue
    if out_path.exists() and not is_degenerate_embedding(out_path): continue
    log(f"\n{'='*70}\nEmbedding: {label}")
    adata = sc.read_h5ad(in_path)
    log(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")
    adata = swap_real_cell_counts(adata, raw_adata)
    adata = ensure_ensembl_id(adata, label)
    adata = ensure_n_counts(adata, label)
    adata = ensure_obs_id(adata)
    our_ids = np.array([g.split(".")[0] for g in adata.var.index])
    keep    = np.isin(our_ids, list(GF_IDS))
    adata   = adata[:, keep].copy()
    adata.var.index         = our_ids[keep]
    adata.var["ensembl_id"] = adata.var.index
    log(f"  Vocab subset: {keep.sum():,} / {len(keep):,} genes retained")
    tok_dir     = TOKENIZE_DIR / label; tok_dir.mkdir(exist_ok=True)
    emb_out_dir = TOKENIZE_DIR / f"{label}_embs"
    t0 = time.time()
    dataset_path = tokenize_dataset(adata, label, tok_dir)
    emb_df       = extract_embeddings(dataset_path, emb_out_dir, label)
    log(f"  Runtime: {(time.time()-t0)/60:.1f} min")
    emb_df  = emb_df.set_index("obs_id") if "obs_id" in emb_df.columns else emb_df
    obs_ids = adata.obs["obs_id"].values
    missing = [i for i in obs_ids if i not in emb_df.index]
    if missing: log(f"  WARNING: {len(missing)} cells dropped by tokenizer -- filling with zeros.")
    emb_matrix   = np.zeros((len(obs_ids), emb_df.shape[1]), dtype=np.float32)
    present_mask = np.array([i in emb_df.index for i in obs_ids])
    emb_matrix[present_mask] = emb_df.loc[obs_ids[present_mask]].values.astype(np.float32)
    n_unique = len(np.unique(emb_matrix, axis=0))
    log(f"  Unique embedding rows: {n_unique:,}")
    if n_unique <= MIN_UNIQUE_ROWS: log(f"  ERROR: degenerate output -- NOT saving."); continue
    adata.obsm[EMB_KEY] = emb_matrix
    # Restore disease/age labels from source h5ad (tokenizer corrupts categorical encoding)
    src_obs = sc.read_h5ad(in_path).obs
    for col in ["disease", "age_bin_10yr", "self_reported_ethnicity", "sex", "cell_type", "donor_id"]:
        if col in src_obs.columns:
            adata.obs[col] = src_obs.loc[adata.obs.index, col].astype(str).values
    adata.write_h5ad(out_path)
    log(f"  Saved -> {out_fname}")
    shutil.rmtree(tok_dir, ignore_errors=True); shutil.rmtree(emb_out_dir, ignore_errors=True)

log("STEP 2a CRC AGE GENEFORMER COMPLETE")
for label, (_, out_fname) in DATASETS.items():
    status = "OK" if (OUTDIR/out_fname).exists() else "MISSING"
    log(f"  [{status}] {out_fname}")
