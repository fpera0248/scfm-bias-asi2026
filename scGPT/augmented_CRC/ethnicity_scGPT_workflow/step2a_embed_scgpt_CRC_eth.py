#!/usr/bin/env python3
"""STEP 2a — scGPT Cell Embedding (CRC AGE)"""
import pathlib, time, warnings
import numpy as np
import scanpy as sc
import scipy.sparse as sp
warnings.filterwarnings("ignore")

BASE        = pathlib.Path("/oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/ethnicity_scGPT_workflow")
OUTDIR      = BASE / "step2a_embeddings"
SCGPT_DIR   = pathlib.Path("/oscar/home/fperalta/data/fperalta/scGPT")
OUTDIR.mkdir(exist_ok=True)
CHECKPOINT  = SCGPT_DIR / "scGPT_human"
GENE_INFO   = SCGPT_DIR / "gene_info.csv"
RAW_COUNTS_PATH = BASE / "ColorectalCancer_RawCounts_ETH.h5ad"
OUTPUT_BASE = "CRC_Eth_Pilot"

DATASETS = {
    "Proportional_2497":           BASE / f"{OUTPUT_BASE}_Proportional_2497_ETHNICITY.h5ad",
    "BalancedAugmented_1880Each":  BASE / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETHNICITY.h5ad",
    "BalancedUpsampled_1880Each":  BASE / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETHNICITY.h5ad",
    "Downsampled_48Each":          BASE / f"{OUTPUT_BASE}_Downsampled_48Each_ETHNICITY.h5ad",
    "ExternalValidation_8572":     BASE / "CRC_Eth_External_Validation_8572.h5ad",
}
OUTPUT_NAMES = {
    "Proportional_2497":           f"{OUTPUT_BASE}_Proportional_2497_ETH_scgpt.h5ad",
    "BalancedAugmented_1880Each":  f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETH_scgpt.h5ad",
    "BalancedUpsampled_1880Each":  f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETH_scgpt.h5ad",
    "Downsampled_48Each":          f"{OUTPUT_BASE}_Downsampled_48Each_ETH_scgpt.h5ad",
    "ExternalValidation_8572":     "CRC_Eth_External_Validation_8572_scgpt.h5ad",
}

EMB_KEY = "X_scGPT"; MIN_UNIQUE_EMB = 10

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_gene_map(p):
    import pandas as pd
    df = pd.read_csv(p)
    return dict(zip(df["feature_id"], df["feature_name"]))

def swap_real_cell_counts(adata, raw_adata):
    if "source" not in adata.obs.columns: return adata
    real_mask     = (adata.obs["source"] == "real").values
    real_barcodes = adata.obs.index[real_mask]
    available     = real_barcodes[real_barcodes.isin(raw_adata.obs.index)]
    if len(available) == 0: return adata
    adata_genes  = np.array([g.split(".")[0] for g in adata.var.index])
    raw_genes    = np.array([g.split(".")[0] for g in raw_adata.var.index])
    raw_gene_map = {g: i for i, g in enumerate(raw_genes)}
    col_map      = np.array([raw_gene_map.get(g, -1) for g in adata_genes])
    valid_cols   = col_map >= 0
    raw_sub  = raw_adata[available, :]
    raw_X    = raw_sub.X.toarray().astype(np.float32) if sp.issparse(raw_sub.X) else raw_sub.X.astype(np.float32)
    aligned  = np.zeros((len(available), adata.n_vars), dtype=np.float32)
    aligned[:, valid_cols] = raw_X[:, col_map[valid_cols]]
    X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else adata.X.astype(np.float32).copy()
    avail_pos    = np.where(adata.obs.index.isin(available))[0]
    X[avail_pos] = aligned
    adata.X      = sp.csr_matrix(X)
    log(f"  [SWAP] {len(available):,}/{real_mask.sum():,} real cells -> full-transcriptome counts")
    return adata

def embed_dataset(label, in_path, out_path, raw_adata, gene_map, vocab_genes):
    import scgpt
    log(f"\n{'='*70}\nEmbedding: {label}")
    adata = sc.read_h5ad(in_path)
    log(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")
    adata = swap_real_cell_counts(adata, raw_adata)
    our_ids = np.array([g.split(".")[0] for g in adata.var.index])
    hgnc    = np.array([gene_map.get(g, "") for g in our_ids])
    keep    = np.isin(hgnc, list(vocab_genes)) & (hgnc != "")
    adata   = adata[:, keep].copy()
    adata.var.index = hgnc[keep]
    log(f"  Vocab subset: {keep.sum():,} / {len(keep):,} genes retained")
    sc.pp.filter_cells(adata, min_counts=1)
    log(f"  After filter_cells: {adata.n_obs:,} cells")
    if adata.n_obs == 0: log("  ERROR: No cells remain -- skipping."); return
    t0 = time.time()
    embeddings = scgpt.tasks.embed_data(
        adata, model_dir=str(CHECKPOINT), gene_col="index",
        obs_to_save=list(adata.obs.columns), batch_size=64, return_new_adata=True)
    log(f"  Runtime: {(time.time()-t0)/60:.1f} min")
    emb_matrix = np.array(embeddings.X, dtype=np.float32)
    n_unique   = len(np.unique(emb_matrix, axis=0))
    log(f"  Unique rows: {n_unique}  Zero-vector: {(np.abs(emb_matrix).sum(axis=1)==0).sum()}")
    if n_unique <= MIN_UNIQUE_EMB: log("  ERROR: degenerate -- NOT saving."); return
    orig     = sc.read_h5ad(in_path)
    orig     = swap_real_cell_counts(orig, raw_adata)
    emb_full = np.zeros((orig.n_obs, emb_matrix.shape[1]), dtype=np.float32)
    dst_idx  = orig.obs_names.get_indexer(embeddings.obs_names)
    valid    = dst_idx >= 0
    emb_full[dst_idx[valid]] = emb_matrix[valid]
    orig.obsm[EMB_KEY] = emb_full
    orig.write_h5ad(out_path)
    log(f"  Saved -> {out_path.name}")

def main():
    import json, scgpt
    log("STEP 2a -- scGPT Cell Embedding (CRC AGE)")
    gene_map = load_gene_map(GENE_INFO)
    with open(CHECKPOINT / "vocab.json") as f: vocab = json.load(f)
    vocab_genes = set(vocab.keys()) - {"<pad>", "<cls>", "<eoc>"}
    log(f"  Vocab size: {len(vocab_genes):,} genes")
    log(f"Loading raw counts from {RAW_COUNTS_PATH.name}...")
    raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
    log(f"  {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes")
    for label, in_path in DATASETS.items():
        out_path = OUTDIR / OUTPUT_NAMES[label]
        if not in_path.exists(): log(f"\nSkipping {label} -- not found"); continue
        if out_path.exists():
            ad_check = sc.read_h5ad(out_path)
            if EMB_KEY in ad_check.obsm:
                n_unique = len(np.unique(np.array(ad_check.obsm[EMB_KEY]), axis=0))
                if n_unique > MIN_UNIQUE_EMB:
                    log(f"\nSkipping {label} -- valid embedding exists ({n_unique} unique rows)")
                    continue
        embed_dataset(label, in_path, out_path, raw_adata, gene_map, vocab_genes)
    log("\nSTEP 2a CRC ETH scGPT COMPLETE")
    for label, name in OUTPUT_NAMES.items():
        status = "OK" if (OUTDIR/name).exists() else "MISSING"
        log(f"  [{status}] {name}")

if __name__ == "__main__":
    main()
