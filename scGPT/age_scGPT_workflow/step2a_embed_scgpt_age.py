#!/usr/bin/env python3
"""
STEP 2a — scGPT Cell Embedding (AGE)

Changes vs sex scGPT version:
  [AGE 1] BASE/INDIR     -> age_scGPT_workflow
  [AGE 2] OUTDIR         -> age_scGPT_workflow/step2a_embeddings
  [AGE 3] RAW_COUNTS     -> InterstitialLungDisease_RawCounts_AGE.h5ad
  [AGE 4] DATASETS       -> age filenames (1262Each, 25Each, 2495)
  [AGE 5] VALIDATION     -> ILD_Age_External_Validation_10500.h5ad
  [AGE 6] EMB_KEY        -> X_scGPT
"""

import pathlib
import time
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

BASE        = pathlib.Path("/data/scGPT/age_scGPT_workflow")  # [AGE 1]
OUTDIR      = BASE / "step2a_embeddings"                                                    # [AGE 2]
SCGPT_DIR   = pathlib.Path("/data/scGPT")
OUTDIR.mkdir(exist_ok=True)

CHECKPOINT  = SCGPT_DIR / "scGPT_human"
GENE_INFO   = SCGPT_DIR / "gene_info.csv"

RAW_COUNTS_PATH = BASE / "InterstitialLungDisease_RawCounts_AGE.h5ad"   # [AGE 3]

OUTPUT_BASE = "ILD_Age_Pilot"

DATASETS = {                                                                    # [AGE 4]
    "Proportional_2495":          BASE / f"{OUTPUT_BASE}_Proportional_2495_AGE.h5ad",
    "BalancedAugmented_1262Each": BASE / f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE.h5ad",
    "BalancedUpsampled_1262Each": BASE / f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE.h5ad",
    "Downsampled_25Each":         BASE / f"{OUTPUT_BASE}_Downsampled_25Each_AGE.h5ad",
    "ExternalValidation_10500":   BASE / "ILD_Age_External_Validation_10500.h5ad",          # [AGE 5]
}

OUTPUT_NAMES = {
    "Proportional_2495":          f"{OUTPUT_BASE}_Proportional_2495_AGE_scgpt.h5ad",
    "BalancedAugmented_1262Each": f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE_scgpt.h5ad",
    "BalancedUpsampled_1262Each": f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE_scgpt.h5ad",
    "Downsampled_25Each":         f"{OUTPUT_BASE}_Downsampled_25Each_AGE_scgpt.h5ad",
    "ExternalValidation_10500":   "ILD_Age_External_Validation_10500_scgpt.h5ad",
}

EMB_KEY        = "X_scGPT"                                              # [AGE 6]
MIN_UNIQUE_EMB = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_gene_map(gene_info_path):
    import pandas as pd
    df = pd.read_csv(gene_info_path)
    return dict(zip(df["feature_id"], df["feature_name"]))


def swap_real_cell_counts(adata, raw_adata):
    if "source" not in adata.obs.columns:
        return adata
    real_mask     = (adata.obs["source"] == "real").values
    real_barcodes = adata.obs.index[real_mask]
    available     = real_barcodes[real_barcodes.isin(raw_adata.obs.index)]
    if len(available) == 0:
        log("  [SWAP] No real barcodes in raw counts -- skipping.")
        return adata
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
    log(f"\n{'='*70}")
    log(f"Embedding: {label}")
    log(f"  Input : {in_path}")

    adata = sc.read_h5ad(in_path)
    log(f"  Cells : {adata.n_obs:,}  Genes: {adata.n_vars:,}")

    adata = swap_real_cell_counts(adata, raw_adata)

    our_ids = np.array([g.split(".")[0] for g in adata.var.index])
    hgnc    = np.array([gene_map.get(g, "") for g in our_ids])
    keep    = np.isin(hgnc, list(vocab_genes)) & (hgnc != "")
    adata   = adata[:, keep].copy()
    adata.var.index = hgnc[keep]
    log(f"  Vocab subset: {keep.sum():,} / {len(keep):,} genes retained")

    sc.pp.filter_cells(adata, min_counts=1)
    log(f"  After filter_cells: {adata.n_obs:,} cells")

    if adata.n_obs == 0:
        log("  ERROR: No cells remain after filtering -- skipping.")
        return

    t0 = time.time()
    embeddings = scgpt.tasks.embed_data(
        adata,
        model_dir=str(CHECKPOINT),
        gene_col="index",
        obs_to_save=list(adata.obs.columns),
        batch_size=64,
        return_new_adata=True,
    )
    elapsed = time.time() - t0
    log(f"  Embedding runtime: {elapsed/60:.1f} min ({elapsed/adata.n_obs:.3f} s/cell)")

    emb_matrix = np.array(embeddings.X, dtype=np.float32)
    log(f"  Embedding shape: {emb_matrix.shape}")

    n_zeros  = (np.abs(emb_matrix).sum(axis=1) == 0).sum()
    n_unique = len(np.unique(emb_matrix, axis=0))
    log(f"  Zero-vector cells: {n_zeros}  |  Unique rows: {n_unique}")

    if n_unique <= MIN_UNIQUE_EMB:
        log(f"  ERROR: degenerate embedding ({n_unique} unique rows) -- NOT saving.")
        return

    orig      = sc.read_h5ad(in_path)
    orig      = swap_real_cell_counts(orig, raw_adata)
    emb_full  = np.zeros((orig.n_obs, emb_matrix.shape[1]), dtype=np.float32)
    dst_idx   = orig.obs_names.get_indexer(embeddings.obs_names)
    valid     = dst_idx >= 0
    emb_full[dst_idx[valid]] = emb_matrix[valid]
    orig.obsm[EMB_KEY] = emb_full

    orig.write_h5ad(out_path)
    log(f"  Saved -> {out_path.name}")


def main():
    log("="*70)
    log("STEP 2a -- scGPT Cell Embedding (AGE)")                         # [AGE]
    log("="*70)

    import json
    import scgpt

    log("Loading gene map...")
    gene_map = load_gene_map(GENE_INFO)

    log("Loading scGPT vocab...")
    vocab_path = CHECKPOINT / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_genes = set(vocab.keys()) - {"<pad>", "<cls>", "<eoc>"}
    log(f"  Vocab size: {len(vocab_genes):,} genes")

    log(f"Loading raw counts from {RAW_COUNTS_PATH.name}...")
    raw_adata = sc.read_h5ad(RAW_COUNTS_PATH)
    log(f"  Raw counts: {raw_adata.n_obs:,} cells, {raw_adata.n_vars:,} genes")

    for label, in_path in DATASETS.items():
        out_path = OUTDIR / OUTPUT_NAMES[label]
        if not in_path.exists():
            log(f"\nSkipping {label} -- input not found: {in_path.name}")
            continue
        if out_path.exists():
            ad_check = sc.read_h5ad(out_path)
            if EMB_KEY in ad_check.obsm:
                n_unique = len(np.unique(np.array(ad_check.obsm[EMB_KEY]), axis=0))
                if n_unique > MIN_UNIQUE_EMB:
                    log(f"\nSkipping {label} -- valid embedding exists ({n_unique} unique rows)")
                    continue
        embed_dataset(label, in_path, out_path, raw_adata, gene_map, vocab_genes)

    log("\n" + "="*70)
    log("STEP 2a AGE scGPT COMPLETE")
    for label, name in OUTPUT_NAMES.items():
        out    = OUTDIR / name
        status = "OK" if out.exists() else "MISSING"
        log(f"  [{status}] {name}")


if __name__ == "__main__":
    main()
