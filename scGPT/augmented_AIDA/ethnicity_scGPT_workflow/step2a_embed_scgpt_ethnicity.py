#!/usr/bin/env python3
"""
STEP 2a — Cell Embedding (ETHNICITY, scGPT whole-human)

Embeds each ethnicity pilot dataset using scGPT's frozen whole-human checkpoint.
Outputs 512-dim cell embeddings stored in obsm["X_scGPT"].

Gene name handling:
  Input h5ads use Ensembl IDs as var_names.
  scGPT vocab uses HGNC gene symbols.
  gene_info.csv maps Ensembl -> symbol; genes not in scGPT vocab are dropped.
"""

import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scgpt

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE       = pathlib.Path("/data/scGPT")
MODEL_DIR  = BASE / "scGPT_human"
GENE_INFO  = BASE / "gene_info.csv"
ETH_DIR    = BASE / "augmented_AIDA" / "ethnicity_scGPT_workflow"
OUTDIR     = ETH_DIR / "step2a_embeddings"
OUTDIR.mkdir(exist_ok=True)

FILES = {
    "Proportional_2500":          ETH_DIR / "AIDA_Ethnicity_Pilot_Proportional_2500_ETHNICITY.h5ad",
    "BalancedAugmented_779Each": ETH_DIR / "AIDA_Ethnicity_Pilot_BalancedAugmented_779Each_ETHNICITY.h5ad",
    "BalancedUpsampled_779Each": ETH_DIR / "AIDA_Ethnicity_Pilot_BalancedUpsampled_779Each_ETHNICITY.h5ad",
    "Downsampled_92Each":         ETH_DIR / "AIDA_Ethnicity_Pilot_Downsampled_92Each_ETHNICITY.h5ad",
}

VALIDATION_FILE = ETH_DIR / "AIDA_Ethnicity_External_Validation_12500.h5ad"

BATCH_SIZE  = 64
MAX_LENGTH  = 1200
EMB_KEY     = "X_scGPT"

# ============================================================
# HELPERS
# ============================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_gene_map(gene_info_path):
    """Return dict: ensembl_id -> gene_symbol from gene_info.csv."""
    df = pd.read_csv(gene_info_path)
    # columns: gene_id, gene_name (or similar — handle both conventions)
    id_col  = "feature_id"
    sym_col = "feature_name"
    log(f"  gene_info columns: {list(df.columns)} -> using '{id_col}' and '{sym_col}'")
    return dict(zip(df[id_col], df[sym_col]))


def prepare_adata(adata, gene_map):
    """
    Convert var_names from Ensembl IDs to HGNC symbols.
    Genes with no mapping or not in scGPT vocab are kept with their Ensembl ID
    as fallback — scGPT's embed_data will drop unmapped genes internally.
    """
    adata = adata.copy()
    adata.var["ensembl_id"] = adata.var_names.tolist()
    adata.var["gene_name"]  = [gene_map.get(g, g) for g in adata.var_names]
    adata.var_names         = adata.var["gene_name"].tolist()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    # Filter cells with zero total counts (would crash binning)
    sc.pp.filter_cells(adata, min_counts=1)
    return adata


def embed(adata, label):
    log(f"  Embedding {label}: {adata.n_obs} cells")
    emb_adata = scgpt.tasks.embed_data(
        adata,
        MODEL_DIR,
        gene_col="gene_name",
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        obs_to_save=list(adata.obs.columns),
        device="cuda",
        use_fast_transformer=False,
        return_new_adata=True,
    )
    log(f"  Embedding shape: {emb_adata.X.shape}")

    # Write embedding back into original obs structure
    out = adata.copy()
    # emb_adata.X is the 512-dim embedding matrix
    # align by obs_names in case any cells were dropped internally
    common = adata.obs_names.intersection(emb_adata.obs_names)
    if len(common) < adata.n_obs:
        log(f"  WARNING: {adata.n_obs - len(common)} cells dropped during embedding")

    emb_matrix = np.zeros((adata.n_obs, emb_adata.X.shape[1]), dtype=np.float32)
    src_idx = emb_adata.obs_names.get_indexer(common)
    dst_idx = adata.obs_names.get_indexer(common)
    emb_matrix[dst_idx] = np.array(emb_adata.X[src_idx], dtype=np.float32)

    out.obsm[EMB_KEY] = emb_matrix
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 60)
    log("STEP 2a -- Cell Embedding (ETHNICITY, scGPT)")
    log("=" * 60)

    gene_map = load_gene_map(GENE_INFO)
    log(f"Gene map loaded: {len(gene_map):,} Ensembl -> symbol entries")

    # Pilot datasets
    for label, path in FILES.items():
        if not path.exists():
            log(f"  Skipping {label} -- file not found")
            continue

        log(f"\n>> {label}")
        adata = sc.read_h5ad(path)
        adata = prepare_adata(adata, gene_map)
        out   = embed(adata, label)

        out_path = OUTDIR / f"AIDA_Ethnicity_Pilot_{label}_scgpt.h5ad"
        out.write_h5ad(out_path)
        log(f"  Saved -> {out_path.name}")

    # External validation
    log(f"\n>> External Validation")
    adata_val = sc.read_h5ad(VALIDATION_FILE)
    adata_val = prepare_adata(adata_val, gene_map)
    out_val   = embed(adata_val, "ExternalValidation_12500")

    val_path = OUTDIR / "AIDA_Ethnicity_External_Validation_12500_scgpt.h5ad"
    out_val.write_h5ad(val_path)
    log(f"  Saved -> {val_path.name}")

    log("\nSTEP 2a COMPLETE")


if __name__ == "__main__":
    main()
