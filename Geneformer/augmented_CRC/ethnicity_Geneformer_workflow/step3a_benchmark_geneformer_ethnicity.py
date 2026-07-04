#!/usr/bin/env python3
"""
STEP 3a — scIB Benchmarking (CRC ETHNICITY, Pilot)
Geneformer V2-316M embeddings
"""

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import time
import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

BASE   = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow")
INDIR  = BASE
OUTDIR = BASE / "benchmark_outputs_geneformer_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = "CRC_Eth_Pilot"

DATASETS = {
    "BalancedAugmented_1880Each": f"{OUTPUT_BASE}_BalancedAugmented_1880Each_ETH_geneformer.h5ad",
    "Proportional_2497":          f"{OUTPUT_BASE}_Proportional_2497_ETH_geneformer.h5ad",
    "BalancedUpsampled_1880Each": f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_ETH_geneformer.h5ad",
    "Downsampled_48Each":         f"{OUTPUT_BASE}_Downsampled_48Each_ETH_geneformer.h5ad",
}

EMB_KEY   = "X_geneformer"
LABEL_KEY = "cluster_labels"

ETHNICITY_COL_CANDIDATES = [
    "self_reported_ethnicity", "ethnicity", "Ethnicity", "ETHNICITY",
]
UNKNOWN_ETHNICITY_VALUES = {
    "unknown", "na", "n/a", "not reported", "", "nan",
    "multiethnic", "na na", "not applicable", "prefer not to say",
}
MIN_CELLS_PER_GROUP = 5


def detect_ethnicity_col(ad, fname):
    for c in ETHNICITY_COL_CANDIDATES:
        if c in ad.obs.columns:
            return c
    raise RuntimeError(f"No ethnicity column found in {fname}.")


def canonicalize_ethnicity(ad, col):
    raw = ad.obs[col].astype(str).str.strip().str.lower()
    unknown_mask = raw.isin(UNKNOWN_ETHNICITY_VALUES) | raw.isna()
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0:
        print(f"   Dropping {n_unknown} cells with unknown/missing ethnicity.")
        ad  = ad[~unknown_mask].copy()
        raw = raw[~unknown_mask]
    ad.obs[col] = raw.values
    grp_counts = ad.obs[col].value_counts()
    small_grps = grp_counts[grp_counts < MIN_CELLS_PER_GROUP].index.tolist()
    if small_grps:
        print(f"   Dropping groups with < {MIN_CELLS_PER_GROUP} cells: {small_grps}")
        ad = ad[~ad.obs[col].isin(small_grps)].copy()
    return ad, col


def run_scib(mode: str, fname: str):
    path = INDIR / fname
    if not path.exists():
        print(f"\n  Skipping {mode} -- file not found: {fname}")
        return None
    print(f"\n{'='*70}")
    print(f"Benchmarking: {mode}")
    t0 = time.time()
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing embedding '{EMB_KEY}' in {fname}")
    ad.obsm[EMB_KEY] = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask = np.isnan(ad.obsm[EMB_KEY]).any(axis=1)
    if nan_mask.any():
        print(f"   Dropping {nan_mask.sum()} cells with NaN embeddings")
        ad = ad[~nan_mask].copy()
    zero_mask = (np.abs(ad.obsm[EMB_KEY]).sum(axis=1) == 0)
    if zero_mask.any():
        print(f"   Dropping {zero_mask.sum()} cells with all-zero embeddings")
        ad = ad[~zero_mask].copy()
    if ad.n_obs < 50:
        raise RuntimeError(f"Too few cells after filtering ({ad.n_obs})")
    eth_col = detect_ethnicity_col(ad, fname)
    ad, eth_col = canonicalize_ethnicity(ad, eth_col)
    print(f"   Cells: {ad.n_obs:,}  |  Ethnicity col: '{eth_col}'")
    for grp, n in ad.obs[eth_col].value_counts().items():
        print(f"     {grp}: {n}")
    n_groups = ad.obs[eth_col].nunique()
    if n_groups < 2:
        print(f"   SKIP -- only {n_groups} group(s); iLISI requires >= 2.")
        return None
    n_neighbors = 5 if ad.n_obs < 200 else 15
    sc.pp.neighbors(ad, use_rep=EMB_KEY, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", directed=False,
                 n_iterations=2, key_added="leiden")
    ad.obs[LABEL_KEY] = ad.obs["leiden"].astype(str)
    n_clusters = ad.obs[LABEL_KEY].nunique()
    print(f"   Leiden clusters: {n_clusters}  (n_neighbors={n_neighbors})")
    emb_matrix = ad.obsm[EMB_KEY].astype(np.float32)
    ad_slim = sc.AnnData(X=sp.csr_matrix(emb_matrix), obs=ad.obs.copy())
    ad_slim.obsm[EMB_KEY] = emb_matrix
    ad_slim.obsp = ad.obsp
    ad_slim.uns  = ad.uns
    bm = Benchmarker(
        ad_slim, batch_key=eth_col, label_key=LABEL_KEY,
        embedding_obsm_keys=[EMB_KEY],
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=True, clisi_knn=True,
            isolated_labels=False, silhouette_label=False),
        batch_correction_metrics=BatchCorrection(
            ilisi_knn=True, kbet_per_label=True,
            graph_connectivity=False, pcr_comparison=True),
        n_jobs=-1,
    )
    print("   Running scIB metrics...")
    try:
        bm.benchmark()
    except Exception as e:
        if "ArpackError" in type(e).__name__ or "ARPACK" in str(e):
            print(f"   WARNING: ARPACK crash -- retrying without kbet...")
            bm = Benchmarker(
                ad_slim, batch_key=eth_col, label_key=LABEL_KEY,
                embedding_obsm_keys=[EMB_KEY],
                bio_conservation_metrics=BioConservation(
                    nmi_ari_cluster_labels_kmeans=True, clisi_knn=True,
                    isolated_labels=False, silhouette_label=False),
                batch_correction_metrics=BatchCorrection(
                    ilisi_knn=True, kbet_per_label=False,
                    graph_connectivity=False, pcr_comparison=True),
                n_jobs=-1,
            )
            bm.benchmark()
        else:
            raise
    results  = bm.get_results(min_max_scale=False)
    csv_path = OUTDIR / f"{mode}_scib_metrics.csv"
    results.loc[EMB_KEY].to_csv(csv_path)
    print(f"   Metrics saved -> {csv_path.name}")
    dump_path = OUTDIR / f"bm_dict_{mode}.txt"
    with open(dump_path, "w") as fh:
        fh.write(f"Benchmarker __dict__ for {mode}\n\n")
        for k, v in sorted(bm.__dict__.items()):
            s = str(v)
            if len(s) > 15000: s = s[:15000] + " ...[truncated]"
            fh.write(f"{k}: {s}\n\n")
    runtime = round((time.time() - t0) / 60, 2)
    print(f"   Runtime: {runtime} min")
    row = results.loc[EMB_KEY]
    return {
        "dataset":          mode,    "file":       fname,
        "cells":            ad.n_obs, "n_groups":  n_groups,
        "n_clusters":       n_clusters,
        "NMI":              float(row["KMeans NMI"])       if "KMeans NMI"       in row.index else np.nan,
        "ARI":              float(row["KMeans ARI"])       if "KMeans ARI"       in row.index else np.nan,
        "cLISI":            float(row["cLISI"])            if "cLISI"            in row.index else np.nan,
        "silhouette_batch": float(row["Silhouette batch"]) if "Silhouette batch" in row.index else np.nan,
        "iLISI":            float(row["iLISI"])            if "iLISI"            in row.index else np.nan,
        "kBET":             float(row["KBET"])             if "KBET"             in row.index else np.nan,
        "PCR":              float(row["PCR comparison"])   if "PCR comparison"   in row.index else np.nan,
        "runtime_min":      runtime,
    }


def main():
    print("\nSTEP 3a -- scIB Benchmarking (CRC_Eth_Pilot, Geneformer V2-316M)")
    print(f"   Output dir: {OUTDIR}")
    summary = []
    for mode, fname in DATASETS.items():
        result = run_scib(mode, fname)
        if result is not None:
            summary.append(result)
    if not summary:
        print("\nNo datasets benchmarked successfully.")
        return
    df = pd.DataFrame(summary)
    summary_path = OUTDIR / "benchmark_summary_all_modes.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSTEP 3a COMPLETE")
    print(f"Summary -> {summary_path.name}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
