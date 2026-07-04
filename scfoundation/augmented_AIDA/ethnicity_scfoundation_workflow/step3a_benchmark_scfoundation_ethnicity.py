#!/usr/bin/env python3
"""STEP 3a — scIB Benchmarking (ETHNICITY, scFoundation)"""

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import time
import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

BASE   = pathlib.Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow")
OUTDIR = BASE / "benchmark_outputs_scfoundation_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = "AIDA_Ethnicity_Pilot"
DATASETS = {
    "BalancedAugmented_779Each": f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_scfoundation.h5ad",
    "Proportional_2500":          f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_scfoundation.h5ad",
    "BalancedUpsampled_779Each": f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_scfoundation.h5ad",
    "Downsampled_92Each":         f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_scfoundation.h5ad",
}

EMB_KEY   = "X_scfoundation"
LABEL_KEY = "cluster_labels"
ETHNICITY_COL_CANDIDATES = ["self_reported_ethnicity", "ethnicity", "Ethnicity", "ETHNICITY"]
UNKNOWN_ETHNICITY_VALUES = {"unknown", "na", "n/a", "not reported", "", "nan", "multiethnic", "na na", "not applicable", "prefer not to say"}
MIN_CELLS_PER_GROUP = 5

def detect_ethnicity_col(ad, fname):
    for c in ETHNICITY_COL_CANDIDATES:
        if c in ad.obs.columns: return c
    raise RuntimeError(f"No ethnicity column in {fname}. Columns: {list(ad.obs.columns)}")

def canonicalize_ethnicity(ad, col):
    raw = ad.obs[col].astype(str).str.strip().str.lower()
    unknown_mask = raw.isin(UNKNOWN_ETHNICITY_VALUES) | raw.isna()
    if unknown_mask.any():
        ad = ad[~unknown_mask].copy(); raw = raw[~unknown_mask]
    ad.obs[col] = raw.values
    small = ad.obs[col].value_counts(); small = small[small < MIN_CELLS_PER_GROUP].index.tolist()
    if small: ad = ad[~ad.obs[col].isin(small)].copy()
    return ad, col

def run_scib(mode, fname):
    path = BASE / fname
    if not path.exists(): print(f"\n  Skipping {mode} -- not found"); return None
    print(f"\n{'='*70}\nBenchmarking: {mode}")
    t0 = time.time()
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm: raise RuntimeError(f"Missing '{EMB_KEY}'")
    ad.obsm[EMB_KEY] = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask = np.isnan(ad.obsm[EMB_KEY]).any(axis=1)
    zero_mask = (np.abs(ad.obsm[EMB_KEY]).sum(axis=1) == 0)
    if (nan_mask | zero_mask).any(): ad = ad[~(nan_mask | zero_mask)].copy()
    if ad.n_obs < 50: raise RuntimeError(f"Too few cells ({ad.n_obs})")
    eth_col = detect_ethnicity_col(ad, fname)
    ad, eth_col = canonicalize_ethnicity(ad, eth_col)
    print(f"   Cells: {ad.n_obs:,}  Ethnicity col: '{eth_col}'")
    for grp, n in ad.obs[eth_col].value_counts().items(): print(f"     {grp}: {n}")
    n_groups = ad.obs[eth_col].nunique()
    if n_groups < 2: print(f"   SKIP -- only {n_groups} group(s)"); return None
    n_neighbors = 5 if ad.n_obs < 200 else 15
    sc.pp.neighbors(ad, use_rep=EMB_KEY, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", directed=False, n_iterations=2, key_added="leiden")
    ad.obs[LABEL_KEY] = ad.obs["leiden"].astype(str)
    n_clusters = ad.obs[LABEL_KEY].nunique()
    print(f"   Leiden clusters: {n_clusters}")
    emb_matrix = ad.obsm[EMB_KEY].astype(np.float32)
    ad_slim = sc.AnnData(X=sp.csr_matrix(emb_matrix), obs=ad.obs.copy())
    ad_slim.obsm[EMB_KEY] = emb_matrix; ad_slim.obsp = ad.obsp; ad_slim.uns = ad.uns
    bm = Benchmarker(ad_slim, batch_key=eth_col, label_key=LABEL_KEY,
        embedding_obsm_keys=[EMB_KEY],
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=True, clisi_knn=True, isolated_labels=False, silhouette_label=False),
        batch_correction_metrics=BatchCorrection(bras=True, ilisi_knn=True, kbet_per_label=True, graph_connectivity=False, pcr_comparison=True),
        n_jobs=-1)
    try:
        bm.benchmark()
    except Exception as e:
        if "arpack" in str(e).lower():
            bm = Benchmarker(ad_slim, batch_key=eth_col, label_key=LABEL_KEY,
                embedding_obsm_keys=[EMB_KEY],
                bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=True, clisi_knn=True, isolated_labels=False, silhouette_label=False),
                batch_correction_metrics=BatchCorrection(bras=True, ilisi_knn=True, kbet_per_label=False, graph_connectivity=False, pcr_comparison=True),
                n_jobs=-1)
            bm.benchmark()
        else: raise
    results = bm.get_results(min_max_scale=False)
    if results.empty: raise RuntimeError(f"scIB empty for {mode}")
    results.loc[EMB_KEY].to_csv(OUTDIR / f"{mode}_scib_metrics.csv")
    runtime = round((time.time() - t0) / 60, 2)
    print(f"   Runtime: {runtime} min")
    row = results.loc[EMB_KEY]
    return {
        "dataset": mode, "file": fname, "cells": ad.n_obs,
        "n_groups": n_groups, "n_clusters": n_clusters,
        "NMI":              float(row["KMeans NMI"])       if "KMeans NMI"       in row.index else np.nan,
        "ARI":              float(row["KMeans ARI"])       if "KMeans ARI"       in row.index else np.nan,
        "cLISI":            float(row["cLISI"])            if "cLISI"            in row.index else np.nan,
        "silhouette_batch": float(row["Silhouette batch"]) if "Silhouette batch" in row.index else np.nan,
        "iLISI":            float(row["iLISI"])            if "iLISI"            in row.index else np.nan,
        "kBET":             float(row["KBET"])             if "KBET"             in row.index else np.nan,
        "PCR":              float(row["PCR comparison"])   if "PCR comparison"   in row.index else np.nan,
        "runtime_min": runtime,
    }

def main():
    print("\nSTEP 3a -- scIB Benchmarking (ETHNICITY, scFoundation)")
    summary = []
    for mode, fname in DATASETS.items():
        result = run_scib(mode, fname)
        if result is not None: summary.append(result)
    if not summary: print("\nNo datasets benchmarked."); return
    df = pd.DataFrame(summary)
    df.to_csv(OUTDIR / "benchmark_summary_all_modes.csv", index=False)
    print(f"\n{'='*70}\nSTEP 3a COMPLETE (ETHNICITY, scFoundation)\n\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
