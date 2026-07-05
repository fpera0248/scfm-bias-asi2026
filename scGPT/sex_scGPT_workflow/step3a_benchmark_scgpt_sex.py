#!/usr/bin/env python3
"""
STEP 3a — scIB Benchmarking (SEX, scGPT)

Changes vs ethnicity scGPT version:
  [SEX 1] BASE/OUTDIR  -> sex_scGPT_workflow
  [SEX 2] GROUP_KEY    -> "sex"
  [SEX 3] DATASETS     -> sex filenames (1413Each, 586Each, 1999)
  [SEX 4] OUTDIR       -> benchmark_outputs_scgpt_sex
"""

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import time
import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

BASE  = pathlib.Path("/data/scGPT/sex_scGPT_workflow")
INDIR = BASE / "step2a_embeddings"
OUTDIR = BASE / "benchmark_outputs_scgpt_sex"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = "ILD_Sex_Pilot"

DATASETS = {
    "BalancedAugmented_1413Each": f"{OUTPUT_BASE}_BalancedAugmented_1413Each_scgpt.h5ad",
    "Proportional_1999":          f"{OUTPUT_BASE}_Proportional_1999_scgpt.h5ad",
    "BalancedUpsampled_1413Each": f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_scgpt.h5ad",
    "Downsampled_586Each":        f"{OUTPUT_BASE}_Downsampled_586Each_scgpt.h5ad",
}

EMB_KEY   = "X_scGPT"
LABEL_KEY = "cluster_labels"

SEX_COL_CANDIDATES  = ["sex", "Sex", "SEX", "gender", "Gender"]
UNKNOWN_SEX_VALUES  = {"unknown", "na", "n/a", "not reported", "", "nan", "not applicable"}
MIN_CELLS_PER_GROUP = 5


def detect_sex_col(ad, fname):
    for c in SEX_COL_CANDIDATES:
        if c in ad.obs.columns:
            return c
    raise RuntimeError(f"No sex column found in {fname}.\nColumns: {list(ad.obs.columns)}")


def canonicalize_sex(ad, col):
    raw = ad.obs[col].astype(str).str.strip().str.lower()
    unknown_mask = raw.isin(UNKNOWN_SEX_VALUES) | raw.isna()
    if unknown_mask.any():
        print(f"   Dropping {unknown_mask.sum()} cells with unknown/missing sex.")
        ad  = ad[~unknown_mask].copy()
        raw = raw[~unknown_mask]
    ad.obs[col] = raw.values
    grp_counts = ad.obs[col].value_counts()
    small_grps = grp_counts[grp_counts < MIN_CELLS_PER_GROUP].index.tolist()
    if small_grps:
        ad = ad[~ad.obs[col].isin(small_grps)].copy()
    return ad, col


def run_scib(mode: str, fname: str):
    path = INDIR / fname
    if not path.exists():
        print(f"\n  Skipping {mode} -- file not found: {fname}")
        return None

    print(f"\n{'='*70}\nBenchmarking: {mode}")
    t0 = time.time()
    ad = sc.read_h5ad(path)

    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing '{EMB_KEY}' in {fname}")

    ad.obsm[EMB_KEY] = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    nan_mask  = np.isnan(ad.obsm[EMB_KEY]).any(axis=1)
    zero_mask = (np.abs(ad.obsm[EMB_KEY]).sum(axis=1) == 0)
    if (nan_mask | zero_mask).any():
        ad = ad[~(nan_mask | zero_mask)].copy()

    if ad.n_obs < 50:
        raise RuntimeError(f"Too few cells ({ad.n_obs})")

    sex_col = detect_sex_col(ad, fname)
    ad, sex_col = canonicalize_sex(ad, sex_col)

    print(f"   Cells: {ad.n_obs:,}  Sex col: '{sex_col}'")
    for grp, n in ad.obs[sex_col].value_counts().items():
        print(f"     {grp}: {n}")

    n_groups = ad.obs[sex_col].nunique()
    if n_groups < 2:
        print(f"   SKIP -- only {n_groups} group(s)"); return None

    n_neighbors = 5 if ad.n_obs < 200 else 15
    sc.pp.neighbors(ad, use_rep=EMB_KEY, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", directed=False, n_iterations=2, key_added="leiden")
    ad.obs[LABEL_KEY] = ad.obs["leiden"].astype(str)
    n_clusters = ad.obs[LABEL_KEY].nunique()
    print(f"   Leiden clusters: {n_clusters}")

    emb_matrix = ad.obsm[EMB_KEY].astype(np.float32)
    ad_slim = sc.AnnData(X=sp.csr_matrix(emb_matrix), obs=ad.obs.copy())
    ad_slim.obsm[EMB_KEY] = emb_matrix
    ad_slim.obsp = ad.obsp
    ad_slim.uns  = ad.uns

    bm = Benchmarker(
        ad_slim, batch_key=sex_col, label_key=LABEL_KEY,
        embedding_obsm_keys=[EMB_KEY],
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=True, clisi_knn=True,
            isolated_labels=False, silhouette_label=False),
        batch_correction_metrics=BatchCorrection(
            silhouette_batch=True, ilisi_knn=True, kbet_per_label=True,
            graph_connectivity=False, pcr_comparison=True),
        n_jobs=-1,
    )
    try:
        bm.benchmark()
    except Exception as e:
        if "arpack" in str(e).lower():
            print(f"   WARNING: ARPACK crash -- retrying without kbet...")
            bm = Benchmarker(
                ad_slim, batch_key=sex_col, label_key=LABEL_KEY,
                embedding_obsm_keys=[EMB_KEY],
                bio_conservation_metrics=BioConservation(
                    nmi_ari_cluster_labels_kmeans=True, clisi_knn=True,
                    isolated_labels=False, silhouette_label=False),
                batch_correction_metrics=BatchCorrection(
                    silhouette_batch=True, ilisi_knn=True, kbet_per_label=False,
                    graph_connectivity=False, pcr_comparison=True),
                n_jobs=-1,
            )
            bm.benchmark()
        else:
            raise

    results = bm.get_results(min_max_scale=False)
    if results.empty:
        raise RuntimeError(f"scIB returned empty results for {mode}")

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
    print("\nSTEP 3a -- scIB Benchmarking (SEX, scGPT)")
    summary = []
    for mode, fname in DATASETS.items():
        result = run_scib(mode, fname)
        if result is not None:
            summary.append(result)
    if not summary:
        print("\nNo datasets benchmarked successfully."); return
    df = pd.DataFrame(summary)
    df.to_csv(OUTDIR / "benchmark_summary_all_modes.csv", index=False)
    print(f"\n{'='*70}\nSTEP 3a COMPLETE (SEX, scGPT)\n\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
