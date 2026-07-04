#!/usr/bin/env python3
"""STEP 3a — scIB Benchmarking (AGE, Geneformer)"""
import scanpy as sc
import pandas as pd
import numpy as np
import pathlib, time
import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

BASE   = pathlib.Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow")
EMBDIR = BASE
OUTDIR = BASE / "benchmark_outputs_geneformer_age"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "ILD_Age_Pilot"
DATASETS = {
    "Proportional_2495":          f"{OUTPUT_BASE}_Proportional_2495_AGE_geneformer.h5ad",
    "BalancedAugmented_1262Each": f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE_geneformer.h5ad",
    "BalancedUpsampled_1262Each": f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE_geneformer.h5ad",
    "Downsampled_25Each":         f"{OUTPUT_BASE}_Downsampled_25Each_AGE_geneformer.h5ad",
}

EMB_KEY   = "X_geneformer"
LABEL_KEY = "cluster_labels"
AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group"]
UNKNOWN_VALUES = {"unknown","na","n/a","not reported","","nan"}
MIN_CELLS_PER_GROUP = 5

def detect_age_col(ad):
    for c in AGE_COL_CANDIDATES:
        if c in ad.obs.columns: return c
    raise RuntimeError(f"No age column. Columns: {list(ad.obs.columns)}")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_scib(mode, fname):
    path = EMBDIR / fname
    if not path.exists(): log(f"SKIP {mode} — not found"); return None
    log(f"Benchmarking: {mode}")
    t0 = time.time()
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm: raise RuntimeError(f"Missing {EMB_KEY}")
    emb = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    bad = np.isnan(emb).any(axis=1) | (np.abs(emb).sum(axis=1) == 0)
    if bad.any(): ad = ad[~bad].copy(); emb = emb[~bad]
    age_col = detect_age_col(ad)
    raw = ad.obs[age_col].astype(str).str.strip().str.lower()
    ad = ad[~raw.isin(UNKNOWN_VALUES)].copy()
    ad.obs[age_col] = raw[~raw.isin(UNKNOWN_VALUES)].values
    small = ad.obs[age_col].value_counts()
    small = small[small < MIN_CELLS_PER_GROUP].index
    if len(small): ad = ad[~ad.obs[age_col].isin(small)].copy()
    if ad.n_obs < 50: log(f"SKIP {mode} — too few cells"); return None
    n_groups = ad.obs[age_col].nunique()
    if n_groups < 2: log(f"SKIP {mode} — only {n_groups} group(s)"); return None
    n_neighbors = 5 if ad.n_obs < 200 else 15
    sc.pp.neighbors(ad, use_rep=EMB_KEY, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", directed=False, n_iterations=2, key_added="leiden")
    ad.obs[LABEL_KEY] = ad.obs["leiden"].astype(str)
    emb_m = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    ad_slim = sc.AnnData(X=sp.csr_matrix(emb_m), obs=ad.obs.copy())
    ad_slim.obsm[EMB_KEY] = emb_m; ad_slim.obsp = ad.obsp; ad_slim.uns = ad.uns
    bm = Benchmarker(ad_slim, batch_key=age_col, label_key=LABEL_KEY,
        embedding_obsm_keys=[EMB_KEY],
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=True, clisi_knn=True, isolated_labels=False, silhouette_label=False),
        batch_correction_metrics=BatchCorrection(silhouette_batch=True, ilisi_knn=True, kbet_per_label=True, graph_connectivity=False, pcr_comparison=True),
        n_jobs=-1)
    try: bm.benchmark()
    except Exception as e:
        if "arpack" in str(e).lower():
            bm = Benchmarker(ad_slim, batch_key=age_col, label_key=LABEL_KEY,
                embedding_obsm_keys=[EMB_KEY],
                bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=True, clisi_knn=True, isolated_labels=False, silhouette_label=False),
                batch_correction_metrics=BatchCorrection(silhouette_batch=True, ilisi_knn=True, kbet_per_label=False, graph_connectivity=False, pcr_comparison=True),
                n_jobs=-1)
            bm.benchmark()
        else: raise
    results = bm.get_results(min_max_scale=False)
    if results.empty: raise RuntimeError("Empty scIB results")
    results.loc[EMB_KEY].to_csv(OUTDIR / f"{mode}_scib_metrics.csv")
    row = results.loc[EMB_KEY]
    rt = round((time.time()-t0)/60, 2)
    log(f"  Done in {rt} min")
    return {"dataset": mode, "cells": ad.n_obs, "n_groups": n_groups,
            "NMI":  float(row.get("KMeans NMI", np.nan)),
            "ARI":  float(row.get("KMeans ARI", np.nan)),
            "iLISI": float(row.get("iLISI", np.nan)),
            "kBET":  float(row.get("KBET", np.nan)),
            "runtime_min": rt}

summary = []
for mode, fname in DATASETS.items():
    r = run_scib(mode, fname)
    if r: summary.append(r)
df = pd.DataFrame(summary)
df.to_csv(OUTDIR / "benchmark_summary_all_modes.csv", index=False)
print(df.to_string(index=False))
