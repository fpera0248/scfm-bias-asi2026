#!/usr/bin/env python3
"""
STEP 3a — scIB Benchmarking (ETHNICITY, Pilot)
Geneformer V2-316M embeddings

Changes vs step3a scFoundation ethnicity:
  [GF 1]  BASE path -> ethnicity_Geneformer_workflow
  [GF 2]  EMB_KEY   -> X_geneformer
  [GF 3]  DATASETS  -> _geneformer.h5ad filenames, 2143Each / 48Each / 2497
  [GF 4]  OUTDIR    -> benchmark_outputs_geneformer_ethnicity

All benchmarking logic preserved verbatim.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import time

import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# ============================================================
# PATHS  [GF 1]
# ============================================================

BASE = pathlib.Path(
    "/data/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow"
)

INDIR  = BASE
OUTDIR = BASE / "benchmark_outputs_geneformer_ethnicity"   # [GF 4]
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = "AIDA_Ethnicity_Pilot"

DATASETS = {                                                 # [GF 3]
    "BalancedAugmented_779Each": f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_geneformer.h5ad",
    "Proportional_2500":          f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_geneformer.h5ad",
    "BalancedUpsampled_779Each": f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_geneformer.h5ad",
    "Downsampled_92Each":         f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_geneformer.h5ad",
}

# ============================================================
# REQUIRED KEYS  [GF 2]
# ============================================================

EMB_KEY   = "X_geneformer"
LABEL_KEY = "cluster_labels"

ETHNICITY_COL_CANDIDATES = [
    "self_reported_ethnicity",
    "ethnicity",
    "Ethnicity",
    "ETHNICITY",
]

UNKNOWN_ETHNICITY_VALUES = {
    "unknown", "na", "n/a", "not reported", "", "nan",
    "multiethnic", "na na", "not applicable", "prefer not to say",
}

MIN_CELLS_PER_GROUP = 5

# ============================================================
# HELPERS
# ============================================================

def detect_ethnicity_col(ad, fname):
    for c in ETHNICITY_COL_CANDIDATES:
        if c in ad.obs.columns:
            return c
    raise RuntimeError(
        f"No ethnicity column found in {fname}.\n"
        f"Observed columns: {list(ad.obs.columns)}\n"
        f"Add your column name to ETHNICITY_COL_CANDIDATES."
    )


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


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def run_scib(mode: str, fname: str):
    path = INDIR / fname

    if not path.exists():
        print(f"\n  Skipping {mode} -- file not found: {fname}")
        return None

    print(f"\n{'='*70}")
    print(f"Benchmarking: {mode}")
    print(f"   File: {fname}")
    print(f"   File size: {path.stat().st_size / 1e6:.1f} MB")
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
        print(f"   Dropping {zero_mask.sum()} cells with all-zero embeddings (tokenizer dropout)")
        ad = ad[~zero_mask].copy()

    if ad.n_obs < 50:
        raise RuntimeError(f"Too few cells after filtering ({ad.n_obs})")

    if ad.n_obs < 200:
        print(f"   WARNING: small dataset ({ad.n_obs} cells). Reducing n_neighbors to 5.")

    eth_col = detect_ethnicity_col(ad, fname)
    ad, eth_col = canonicalize_ethnicity(ad, eth_col)

    print(f"   Cells         : {ad.n_obs:,}")
    print(f"   Ethnicity col : '{eth_col}'")
    print(f"   Group breakdown:")
    for grp, n in ad.obs[eth_col].value_counts().items():
        print(f"     {grp}: {n}")

    n_groups = ad.obs[eth_col].nunique()
    if n_groups < 2:
        print(f"   SKIP -- only {n_groups} group(s) after filtering; iLISI requires >= 2.")
        return None

    n_neighbors = 5 if ad.n_obs < 200 else 15
    print("   Computing neighbors + Leiden clusters...")
    sc.pp.neighbors(ad, use_rep=EMB_KEY, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", directed=False,
                 n_iterations=2, key_added="leiden")
    ad.obs[LABEL_KEY] = ad.obs["leiden"].astype(str)
    n_clusters = ad.obs[LABEL_KEY].nunique()
    print(f"   Leiden clusters: {n_clusters}  (n_neighbors={n_neighbors}, resolution=0.3)")

    emb_matrix = ad.obsm[EMB_KEY].astype(np.float32)
    ad_slim = sc.AnnData(
        X   = sp.csr_matrix(emb_matrix),
        obs = ad.obs.copy(),
    )
    ad_slim.obsm[EMB_KEY] = emb_matrix
    ad_slim.obsp = ad.obsp
    ad_slim.uns  = ad.uns

    bm = Benchmarker(
        ad_slim,
        batch_key=eth_col,
        label_key=LABEL_KEY,
        embedding_obsm_keys=[EMB_KEY],
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=True,
            clisi_knn=True,
            isolated_labels=False,
            silhouette_label=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True,
            ilisi_knn=True,
            kbet_per_label=True,
            graph_connectivity=False,
            pcr_comparison=True,
        ),
        n_jobs=-1,
    )

    print("   Running scIB metrics...")
    try:
        bm.benchmark()
    except Exception as e:
        if "ArpackError" in type(e).__name__ or "ARPACK" in str(e) or "arpack" in str(e).lower():
            print(f"   WARNING: kbet_per_label ARPACK crash ({type(e).__name__}). Retrying without kbet...")
            bm = Benchmarker(
                ad_slim,
                batch_key=eth_col,
                label_key=LABEL_KEY,
                embedding_obsm_keys=[EMB_KEY],
                bio_conservation_metrics=BioConservation(
                    nmi_ari_cluster_labels_kmeans=True,
                    clisi_knn=True,
                    isolated_labels=False,
                    silhouette_label=False,
                ),
                batch_correction_metrics=BatchCorrection(
                    bras=True,
                    ilisi_knn=True,
                    kbet_per_label=False,
                    graph_connectivity=False,
                    pcr_comparison=True,
                ),
                n_jobs=-1,
            )
            bm.benchmark()
        else:
            raise

    results = bm.get_results(min_max_scale=False)
    if results.empty:
        raise RuntimeError(f"scIB returned empty results for {mode}")

    csv_path = OUTDIR / f"{mode}_scib_metrics.csv"
    results.loc[EMB_KEY].to_csv(csv_path)
    print(f"   Metrics saved -> {csv_path.name}")

    dump_path = OUTDIR / f"bm_dict_{mode}.txt"
    with open(dump_path, "w") as fh:
        fh.write(f"Benchmarker __dict__ for {mode}\n\n")
        for k, v in sorted(bm.__dict__.items()):
            s = str(v)
            if len(s) > 15000:
                s = s[:15000] + " ...[truncated]"
            fh.write(f"{k}: {s}\n\n")

    runtime = round((time.time() - t0) / 60, 2)
    print(f"   Runtime: {runtime} min")

    row = results.loc[EMB_KEY]
    return {
        "dataset":          mode,
        "file":             fname,
        "cells":            ad.n_obs,
        "n_groups":         n_groups,
        "n_clusters":       n_clusters,
        "NMI":              float(row["KMeans NMI"])        if "KMeans NMI"        in row.index else np.nan,
        "ARI":              float(row["KMeans ARI"])        if "KMeans ARI"        in row.index else np.nan,
        "cLISI":            float(row["cLISI"])             if "cLISI"             in row.index else np.nan,
        "bras": float(row["Silhouette batch"])  if "Silhouette batch"  in row.index else np.nan,
        "iLISI":            float(row["iLISI"])             if "iLISI"             in row.index else np.nan,
        "kBET":             float(row["KBET"])              if "KBET"              in row.index else np.nan,
        "PCR":              float(row["PCR comparison"])    if "PCR comparison"    in row.index else np.nan,
        "runtime_min":      runtime,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nSTEP 3a -- scIB Benchmarking (ETHNICITY, Geneformer V2-316M)")
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

    print(f"\n{'='*70}")
    print("STEP 3a COMPLETE (ETHNICITY, Geneformer)")
    print(f"\nSummary -> {summary_path.name}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()