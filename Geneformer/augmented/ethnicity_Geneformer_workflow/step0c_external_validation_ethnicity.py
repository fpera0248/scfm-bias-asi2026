#!/usr/bin/env python3
"""
STEP 0c — External Validation Set Construction (ETHNICITY, Geneformer workflow)

Changes vs. augmentedv4 version:
  [GF 1] BASE path updated to Geneformer/augmented/ethnicity_Geneformer_workflow
  [GF 2] PILOT_FILES = [] — no prior pilot exists in this workflow yet.
          Run step0c before step0b. step0b will exclude these barcodes.
  [GF 3] CELLS_PER_GROUP = 2500 (unchanged)
  All other logic preserved verbatim.

Run order: step0a -> step0c -> step0b
step0b reads VALIDATION_H5AD to exclude those barcodes from training.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"
)

SOURCE_H5AD = BASE / "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"

# [GF 2] No prior pilot in this workflow. step0c runs first.
PILOT_FILES: list[Path] = []

OUT_FILE = BASE / "ILD_Ethnicity_External_Validation_12500.h5ad"
LOGFILE  = BASE / "step0c_external_validation_ethnicity_log.txt"

# ============================================================
# CONFIG
# ============================================================

GROUP_COL        = "self_reported_ethnicity"
DISEASE_COL      = "disease"
SOURCE_COL       = "source"
CELLS_PER_GROUP  = 2500
RANDOM_STATE     = 42

# ============================================================
# LOGGING
# ============================================================

log_fh = open(LOGFILE, "w")

def log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    print(line, file=log_fh, flush=True)

# ============================================================
# HELPERS
# ============================================================

def to_binary_disease(x):
    if pd.isna(x):
        return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"


def collect_pilot_barcodes(pilot_files):
    barcodes = set()
    if not pilot_files:
        log("  No pilot files configured; skipping pilot exclusion.")
        return barcodes
    for f in pilot_files:
        if not f.exists():
            raise FileNotFoundError(f"Pilot file not found: {f}")
        ad = sc.read_h5ad(f, backed="r")
        if SOURCE_COL in ad.obs.columns:
            real_mask    = (ad.obs[SOURCE_COL] == "real").to_numpy()
            real_barcodes = ad.obs.index[real_mask].tolist()
        else:
            real_barcodes = ad.obs.index.tolist()
        log(f"  {f.name}: {len(real_barcodes):,} real cells")
        barcodes.update(real_barcodes)
        ad.file.close()
    return barcodes

# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(RANDOM_STATE)

    log("=" * 70)
    log("STEP 0c -- External Validation Set Construction (ETHNICITY, Geneformer)")
    log("=" * 70)
    log(f"Source       = {SOURCE_H5AD.name}")
    log(f"Cells/group  = {CELLS_PER_GROUP:,}")
    log(f"Random seed  = {RANDOM_STATE}")
    log(f"Output       = {OUT_FILE.name}")

    if not SOURCE_H5AD.exists():
        raise FileNotFoundError(f"Source h5ad not found: {SOURCE_H5AD}")

    log(f"\nCollecting pilot barcodes from {len(PILOT_FILES)} files...")
    pilot_barcodes = collect_pilot_barcodes(PILOT_FILES)
    log(f"  Total unique pilot barcodes: {len(pilot_barcodes):,}")

    log(f"\nLoading source cohort: {SOURCE_H5AD.name}")
    t0 = time.time()
    adata = sc.read_h5ad(SOURCE_H5AD)
    log(f"  Loaded in {time.time()-t0:.1f}s")
    log(f"  Source shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if GROUP_COL not in adata.obs.columns:
        raise RuntimeError(
            f"Missing '{GROUP_COL}' in source obs.\n"
            f"Available: {list(adata.obs.columns)}"
        )
    if DISEASE_COL not in adata.obs.columns:
        raise RuntimeError(f"Missing '{DISEASE_COL}' in source obs.")

    adata.obs[GROUP_COL] = adata.obs[GROUP_COL].astype(str).str.strip().str.lower()

    if pilot_barcodes:
        n_before  = adata.n_obs
        keep_mask = ~adata.obs.index.isin(pilot_barcodes)
        adata     = adata[keep_mask].copy()
        n_removed = n_before - adata.n_obs
        log(f"\nExcluded {n_removed:,} pilot cells; {adata.n_obs:,} cells remaining")
    else:
        log(f"\nNo pilot barcodes to exclude; using full cohort ({adata.n_obs:,} cells)")

    valid_disease     = ~adata.obs[DISEASE_COL].isna()
    n_dropped_disease = int((~valid_disease).sum())
    if n_dropped_disease > 0:
        log(f"  Dropping {n_dropped_disease:,} cells with missing disease label")
        adata = adata[valid_disease].copy()

    adata.obs["disease_binary"] = adata.obs[DISEASE_COL].apply(to_binary_disease)

    log(f"\nAvailable cells per ethnicity group:")
    counts = adata.obs[GROUP_COL].value_counts().sort_index()
    for grp, n in counts.items():
        marker = "" if n >= CELLS_PER_GROUP else f"  [INSUFFICIENT: need {CELLS_PER_GROUP:,}]"
        log(f"  {grp}: {n:,}{marker}")

    insufficient = counts[counts < CELLS_PER_GROUP]
    if len(insufficient) > 0:
        raise RuntimeError(
            f"Cannot sample {CELLS_PER_GROUP:,} cells from: {insufficient.to_dict()}"
        )

    log(f"\nStratified sampling {CELLS_PER_GROUP:,} cells per group "
        "(disease/normal proportional within group)...")

    selected_idx = []
    for grp in sorted(counts.index):
        grp_mask = (adata.obs[GROUP_COL] == grp).to_numpy()
        grp_obs  = adata.obs[grp_mask]
        grp_idx  = adata.obs.index[grp_mask].to_numpy()

        disease_mask    = (grp_obs["disease_binary"] == "disease").to_numpy()
        normal_mask     = (grp_obs["disease_binary"] == "normal").to_numpy()
        n_disease_avail = int(disease_mask.sum())
        n_normal_avail  = int(normal_mask.sum())
        n_total_avail   = n_disease_avail + n_normal_avail

        if n_total_avail == 0:
            log(f"  {grp}: 0 cells with disease label, skipping")
            continue

        target_disease = round(CELLS_PER_GROUP * n_disease_avail / n_total_avail)
        target_normal  = CELLS_PER_GROUP - target_disease
        target_disease = min(target_disease, n_disease_avail)
        target_normal  = min(target_normal,  n_normal_avail)

        chosen = []
        if target_disease > 0:
            chosen.extend(rng.choice(
                grp_idx[disease_mask], size=target_disease, replace=False
            ).tolist())
        if target_normal > 0:
            chosen.extend(rng.choice(
                grp_idx[normal_mask], size=target_normal, replace=False
            ).tolist())

        shortfall = CELLS_PER_GROUP - len(chosen)
        if shortfall > 0:
            remaining_idx = np.setdiff1d(grp_idx, np.array(chosen), assume_unique=True)
            if len(remaining_idx) >= shortfall:
                chosen.extend(rng.choice(
                    remaining_idx, size=shortfall, replace=False
                ).tolist())
                log(f"  {grp}: topped up {shortfall} cells (single disease class)")

        selected_idx.extend(chosen)
        log(f"  {grp}: {target_disease} disease + {target_normal} normal = {len(chosen)} cells")

    val = adata[selected_idx].copy()
    log(f"\nValidation set built: {val.n_obs:,} cells x {val.n_vars:,} genes")
    val.obs["source"] = "real"

    log(f"\nDisease distribution:")
    for label, n in val.obs[DISEASE_COL].value_counts().items():
        log(f"  {label}: {n:,}")

    log(f"\nBinary disease distribution:")
    for label, n in val.obs["disease_binary"].value_counts().items():
        log(f"  {label}: {n:,}")

    log(f"\nDisease distribution by ethnicity (binary):")
    cross = val.obs.groupby([GROUP_COL, "disease_binary"]).size().reset_index(name="count")
    for _, row in cross.iterrows():
        log(f"  [{row[GROUP_COL]}] {row['disease_binary']}: {row['count']:,}")

    if pilot_barcodes:
        overlap = set(val.obs.index) & pilot_barcodes
        if overlap:
            raise RuntimeError(f"Validation set has {len(overlap)} cells overlapping with pilot.")
        log(f"\nOverlap check passed: 0 cells shared with pilot")

    log(f"\nWriting -> {OUT_FILE.name}")
    val.write_h5ad(OUT_FILE)
    log(f"  File size: {OUT_FILE.stat().st_size / 1024**2:.1f} MB")

    log(f"\n{'='*70}")
    log("STEP 0c COMPLETE (ETHNICITY, Geneformer)")
    log(f"  Output -> {OUT_FILE}")
    log(f"  Log    -> {LOGFILE.name}")
    log(f"  Next   -> step0b (will exclude these barcodes from training)")
    log_fh.close()


if __name__ == "__main__":
    main()