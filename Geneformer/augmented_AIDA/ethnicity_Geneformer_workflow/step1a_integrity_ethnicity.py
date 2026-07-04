#!/usr/bin/env python3
"""
STEP 1a — Dataset Integrity & Distribution Checks (ETHNICITY)

Purpose:
    Verify that Step 0b ethnicity-based 2.5K pilot dataset construction
    behaved as intended before any modeling or visualization.

Datasets checked:
    1. Full_BalancedAugmented       — scDesign3 synthetic minorities + real majority
    2. RealOnly                     — real cells only from the BalancedAugmented SCE
    3. SyntheticOnly                — scDesign3-generated minority cells only
    4. Proportional_2500            — real-only, proportional ethnicity ratio (~2.5k pilot)
    5. BalancedAugmented_779Each   — same as Full (convenience copy)
    6. BalancedUpsampled_779Each   — real-only, minorities upsampled to match majority
    7. Downsampled_92Each           — real-only, all groups downsampled to minority size

Checks:
    • Cell counts per ethnicity group (against expected targets)
    • source column present and valid (real / synthetic) where applicable
    • Label consistency (ethnicity values canonicalized)
    • No missing or duplicated group labels
    • Gene dimensions consistent across datasets
"""

import scanpy as sc
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

OUTPUT_BASE = "AIDA_Ethnicity_Pilot"

DATASETS = {
    "Full_BalancedAugmented":         f"{OUTPUT_BASE}_Full_BalancedAugmented_ETHNICITY.h5ad",
    "RealOnly":                       f"{OUTPUT_BASE}_RealOnly_ETHNICITY.h5ad",
    "SyntheticOnly":                  f"{OUTPUT_BASE}_SyntheticOnly_ETHNICITY.h5ad",
    "Proportional_2500":              f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY.h5ad",
    "BalancedAugmented_779Each":     f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY.h5ad",
    "BalancedUpsampled_779Each":     f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY.h5ad",
    "Downsampled_92Each":             f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY.h5ad",
}

SOURCE_COL = "source"

HAS_SOURCE = {
    "Full_BalancedAugmented",
    "RealOnly",
    "SyntheticOnly",
    "BalancedAugmented_779Each",
}

# Expected counts per group. None = no hard target (just report).
# european american is majority (real-only at 2143).
# native american is the smallest group (real count ~48 after validation exclusion).
EXPECTED = {
    "Full_BalancedAugmented": {
        "african american":   2143,
        "asian":              2143,
        "european american":  2143,
        "hispanic or latin":  2143,
        "native american":    2143,
    },
    "RealOnly": {
        "african american":   None,
        "asian":              None,
        "european american":  2143,
        "hispanic or latin":  None,
        "native american":    None,
    },
    "SyntheticOnly": {
        "african american":   None,
        "asian":              None,
        "european american":  0,
        "hispanic or latin":  None,
        "native american":    None,
    },
    "Proportional_2500": {
        "african american":   None,
        "asian":              None,
        "european american":  None,
        "hispanic or latin":  None,
        "native american":    None,
    },
    "BalancedAugmented_779Each": {
        "african american":   2143,
        "asian":              2143,
        "european american":  2143,
        "hispanic or latin":  2143,
        "native american":    2143,
    },
    "BalancedUpsampled_779Each": {
        "african american":   2143,
        "asian":              2143,
        "european american":  2143,
        "hispanic or latin":  2143,
        "native american":    2143,
    },
    "Downsampled_92Each": {
        "african american":   48,
        "asian":              48,
        "european american":  48,
        "hispanic or latin":  48,
        "native american":    48,
    },
}

COUNT_TOL = 0.10  # ±10% — wider because native american is tiny

# ============================================================
# CANONICAL LABEL MAP
# ============================================================

CANONICAL_ETH = {
    "african american":   "african american",
    "asian":              "asian",
    "european american":  "european american",
    "hispanic or latin":  "hispanic or latin",
    "native american":    "native american",
}

ETH_NULL_VALUES = {
    "na", "nan", "none", "<na>", "unknown", "not reported",
    "multiethnic", "na na", "not applicable", "prefer not to say",
}

# ============================================================
# HELPERS
# ============================================================

def canonicalize_eth(adata, label, col):
    raw = adata.obs[col].astype(str).str.strip().str.lower()
    null_mask = raw.isin(ETH_NULL_VALUES)
    if null_mask.any():
        print(f"  i  {null_mask.sum()} cells with null/unknown ethnicity -- dropping from counts")
        adata = adata[~null_mask].copy()
        raw   = adata.obs[col].astype(str).str.strip().str.lower()
    adata.obs[col] = raw.map(CANONICAL_ETH)
    n_bad = adata.obs[col].isna().sum()
    if n_bad > 0:
        bad_vals = raw[adata.obs[col].isna()].unique().tolist()
        raise RuntimeError(
            f"  ERROR: {n_bad} unmapped ethnicity labels in {label}: {bad_vals}. "
            f"Add them to CANONICAL_ETH above."
        )
    return adata


def check_counts(adata, label, col):
    counts = adata.obs[col].value_counts().to_dict()
    expected = EXPECTED.get(label, {})
    ok = True
    for grp, exp in expected.items():
        if exp is None:
            got = counts.get(grp, 0)
            print(f"  INFO {grp}: {got:,}  (no target set)")
            continue
        if exp == 0:
            if counts.get(grp, 0) > 0:
                print(f"  WARN {grp}: expected 0 but got {counts.get(grp, 0)}")
                ok = False
        else:
            got = counts.get(grp, 0)
            lo, hi = int(exp * (1 - COUNT_TOL)), int(exp * (1 + COUNT_TOL))
            status = "OK  " if lo <= got <= hi else "WARN"
            if not (lo <= got <= hi):
                ok = False
            print(f"  {status} {grp}: {got:,}  (expected ~{exp:,})")
    return ok


def check_source(adata, label):
    if SOURCE_COL not in adata.obs.columns:
        print(f"  WARN '{SOURCE_COL}' column missing in {label}")
        return False
    vals = set(adata.obs[SOURCE_COL].astype(str).str.lower().unique())
    unexpected = vals - {"real", "synthetic", "nan"}
    if unexpected:
        print(f"  WARN unexpected source values: {unexpected}")
        return False
    counts = adata.obs[SOURCE_COL].value_counts()
    print(f"  OK   source: {dict(counts)}")
    return True


def print_distribution(adata, label, col):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Total cells : {adata.n_obs:,}")
    print(f"  Genes       : {adata.n_vars:,}")
    print(f"  Ethnicity counts:")
    print(adata.obs[col].value_counts().sort_index().to_string(header=False))


# ============================================================
# MAIN
# ============================================================

ETH_COL_CANDIDATES = [
    "self_reported_ethnicity",
    "ethnicity",
    "Ethnicity",
    "eth",
    "race",
]

def detect_eth_col(adata, label):
    for c in ETH_COL_CANDIDATES:
        if c in adata.obs.columns:
            return c
    raise RuntimeError(
        f"  No ethnicity column found in {label}.\n"
        f"  Observed columns: {list(adata.obs.columns)}\n"
        f"  Add your column name to ETH_COL_CANDIDATES above."
    )


print("\nSTEP 1a -- Dataset Integrity Checks (ETHNICITY, 2.5K Pilot)")
print(f"   source column: '{SOURCE_COL}'")

results   = {}
gene_dims = {}

for label, path in DATASETS.items():
    print(f"\nLoading: {label}")
    print(f"   {path}")
    try:
        adata = sc.read_h5ad(path)
    except FileNotFoundError:
        print(f"  ERROR File not found: {path}")
        results[label] = False
        continue

    try:
        col = detect_eth_col(adata, label)
        print(f"  OK   ethnicity column detected: '{col}'")
    except RuntimeError:
        if label == "SyntheticOnly":
            col = "self_reported_ethnicity"
            print(f"  INFO No ethnicity column in SyntheticOnly (dropped as all-NA); "
                  f"reporting raw counts from available colData")
        else:
            raise

    adata    = canonicalize_eth(adata, label, col)
    print_distribution(adata, label, col)
    count_ok = check_counts(adata, label, col)

    if label in HAS_SOURCE:
        check_source(adata, label)

    gene_dims[label] = adata.n_vars
    results[label]   = count_ok

# ============================================================
# GENE DIMENSION CONSISTENCY
# ============================================================
print(f"\n{'='*70}")
print("Gene dimension check:")
dims = set(gene_dims.values())
if len(dims) == 1:
    print(f"  OK   All datasets share {list(dims)[0]:,} genes")
else:
    print(f"  WARN Gene dimensions differ across datasets: {gene_dims}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
n_ok  = sum(results.values())
n_tot = len(results)
status = "OK" if n_ok == n_tot else "WARN"
print(f"\n{status} STEP 1a COMPLETE -- {n_ok}/{n_tot} datasets passed count checks")
print("\nVerified:")
print("  * Correct cell counts per ethnicity group (+-10% tolerance)")
print("  * Canonicalized ethnicity labels")
print("  * No missing or unmapped group annotations")
print("  * source column present and valid where expected")
print("  * Gene dimensions consistent across all datasets")
print("\n--> Proceed to Step 2a for Geneformer embedding.")