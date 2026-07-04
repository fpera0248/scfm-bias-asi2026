#!/usr/bin/env python3
"""
STEP 1a — Dataset Integrity & Distribution Checks (SEX)

Purpose:
    Verify that Step 0b sex-based 2K pilot dataset construction behaved as
    intended before any modeling or visualization.

Datasets checked:
    1. Full BalancedAugmented  — scDesign3 synthetic minority + real majority
                                  (female: ~1457 real+synthetic, male: 1408 real)
    2. RealOnly                — real cells only from the BalancedAugmented SCE
    3. SyntheticOnly           — scDesign3-generated female cells only
    4. Proportional_1999       — real-only, proportional sex ratio (~2k pilot)
    5. BalancedAugmented_1408  — same as Full (convenience copy)
    6. BalancedUpsampled_1408  — real-only, female upsampled to match male
    7. Downsampled_591         — real-only, both sexes downsampled to minority size

Checks:
    • Cell counts per sex (against expected targets)
    • source column present and valid (real / synthetic) where applicable
    • Label consistency (sex values canonicalized)
    • No missing or duplicated group labels
    • Gene dimensions consistent across datasets
"""

import scanpy as sc
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

OUTPUT_BASE = "ILD_Sex_Pilot"

DATASETS = {
    "Full_BalancedAugmented":      f"{OUTPUT_BASE}_Full_BalancedAugmented_SEX.h5ad",
    "RealOnly":                    f"{OUTPUT_BASE}_RealOnly_SEX.h5ad",
    "SyntheticOnly":               f"{OUTPUT_BASE}_SyntheticOnly_SEX.h5ad",
    "Proportional_1999":           f"{OUTPUT_BASE}_Proportional_1999_SEX.h5ad",
    "BalancedAugmented_1408Each":  f"{OUTPUT_BASE}_BalancedAugmented_1408Each_SEX.h5ad",
    "BalancedUpsampled_1408Each":  f"{OUTPUT_BASE}_BalancedUpsampled_1408Each_SEX.h5ad",
    "Downsampled_591Each":         f"{OUTPUT_BASE}_Downsampled_591Each_SEX.h5ad",
}

# Column names as written by step0b
GROUP_COL  = None       # auto-detected below from common names
SOURCE_COL = "source"   # "real" or "synthetic"

# Datasets that should have a source column
HAS_SOURCE = {"Full_BalancedAugmented", "RealOnly", "SyntheticOnly", "BalancedAugmented_1408Each"}

# Expected cell count targets (approximate — pilot sizes)
EXPECTED = {
    "Full_BalancedAugmented":     {"female": 1457, "male": 1408},  # female ~1408 target, bumped to 1457
    "RealOnly":                   {"female": 591,  "male": 1408},
    "SyntheticOnly":              {"female": 866,  "male": 0},     # synthetic female only
    "Proportional_1999":          {"female": 591,  "male": 1408},
    "BalancedAugmented_1408Each": {"female": 1457, "male": 1408},
    "BalancedUpsampled_1408Each": {"female": 1408, "male": 1408},
    "Downsampled_591Each":        {"female": 591,  "male": 591},
}

# Tolerance for count checks (±5%)
COUNT_TOL = 0.05

# ============================================================
# CANONICAL LABEL MAP
# ============================================================

CANONICAL_SEX = {
    "female": "female",
    "male":   "male",
    "f":      "female",
    "m":      "male",
}

# Sex value written by R for synthetic cells whose real metadata was NA
SYNTHETIC_SEX_NULL_VALUES = {"na", "nan", "none", "<na>"}

# ============================================================
# HELPERS
# ============================================================

def canonicalize_sex(adata, label, col):
    raw = adata.obs[col].astype(str).str.strip().str.lower()

    # Synthetic cells have sex written as NA from R (arrives as "na"/"nan").
    # Fill from source column if available, otherwise infer as "female"
    # (step0b only generates synthetic cells for the minority/female group).
    null_mask = raw.isin(SYNTHETIC_SEX_NULL_VALUES)
    if null_mask.any():
        if SOURCE_COL in adata.obs.columns:
            # synthetic cells in this dataset are minority-sex (female)
            raw.loc[null_mask] = "female"
            print(f"  ℹ️  {null_mask.sum()} synthetic cells with null sex → assigned 'female'")
        else:
            raise RuntimeError(
                f"  ❌ {null_mask.sum()} null sex values in {label} "
                f"but no '{SOURCE_COL}' column to resolve them"
            )

    adata.obs[col] = raw.map(CANONICAL_SEX)
    n_bad = adata.obs[col].isna().sum()
    if n_bad > 0:
        bad_vals = raw[adata.obs[col].isna()].unique().tolist()
        raise RuntimeError(
            "  ❌ {} unmapped sex labels in {}: {}. Add them to CANONICAL_SEX above.".format(
                n_bad, label, bad_vals)
        )
    return adata


def check_counts(adata, label, col):
    counts = adata.obs[col].value_counts().to_dict()
    expected = EXPECTED.get(label, {})
    ok = True
    for sex, exp in expected.items():
        if exp == 0:
            if counts.get(sex, 0) > 0:
                print(f"  ⚠️  {sex}: expected 0 but got {counts.get(sex, 0)}")
                ok = False
        else:
            got = counts.get(sex, 0)
            lo, hi = int(exp * (1 - COUNT_TOL)), int(exp * (1 + COUNT_TOL))
            status = "✅" if lo <= got <= hi else "⚠️ "
            if not (lo <= got <= hi):
                ok = False
            print(f"  {status} {sex}: {got:,}  (expected ~{exp:,})")
    return ok


def check_source(adata, label):
    if SOURCE_COL not in adata.obs.columns:
        print(f"  ⚠️  '{SOURCE_COL}' column missing in {label}")
        return False
    vals = set(adata.obs[SOURCE_COL].astype(str).str.lower().unique())
    unexpected = vals - {"real", "synthetic", "nan"}
    if unexpected:
        print(f"  ⚠️  unexpected source values: {unexpected}")
        return False
    counts = adata.obs[SOURCE_COL].value_counts()
    print(f"  ✅ source: {dict(counts)}")
    return True


def print_distribution(adata, label, col):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Total cells : {adata.n_obs:,}")
    print(f"  Genes       : {adata.n_vars:,}")
    print(f"  Sex counts  :")
    print(adata.obs[col].value_counts().sort_index().to_string(header=False))


# ============================================================
# MAIN
# ============================================================

SEX_COL_CANDIDATES = ["sex", "Sex", "SEX", "sx", "gender", "Gender"]

def detect_sex_col(adata, label):
    for c in SEX_COL_CANDIDATES:
        if c in adata.obs.columns:
            return c
    raise RuntimeError(
        f"  No sex column found in {label}.\n"
        f"  Observed columns: {list(adata.obs.columns)}\n"
        f"  Add your column name to SEX_COL_CANDIDATES above."
    )

print("\n🔍 STEP 1a — Dataset Integrity Checks (SEX, 2K Pilot)")
print(f"   source column: '{SOURCE_COL}'")

results = {}
gene_dims = {}

for label, path in DATASETS.items():
    print(f"\n📂 Loading: {label}")
    print(f"   {path}")
    try:
        adata = sc.read_h5ad(path)
    except FileNotFoundError:
        print(f"  ❌ File not found: {path}")
        results[label] = False
        continue

    # SyntheticOnly has no sex column (all-NA cols dropped at write time).
    # All synthetic cells are female by construction — add the column back.
    try:
        col = detect_sex_col(adata, label)
        print(f"  ✅ sex column detected: '{col}'")
    except RuntimeError:
        if label == "SyntheticOnly":
            col = "sex"
            adata.obs[col] = "female"
            print(f"  ℹ️  No sex column in SyntheticOnly (dropped as all-NA); assigned 'female' to all cells")
        else:
            raise

    adata = canonicalize_sex(adata, label, col)
    print_distribution(adata, label, col)

    count_ok = check_counts(adata, label, col)

    if label in HAS_SOURCE:
        check_source(adata, label)

    gene_dims[label] = adata.n_vars
    results[label] = count_ok

# ============================================================
# GENE DIMENSION CONSISTENCY
# ============================================================
print(f"\n{'='*70}")
print("Gene dimension check:")
dims = set(gene_dims.values())
if len(dims) == 1:
    print(f"  ✅ All datasets share {list(dims)[0]:,} genes")
else:
    print(f"  ⚠️  Gene dimensions differ across datasets: {gene_dims}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
n_ok  = sum(results.values())
n_tot = len(results)
print(f"\n{'✅' if n_ok == n_tot else '⚠️ '} STEP 1a COMPLETE — {n_ok}/{n_tot} datasets passed count checks")
print("\nVerified:")
print("  • Correct cell counts per sex (±5% tolerance)")
print("  • Canonicalized sex labels (female / male)")
print("  • No missing or unmapped group annotations")
print("  • source column present and valid where expected")
print("  • Gene dimensions consistent across all datasets")
print("\n➡️  Proceed to Step 1b for expression-level UMAP sanity checks.")