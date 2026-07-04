#!/usr/bin/env python3
"""
Generates steps 3a-9 for CRC ethnicity Geneformer workflow.
Run on Oscar: python3 generate_CRC_eth_geneformer_steps3to9.py
"""
import os

BASE_CRC = "/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow"
OUTPUT_BASE = "CRC_Eth_Pilot"
VAL_FILE    = "CRC_Eth_External_Validation_8572"

os.makedirs(BASE_CRC, exist_ok=True)
os.makedirs(f"{BASE_CRC}/logs", exist_ok=True)

# ── common substitution helper ─────────────────────────────────────────────
def adapt(src: str) -> str:
    return (src
        .replace("/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow",
                 BASE_CRC)
        .replace("ILD_Ethnicity_Pilot", OUTPUT_BASE)
        .replace("BalancedAugmented_2143Each", "BalancedAugmented_1504Each")
        .replace("BalancedUpsampled_2143Each", "BalancedUpsampled_1504Each")
        .replace("Proportional_2497",          "Proportional_1998")
        .replace("Downsampled_48Each",          "Downsampled_90Each")
        .replace("ILD_Ethnicity_External_Validation_12500", VAL_FILE)
        .replace('UNDERREP_GROUP = "native american"',      'UNDERREP_GROUP = "african american"')
        .replace("native american",             "african american")
        .replace("Native American",             "African American")
        .replace("Native Am.",                  "African Am.")
        # step4 disease groups - ILD used AA+EA only; CRC has all 4
        .replace('DISEASE_GROUPS = {"african american", "european american"}',
                 'DISEASE_GROUPS = {"african american", "european american", "hispanic or latin", "asian"}')
        # step9 display strings
        .replace('"Proportional_1998":          "Proportional (2,497 cells — real only)"',
                 '"Proportional_1998":          "Proportional (1,998 cells — real only)"')
        .replace('"BalancedAugmented_1504Each": "Balanced Augmented (2,143/group — scDesign3)"',
                 '"BalancedAugmented_1504Each": "Balanced Augmented (1,504/group — scDesign3)"')
        .replace('"BalancedUpsampled_1504Each": "Balanced Upsampled (2,143/group — real only)"',
                 '"BalancedUpsampled_1504Each": "Balanced Upsampled (1,504/group — real only)"')
        .replace('"Downsampled_90Each":         "Downsampled (48/group — real only)"',
                 '"Downsampled_90Each":         "Downsampled (90/group — real only)"')
        # step9 short labels
        .replace('"Proportional_1998":          "Proportional\\n(2,497 cells)"',
                 '"Proportional_1998":          "Proportional\\n(1,998 cells)"')
        .replace('"BalancedAugmented_1504Each": "scDesign3\\nAugmented\\n(2,143/group)"',
                 '"BalancedAugmented_1504Each": "scDesign3\\nAugmented\\n(1,504/group)"')
        .replace('"BalancedUpsampled_1504Each": "Upsampled\\n(2,143/group)"',
                 '"BalancedUpsampled_1504Each": "Upsampled\\n(1,504/group)"')
        .replace('"Downsampled_90Each":         "Downsampled\\n(48/group)"',
                 '"Downsampled_90Each":         "Downsampled\\n(90/group)"')
        # step9 UMAP_DATASETS stems
        .replace('"Proportional_1998":          "ILD_Ethnicity_Pilot_Proportional_1998_ETHNICITY"',
                 '"Proportional_1998":          "CRC_Eth_Pilot_Proportional_1998_ETH"')
        .replace('"BalancedAugmented_1504Each": "ILD_Ethnicity_Pilot_BalancedAugmented_1504Each_ETHNICITY"',
                 '"BalancedAugmented_1504Each": "CRC_Eth_Pilot_BalancedAugmented_1504Each_ETH"')
        .replace('"BalancedUpsampled_1504Each": "ILD_Ethnicity_Pilot_BalancedUpsampled_1504Each_ETHNICITY"',
                 '"BalancedUpsampled_1504Each": "CRC_Eth_Pilot_BalancedUpsampled_1504Each_ETH"')
        .replace('"Downsampled_90Each":         "ILD_Ethnicity_Pilot_Downsampled_90Each_ETHNICITY"',
                 '"Downsampled_90Each":         "CRC_Eth_Pilot_Downsampled_90Each_ETH"')
        # step9 titles
        .replace("ETHNICITY, Geneformer", "CRC ETHNICITY, Geneformer")
        .replace("Ethnicity Fairness (Pilot, Geneformer)", "CRC Ethnicity Fairness (Pilot, Geneformer)")
        # canonical eth map — remove native american entry
        .replace(
            '    "native american":   "Native American",\n',
            '')
        # step9 reference baseline label in titles
        .replace("Proportional_1998 baseline", "Proportional_1998 baseline")
        # fix misplaced fig.savefig in step9 main()
        .replace(
            '    try:\n    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")\n        generate_individual_umaps()',
            '    try:\n        generate_individual_umaps()')
    )


# ── step 3a ───────────────────────────────────────────────────────────────────
step3a = adapt('''#!/usr/bin/env python3
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

BASE   = pathlib.Path("/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow")
INDIR  = BASE
OUTDIR = BASE / "benchmark_outputs_geneformer_ethnicity"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = "ILD_Ethnicity_Pilot"

DATASETS = {
    "BalancedAugmented_2143Each": f"{OUTPUT_BASE}_BalancedAugmented_2143Each_ETHNICITY_geneformer.h5ad",
    "Proportional_2497":          f"{OUTPUT_BASE}_Proportional_2497_ETHNICITY_geneformer.h5ad",
    "BalancedUpsampled_2143Each": f"{OUTPUT_BASE}_BalancedUpsampled_2143Each_ETHNICITY_geneformer.h5ad",
    "Downsampled_48Each":         f"{OUTPUT_BASE}_Downsampled_48Each_ETHNICITY_geneformer.h5ad",
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
        print(f"\\n  Skipping {mode} -- file not found: {fname}")
        return None
    print(f"\\n{\'=\'*70}")
    print(f"Benchmarking: {mode}")
    t0 = time.time()
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing embedding \'{EMB_KEY}\' in {fname}")
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
    print(f"   Cells: {ad.n_obs:,}  |  Ethnicity col: \'{eth_col}\'")
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
            silhouette_batch=True, ilisi_knn=True, kbet_per_label=True,
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
                    silhouette_batch=True, ilisi_knn=True, kbet_per_label=False,
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
        fh.write(f"Benchmarker __dict__ for {mode}\\n\\n")
        for k, v in sorted(bm.__dict__.items()):
            s = str(v)
            if len(s) > 15000: s = s[:15000] + " ...[truncated]"
            fh.write(f"{k}: {s}\\n\\n")
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
    print("\\nSTEP 3a -- scIB Benchmarking (ILD_Ethnicity_Pilot, Geneformer V2-316M)")
    print(f"   Output dir: {OUTDIR}")
    summary = []
    for mode, fname in DATASETS.items():
        result = run_scib(mode, fname)
        if result is not None:
            summary.append(result)
    if not summary:
        print("\\nNo datasets benchmarked successfully.")
        return
    df = pd.DataFrame(summary)
    summary_path = OUTDIR / "benchmark_summary_all_modes.csv"
    df.to_csv(summary_path, index=False)
    print(f"\\nSTEP 3a COMPLETE")
    print(f"Summary -> {summary_path.name}")
    print("\\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
''')

# ── step 3b ───────────────────────────────────────────────────────────────────
step3b = adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step3b_label_propagation_geneformer_ethnicity.py").read())

# ── step 3d ───────────────────────────────────────────────────────────────────
# For CRC, validation is already embedded by step2a. This step just verifies.
step3d = f'''#!/usr/bin/env python3
"""
STEP 3d — Verify External Validation Embedding exists (CRC ETHNICITY)
For CRC, the validation set was embedded during step2a. This step confirms
the output exists and is non-degenerate.
"""
import sys, pathlib
import numpy as np
import scanpy as sc

BASE     = pathlib.Path("{BASE_CRC}")
OUT_FILE = BASE / "step2a_embeddings" / "CRC_Eth_External_Validation_8572_geneformer.h5ad"
EMB_KEY  = "X_geneformer"
MIN_UNIQUE_ROWS = 10

print(f"STEP 3d -- Verify validation embedding", flush=True)
print(f"  Expected: {{OUT_FILE}}", flush=True)

if not OUT_FILE.exists():
    print(f"  ERROR: File not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: Embedding key \'{{EMB_KEY}}\' missing.", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {{ad.n_obs:,}}  |  Unique embedding rows: {{n_unique:,}}", flush=True)
if n_unique <= MIN_UNIQUE_ROWS:
    print(f"  ERROR: Degenerate embedding ({{n_unique}} unique rows).", flush=True)
    sys.exit(1)

print(f"  OK -- validation embedding valid.", flush=True)

# Symlink to expected location for downstream steps
LINK_TARGET = BASE / "CRC_Eth_External_Validation_8572_geneformer.h5ad"
if not LINK_TARGET.exists():
    import shutil
    shutil.copy(str(OUT_FILE), str(LINK_TARGET))
    print(f"  Copied to: {{LINK_TARGET.name}}", flush=True)

print("STEP 3d COMPLETE", flush=True)
'''

# ── steps 4 through 9 ─────────────────────────────────────────────────────────
scripts = {
    "step4_external_validation_geneformer_ethnicity.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step4_external_validation_geneformer_ethnicity.py").read()),
    "step4a_downstream_results_eth_AR_EOS_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step4a_downstream_results_eth_AR_EOS_geneformer.py").read()),
    "step4b_overfitting_stress_tests_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step4b_overfitting_stress_tests_geneformer.py").read()),
    "step5_fairness_eth_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step5_fairness_eth_geneformer.py").read()),
    "step6_per_ethnicity_diagnostics_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step6_per_ethnicity_diagnostics_geneformer.py").read()),
    "step7_representation_diagnostics_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step7_representation_diagnostics_geneformer.py").read()),
    "step8_eth_conditioned_disease_geneformer.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step8_eth_conditioned_disease_geneformer.py").read()),
    "step9_visualizations_geneformer_ethnicity.py":
        adapt(open("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/step9_visualizations_geneformer_ethnicity.py").read()),
}

# step4 — also fix VALIDATION_FILE path to use step2a_embeddings output
scripts["step4_external_validation_geneformer_ethnicity.py"] = (
    scripts["step4_external_validation_geneformer_ethnicity.py"]
    .replace(
        f'VALIDATION_FILE = BASE / "{VAL_FILE}_geneformer.h5ad"',
        f'VALIDATION_FILE = BASE / "{VAL_FILE}_geneformer.h5ad"\n'
        f'# Also check step2a_embeddings subdir\n'
        f'if not VALIDATION_FILE.exists():\n'
        f'    VALIDATION_FILE = BASE / "step2a_embeddings" / "{VAL_FILE}_geneformer.h5ad"'
    )
)

# step9 — fix reference baseline text in Proportional 2497->1998 that appears
# inside strings rather than as identifiers
scripts["step9_visualizations_geneformer_ethnicity.py"] = (
    scripts["step9_visualizations_geneformer_ethnicity.py"]
    .replace('"Proportional_1998 dataset"', '"Proportional_1998 dataset"')
    .replace("Proportional (2,497", "Proportional (1,998")
    .replace("2,143/group", "1,504/group")
    .replace("48/group", "90/group")
)

# step4b — fix KNOWN_GROUPS for CRC (no native american)
scripts["step4b_overfitting_stress_tests_geneformer.py"] = (
    scripts["step4b_overfitting_stress_tests_geneformer.py"]
    .replace(
        'KNOWN_GROUPS = ["asian", "european american", "hispanic or latin", "native american"]',
        'KNOWN_GROUPS = ["asian", "european american", "hispanic or latin", "african american"]'
    )
)

# write all files
all_scripts = {
    "step3a_benchmark_geneformer_ethnicity.py": step3a,
    "step3b_label_propagation_geneformer_ethnicity.py": step3b,
    "step3d_embed_validation_geneformer_ethnicity.py": step3d,
    **scripts,
}

for fname, content in all_scripts.items():
    fpath = f"{BASE_CRC}/{fname}"
    with open(fpath, "w") as f:
        f.write(content)
    print(f"Written: {fname}  ({len(content.splitlines())} lines)")

# ── slurm files ───────────────────────────────────────────────────────────────
slurm_specs = [
    ("step3a", "step3a_benchmark_geneformer_ethnicity.py",           "batch", "64G",  "4:00:00"),
    ("step3b", "step3b_label_propagation_geneformer_ethnicity.py",   "batch", "32G",  "2:00:00"),
    ("step3d", "step3d_embed_validation_geneformer_ethnicity.py",    "batch", "16G",  "0:30:00"),
    ("step4",  "step4_external_validation_geneformer_ethnicity.py",  "batch", "32G",  "4:00:00"),
    ("step4a", "step4a_downstream_results_eth_AR_EOS_geneformer.py", "batch", "32G",  "4:00:00"),
    ("step4b", "step4b_overfitting_stress_tests_geneformer.py",      "batch", "32G",  "4:00:00"),
    ("step5",  "step5_fairness_eth_geneformer.py",                   "batch", "32G",  "4:00:00"),
    ("step6",  "step6_per_ethnicity_diagnostics_geneformer.py",      "gpu",   "64G",  "4:00:00"),
    ("step7",  "step7_representation_diagnostics_geneformer.py",     "batch", "32G",  "4:00:00"),
    ("step8",  "step8_eth_conditioned_disease_geneformer.py",        "batch", "32G",  "2:00:00"),
    ("step9",  "step9_visualizations_geneformer_ethnicity.py",       "gpu",   "64G",  "4:00:00"),
]

for tag, script, partition, mem, time_limit in slurm_specs:
    gpu_line = "#SBATCH --gres=gpu:1\n" if partition == "gpu" else ""
    slurm = f"""#!/bin/bash
#SBATCH --job-name=CRC_eth_{tag}
#SBATCH --partition={partition}
#SBATCH --nodes=1
{gpu_line}#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={BASE_CRC}/logs/CRC_eth_{tag}_%j.out
#SBATCH --error={BASE_CRC}/logs/CRC_eth_{tag}_%j.err

module purge
module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1
cd {BASE_CRC}
python {script}
echo "Exit: $?"
"""
    slurm_path = f"{BASE_CRC}/{tag}_CRC_eth_geneformer.slurm"
    with open(slurm_path, "w") as f:
        f.write(slurm)
    print(f"Written: {tag}_CRC_eth_geneformer.slurm")

print("\nAll files written.")
print(f"\nVerify with:")
print(f"  ls -la {BASE_CRC}/*.py {BASE_CRC}/*.slurm")
print(f"\nSpot-check key substitutions:")
import subprocess
for fname in ["step3a_benchmark_geneformer_ethnicity.py",
              "step8_eth_conditioned_disease_geneformer.py"]:
    fpath = f"{BASE_CRC}/{fname}"
    result = subprocess.run(
        ["grep", "-n", "OUTPUT_BASE\|DATASETS\|UNDERREP\|BASE =\|Proportional\|1504\|1998\|90Each\|8572", fpath],
        capture_output=True, text=True)
    print(f"\n--- {fname} ---")
    print(result.stdout[:1500])
