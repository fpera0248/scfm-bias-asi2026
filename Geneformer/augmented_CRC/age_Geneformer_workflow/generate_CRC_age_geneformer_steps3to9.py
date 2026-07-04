#!/usr/bin/env python3
"""
Generates steps 3a-9 for CRC age Geneformer workflow.
Run on Oscar: python3 generate_CRC_age_geneformer_steps3to9.py
"""
import os, subprocess

BASE_ILD = "/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow"
BASE_CRC = "/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/age_Geneformer_workflow"
OUTPUT_BASE = "CRC_Age_Pilot"
VAL_FILE    = "CRC_Age_External_Validation_9402"

os.makedirs(BASE_CRC, exist_ok=True)
os.makedirs(f"{BASE_CRC}/logs", exist_ok=True)

def adapt(src: str) -> str:
    return (src
        # paths
        .replace(BASE_ILD, BASE_CRC)
        # output base
        .replace("ILD_Age_Pilot", OUTPUT_BASE)
        # dataset size keys
        .replace("BalancedAugmented_1262Each", "BalancedAugmented_520Each")
        .replace("BalancedUpsampled_1262Each", "BalancedUpsampled_520Each")
        .replace("Proportional_2495",          "Proportional_1999")
        .replace("Downsampled_25Each",          "Downsampled_99Each")
        # validation file
        .replace("ILD_Age_External_Validation_10500", VAL_FILE)
        # underrep group — ILD is 10_19, CRC is 30_39
        .replace('UNDERREP_GROUP = "10_19"', 'UNDERREP_GROUP = "30_39"')
        # KNOWN_GROUPS — remove ILD-only age bins
        .replace(
            'KNOWN_GROUPS = ["10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_79"]',
            'KNOWN_GROUPS = ["30_39", "40_49", "50_59", "60_69", "70_79"]')
        # step9 UMAP stems
        .replace(
            '"Proportional_1999":          "CRC_Age_Pilot_Proportional_1999_AGE"',
            '"Proportional_1999":          "CRC_Age_Pilot_Proportional_1999_AGE"')
        # step9 short labels
        .replace('"Proportional_2495":          "Proportional\\n(2,495 cells)"',
                 '"Proportional_1999":          "Proportional\\n(1,999 cells)"')
        .replace('"BalancedAugmented_1262Each": "scDesign3\\nAugmented\\n(1,262/bin)"',
                 '"BalancedAugmented_520Each":  "scDesign3\\nAugmented\\n(520/bin)"')
        .replace('"BalancedUpsampled_1262Each": "Upsampled\\n(1,262/bin)"',
                 '"BalancedUpsampled_520Each":  "Upsampled\\n(520/bin)"')
        .replace('"Downsampled_25Each":         "Downsampled\\n(25/bin)"',
                 '"Downsampled_99Each":         "Downsampled\\n(99/bin)"')
        # step9 palette keys
        .replace(
            '"Proportional_2495":          "#4C72B0"',
            '"Proportional_1999":          "#4C72B0"')
        .replace(
            '"BalancedAugmented_1262Each": "#DD8452"',
            '"BalancedAugmented_520Each":  "#DD8452"')
        .replace(
            '"BalancedUpsampled_1262Each": "#55A868"',
            '"BalancedUpsampled_520Each":  "#55A868"')
        .replace(
            '"Downsampled_25Each":         "#C44E52"',
            '"Downsampled_99Each":         "#C44E52"')
        # step9 DATASET_ORDER and DATASET_KEYS
        .replace(
            '    "Proportional_2495",\n    "BalancedAugmented_1262Each",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "Proportional_1999",\n    "BalancedAugmented_520Each",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        .replace(
            '    "BalancedAugmented_1262Each",\n    "Proportional_2495",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "BalancedAugmented_520Each",\n    "Proportional_1999",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        # step9 UMAP_DATASETS
        .replace(
            '    "Proportional_2495":          f"{OUTPUT_BASE}_Proportional_2495_AGE"',
            '    "Proportional_1999":          f"{OUTPUT_BASE}_Proportional_1999_AGE"')
        .replace(
            '    "BalancedAugmented_1262Each": f"{OUTPUT_BASE}_BalancedAugmented_1262Each_AGE"',
            '    "BalancedAugmented_520Each":  f"{OUTPUT_BASE}_BalancedAugmented_520Each_AGE"')
        .replace(
            '    "BalancedUpsampled_1262Each": f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_AGE"',
            '    "BalancedUpsampled_520Each":  f"{OUTPUT_BASE}_BalancedUpsampled_520Each_AGE"')
        .replace(
            '    "Downsampled_25Each":         f"{OUTPUT_BASE}_Downsampled_25Each_AGE"',
            '    "Downsampled_99Each":         f"{OUTPUT_BASE}_Downsampled_99Each_AGE"')
        # step9 UMAP_DISPLAY
        .replace(
            '"Proportional_2495":          "Proportional (2,495 cells — real only)"',
            '"Proportional_1999":          "Proportional (1,999 cells — real only)"')
        .replace(
            '"BalancedAugmented_1262Each": "Balanced Augmented (1,262/bin — scDesign3)"',
            '"BalancedAugmented_520Each":  "Balanced Augmented (520/bin — scDesign3)"')
        .replace(
            '"BalancedUpsampled_1262Each": "Balanced Upsampled (1,262/bin — real only)"',
            '"BalancedUpsampled_520Each":  "Balanced Upsampled (520/bin — real only)"')
        .replace(
            '"Downsampled_25Each":         "Downsampled (25/bin — real only)"',
            '"Downsampled_99Each":         "Downsampled (99/bin — real only)"')
        # kNN mixing note — 5 bins not 7
        .replace(
            "random expected kNN mixing is ~0.86",
            "random expected kNN mixing is ~0.80")
        .replace(
            "(6 out of 7 neighbours expected from other bins by chance).",
            "(4 out of 5 neighbours expected from other bins by chance).")
        .replace(
            "Values well below 0.86 indicate age-specific clustering.",
            "Values well below 0.80 indicate age-specific clustering.")
        # step7 note
        .replace(
            "NOTE: With 7 age bins, random expected kNN mixing is ~0.86",
            "NOTE: With 5 age bins, random expected kNN mixing is ~0.80")
        # step3a scIB iLISI hline
        .replace(
            "hline  = 7.0 if metric == \"iLISI\" else None",
            "hline  = 5.0 if metric == \"iLISI\" else None")
        # step9 iLISI hline label
        .replace(
            '"Theoretical max (7 groups, fully mixed)"',
            '"Theoretical max (5 groups, fully mixed)"')
        # CRC label in titles
        .replace("(AGE, Geneformer)", "(CRC AGE, Geneformer)")
        .replace("(AGE, Pilot)", "(CRC AGE, Pilot)")
        .replace("Age Fairness (Pilot, Geneformer)", "CRC Age Fairness (Pilot, Geneformer)")
        # step4 validation file path fallback
    )

# ILD scripts we need
script_names = {
    "step3a": "step3a_benchmark_geneformer_age.py",
    "step3b": "step3b_label_propagation_geneformer_age.py",
    "step4":  "step4_external_validation_geneformer_age.py",
    "step4a": "step4a_downstream_geneformer_age.py",
    "step4b": "step4b_robustness_geneformer_age.py",
    "step5":  "step5_fairness_geneformer_age.py",
    "step6":  "step6_per_age_diagnostics_geneformer.py",
    "step7":  "step7_representation_diagnostics_geneformer_age.py",
    "step8":  "step8_age_conditioned_disease_geneformer.py",
    "step9":  "step9_visualizations_geneformer_age.py",
}

# output filenames for CRC
out_names = {
    "step3a": "step3a_benchmark_geneformer_CRC_age.py",
    "step3b": "step3b_label_propagation_geneformer_CRC_age.py",
    "step3d": "step3d_verify_validation_CRC_age.py",
    "step4":  "step4_external_validation_geneformer_CRC_age.py",
    "step4a": "step4a_downstream_geneformer_CRC_age.py",
    "step4b": "step4b_robustness_geneformer_CRC_age.py",
    "step5":  "step5_fairness_geneformer_CRC_age.py",
    "step6":  "step6_per_age_diagnostics_geneformer_CRC.py",
    "step7":  "step7_representation_diagnostics_geneformer_CRC_age.py",
    "step8":  "step8_age_conditioned_disease_geneformer_CRC.py",
    "step9":  "step9_visualizations_geneformer_CRC_age.py",
}

# read, adapt, write steps 3a-9
for tag, ild_fname in script_names.items():
    ild_path = f"{BASE_ILD}/{ild_fname}"
    crc_path = f"{BASE_CRC}/{out_names[tag]}"
    with open(ild_path) as f:
        src = f.read()
    adapted = adapt(src)
    # additional per-step fixes
    if tag == "step4":
        # fix validation file to use CRC path with fallback
        adapted = adapted.replace(
            f'VALIDATION_FILE = BASE / "{VAL_FILE}_geneformer.h5ad"',
            f'VALIDATION_FILE = BASE / "{VAL_FILE}_geneformer.h5ad"\n'
            f'if not VALIDATION_FILE.exists():\n'
            f'    VALIDATION_FILE = BASE / "step2a_embeddings" / "{VAL_FILE}_geneformer.h5ad"'
        )
    if tag == "step3a":
        # fix silhouette_batch API for scib_metrics version on Oscar
        adapted = adapted.replace("silhouette_batch=True,", "")
        adapted = adapted.replace("silhouette_batch=False,", "")
        # fix BASE path in EMBDIR
        adapted = adapted.replace(
            "EMBDIR = BASE",
            "EMBDIR = BASE")
    with open(crc_path, "w") as f:
        f.write(adapted)
    print(f"Written: {out_names[tag]}  ({len(adapted.splitlines())} lines)")

# write step3d — verify validation embedding
step3d = f'''#!/usr/bin/env python3
"""
STEP 3d — Verify External Validation Embedding exists (CRC AGE)
Validation was embedded during step2a. This step confirms it and symlinks.
"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE     = pathlib.Path("{BASE_CRC}")
EMB_KEY  = "X_geneformer"
MIN_UNIQUE = 10

# check root dir first, then step2a_embeddings subdir
candidates = [
    BASE / "{VAL_FILE}_geneformer.h5ad",
    BASE / "step2a_embeddings" / "{VAL_FILE}_geneformer.h5ad",
]

OUT_FILE = None
for c in candidates:
    if c.exists():
        OUT_FILE = c
        break

print(f"STEP 3d -- Verify CRC AGE validation embedding", flush=True)

if OUT_FILE is None:
    print(f"  ERROR: Validation embedding not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: {{EMB_KEY}} missing from {{OUT_FILE.name}}", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {{ad.n_obs:,}}  |  Unique embedding rows: {{n_unique:,}}", flush=True)
if n_unique <= MIN_UNIQUE:
    print(f"  ERROR: Degenerate embedding.", flush=True)
    sys.exit(1)

# ensure it exists in root dir for downstream scripts
root_copy = BASE / "{VAL_FILE}_geneformer.h5ad"
if not root_copy.exists():
    shutil.copy(str(OUT_FILE), str(root_copy))
    print(f"  Copied to root: {{root_copy.name}}", flush=True)

print("  OK -- validation embedding valid.", flush=True)
print("STEP 3d COMPLETE", flush=True)
'''
with open(f"{BASE_CRC}/{out_names['step3d']}", "w") as f:
    f.write(step3d)
print(f"Written: {out_names['step3d']}")

# ── slurm files ────────────────────────────────────────────────────────────
slurm_specs = [
    ("step3a", out_names["step3a"], "batch", "64G", "4:00:00"),
    ("step3b", out_names["step3b"], "batch", "32G", "2:00:00"),
    ("step3d", out_names["step3d"], "batch", "16G", "0:30:00"),
    ("step4",  out_names["step4"],  "batch", "32G", "4:00:00"),
    ("step4a", out_names["step4a"], "batch", "64G", "6:00:00"),
    ("step4b", out_names["step4b"], "batch", "64G", "6:00:00"),
    ("step5",  out_names["step5"],  "batch", "64G", "6:00:00"),
    ("step6",  out_names["step6"],  "batch", "64G", "6:00:00"),
    ("step7",  out_names["step7"],  "batch", "64G", "4:00:00"),
    ("step8",  out_names["step8"],  "batch", "32G", "2:00:00"),
    ("step9",  out_names["step9"],  "batch", "64G", "4:00:00"),
]

ENV    = "scfoundation_gpu"
MODULE = "miniforge3/25.3.0-3"

for tag, script, partition, mem, tlimit in slurm_specs:
    slurm = f"""#!/bin/bash
#SBATCH --job-name=CRC_age_{tag}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={tlimit}
#SBATCH --output={BASE_CRC}/logs/CRC_age_{tag}_%j.out
#SBATCH --error={BASE_CRC}/logs/CRC_age_{tag}_%j.err

module purge
module load {MODULE}
source activate {ENV}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd {BASE_CRC}
python {script}
echo "Exit: $?"
"""
    slurm_path = f"{BASE_CRC}/{tag}_CRC_age_gf.slurm"
    with open(slurm_path, "w") as f:
        f.write(slurm)
    print(f"Written: {tag}_CRC_age_gf.slurm")

# ── submit script ──────────────────────────────────────────────────────────
submit = f"""#!/bin/bash
# Submit CRC age Geneformer steps 3a-9 with dependencies.
# Steps 4a-9 wait for step 3b to complete.
# Step 9 waits for all of 4a, 4b, 5, 6, 7, 8.
# Usage: bash submit_CRC_age_steps.sh

BASE={BASE_CRC}
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {{
  local name=$1; local script=$2; local mem=$3; local time=$4; local deps=$5
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${{deps}}"
  local slurm=$BASE/${{name}}_CRC_age_gf.slurm
  local id=$(sbatch $dep_flag $slurm | awk '{{print $NF}}')
  echo "$name: $id" >&2
  echo $id
}}

# step 3a (scIB) — no deps
J3A=$(submit_job "step3a" "{out_names['step3a']}" "64G" "4:00:00" "")

# step 3b (label propagation) — no deps (parallel with 3a)
J3B=$(submit_job "step3b" "{out_names['step3b']}" "32G" "2:00:00" "")

# step 3d (verify validation) — no deps
J3D=$(submit_job "step3d" "{out_names['step3d']}" "16G" "0:30:00" "")

# step 4 (external validation) — needs 3b + 3d
J4=$(submit_job "step4" "{out_names['step4']}" "32G" "4:00:00" "${{J3B}}:${{J3D}}")

# steps 4a, 4b, 5, 6, 7, 8 — each needs step 3b
J4A=$(submit_job "step4a" "{out_names['step4a']}" "64G" "6:00:00" "${{J3B}}")
J4B=$(submit_job "step4b" "{out_names['step4b']}" "64G" "6:00:00" "${{J3B}}")
J5=$(submit_job  "step5"  "{out_names['step5']}"  "64G" "6:00:00" "${{J3B}}")
J6=$(submit_job  "step6"  "{out_names['step6']}"  "64G" "6:00:00" "${{J3B}}")
J7=$(submit_job  "step7"  "{out_names['step7']}"  "64G" "4:00:00" "${{J3B}}")
J8=$(submit_job  "step8"  "{out_names['step8']}"  "32G" "2:00:00" "${{J3B}}")

# step 9 — needs all prior analysis steps
ALL="${{J3A}}:${{J4}}:${{J4A}}:${{J4B}}:${{J5}}:${{J6}}:${{J7}}:${{J8}}"
J9=$(submit_job "step9" "{out_names['step9']}" "64G" "4:00:00" "${{ALL}}")

echo ""
echo "Job chain submitted:"
echo "  step3a : $J3A"
echo "  step3b : $J3B"
echo "  step3d : $J3D"
echo "  step4  : $J4  (after 3b+3d)"
echo "  step4a : $J4A (after 3b)"
echo "  step4b : $J4B (after 3b)"
echo "  step5  : $J5  (after 3b)"
echo "  step6  : $J6  (after 3b)"
echo "  step7  : $J7  (after 3b)"
echo "  step8  : $J8  (after 3b)"
echo "  step9  : $J9  (after all)"
echo ""
echo "Monitor: squeue -u fperalta"
"""
submit_path = f"{BASE_CRC}/submit_CRC_age_steps.sh"
with open(submit_path, "w") as f:
    f.write(submit)
print(f"\nWritten: submit_CRC_age_steps.sh")

# ── verify ─────────────────────────────────────────────────────────────────
print("\nSpot-check key substitutions in step3b and step8:")
for tag in ["step3b", "step8"]:
    path = f"{BASE_CRC}/{out_names[tag]}"
    result = subprocess.run(
        ["grep", "-n", r"OUTPUT_BASE\|DATASETS\|UNDERREP\|BASE =\|Proportional\|520\|1999\|99Each\|9402\|KNOWN_GROUP"],
        stdin=open(path), capture_output=True, text=True)
    print(f"\n--- {out_names[tag]} ---")
    print(result.stdout[:1200])

print("\nAll done. Run:")
print(f"  bash {submit_path}")
