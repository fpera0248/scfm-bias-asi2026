#!/usr/bin/env python3
"""
Generates steps 3a-9 for CRC age scGPT workflow.
Run on Oscar: python3 generate_CRC_age_scGPT_steps3to9.py
"""
import os, subprocess

BASE_ILD = "/data/scGPT/age_scGPT_workflow"
BASE_CRC = "/data/scGPT/augmented_CRC/age_scGPT_workflow"
OUTPUT_BASE = "CRC_Age_Pilot"
VAL_FILE    = "CRC_Age_External_Validation_9402"

os.makedirs(BASE_CRC, exist_ok=True)
os.makedirs(f"{BASE_CRC}/logs", exist_ok=True)
os.makedirs(f"{BASE_CRC}/step3b_labeled", exist_ok=True)

def adapt(src: str) -> str:
    return (src
        # paths
        .replace(BASE_ILD, BASE_CRC)
        # EMBDIR: ILD uses step2a_embeddings subdir; CRC files are in root
        .replace('EMBDIR  = BASE / "step2a_embeddings"', 'EMBDIR  = BASE')
        .replace('EMBDIR = BASE / "step2a_embeddings"', 'EMBDIR = BASE')
        # validation file: ILD reads from step2a_embeddings
        .replace(
            'VALIDATION_FILE = BASE / "step2a_embeddings" / "ILD_Age_External_Validation_10500_scgpt.h5ad"',
            f'VALIDATION_FILE = BASE / "{VAL_FILE}_scgpt.h5ad"')
        # output base
        .replace("ILD_Age_Pilot", OUTPUT_BASE)
        # dataset size keys
        .replace("BalancedAugmented_1262Each", "BalancedAugmented_520Each")
        .replace("BalancedUpsampled_1262Each", "BalancedUpsampled_520Each")
        .replace("Proportional_2495",          "Proportional_1999")
        .replace("Downsampled_25Each",          "Downsampled_99Each")
        # underrep group
        .replace('UNDERREP_GROUP = "10_19"', 'UNDERREP_GROUP = "30_39"')
        # known groups
        .replace(
            'KNOWN_GROUPS = ["10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_79"]',
            'KNOWN_GROUPS = ["30_39", "40_49", "50_59", "60_69", "70_79"]')
        # step9 short labels
        .replace('"Proportional_2495":          "Proportional\\n(2,495 cells)"',
                 '"Proportional_1999":          "Proportional\\n(1,999 cells)"')
        .replace('"BalancedAugmented_1262Each": "scDesign3\\nAugmented\\n(1,262/bin)"',
                 '"BalancedAugmented_520Each":  "scDesign3\\nAugmented\\n(520/bin)"')
        .replace('"BalancedUpsampled_1262Each": "Upsampled\\n(1,262/bin)"',
                 '"BalancedUpsampled_520Each":  "Upsampled\\n(520/bin)"')
        .replace('"Downsampled_25Each":         "Downsampled\\n(25/bin)"',
                 '"Downsampled_99Each":         "Downsampled\\n(99/bin)"')
        # step9 palette
        .replace('"Proportional_2495":          "#4C72B0"',
                 '"Proportional_1999":          "#4C72B0"')
        .replace('"BalancedAugmented_1262Each": "#DD8452"',
                 '"BalancedAugmented_520Each":  "#DD8452"')
        .replace('"BalancedUpsampled_1262Each": "#55A868"',
                 '"BalancedUpsampled_520Each":  "#55A868"')
        .replace('"Downsampled_25Each":         "#C44E52"',
                 '"Downsampled_99Each":         "#C44E52"')
        # step9 DATASET_ORDER / DATASET_KEYS
        .replace(
            '    "Proportional_2495",\n    "BalancedAugmented_1262Each",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "Proportional_1999",\n    "BalancedAugmented_520Each",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        .replace(
            '    "BalancedAugmented_1262Each",\n    "Proportional_2495",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "BalancedAugmented_520Each",\n    "Proportional_1999",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        # step9 UMAP_DATASETS stems
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
        .replace('"Proportional_2495":          "Proportional (2,495 cells — real only)"',
                 '"Proportional_1999":          "Proportional (1,999 cells — real only)"')
        .replace('"BalancedAugmented_1262Each": "Balanced Augmented (1,262/bin — scDesign3)"',
                 '"BalancedAugmented_520Each":  "Balanced Augmented (520/bin — scDesign3)"')
        .replace('"BalancedUpsampled_1262Each": "Balanced Upsampled (1,262/bin — real only)"',
                 '"BalancedUpsampled_520Each":  "Balanced Upsampled (520/bin — real only)"')
        .replace('"Downsampled_25Each":         "Downsampled (25/bin — real only)"',
                 '"Downsampled_99Each":         "Downsampled (99/bin — real only)"')
        # step9 UMAP suffix: ILD uses _scgpt, CRC same but from root not EMB_DIR
        .replace('gf_path = BASE / f"{ds_stem}_scgpt.h5ad"',
                 'gf_path = BASE / f"{ds_stem}_scgpt.h5ad"')
        # kNN mixing note
        .replace("random expected kNN mixing is ~0.86", "random expected kNN mixing is ~0.80")
        .replace("(6 out of 7 neighbours expected from other bins by chance).",
                 "(4 out of 5 neighbours expected from other bins by chance).")
        .replace("Values well below 0.86 indicate age-specific clustering.",
                 "Values well below 0.80 indicate age-specific clustering.")
        .replace("NOTE: With 7 age bins, random expected kNN mixing is ~0.86",
                 "NOTE: With 5 age bins, random expected kNN mixing is ~0.80")
        # remove JAX cpu env var (not needed for scGPT)
        .replace('os.environ["JAX_PLATFORM_NAME"] = "cpu"\n\n', '')
        # titles
        .replace("(AGE, scGPT)", "(CRC AGE, scGPT)")
        .replace("(AGE, Geneformer)", "(CRC AGE, scGPT)")
        .replace("Age Fairness (Pilot, Geneformer)", "CRC Age Fairness (Pilot, scGPT)")
        .replace("scGPT AGE", "CRC scGPT AGE")
    )

script_names = {
    "step3a": "step3a_benchmark_scgpt_age.py",
    "step3b": "step3b_label_propagation_scgpt_age.py",
    "step4":  "step4_external_validation_scgpt_age.py",
    "step4a": "step4a_downstream_scgpt_age.py",
    "step4b": "step4b_robustness_scgpt_age.py",
    "step5":  "step5_fairness_scgpt_age.py",
    "step6":  "step6_per_age_diagnostics_scgpt.py",
    "step7":  "step7_representation_diagnostics_scgpt_age.py",
    "step8":  "step8_age_conditioned_disease_scgpt.py",
    "step9":  "step9_visualizations_scgpt_age.py",
}

out_names = {
    "step3a": "step3a_benchmark_scgpt_CRC_age.py",
    "step3b": "step3b_label_propagation_scgpt_CRC_age.py",
    "step3d": "step3d_verify_validation_scgpt_CRC_age.py",
    "step4":  "step4_external_validation_scgpt_CRC_age.py",
    "step4a": "step4a_downstream_scgpt_CRC_age.py",
    "step4b": "step4b_robustness_scgpt_CRC_age.py",
    "step5":  "step5_fairness_scgpt_CRC_age.py",
    "step6":  "step6_per_age_diagnostics_scgpt_CRC.py",
    "step7":  "step7_representation_diagnostics_scgpt_CRC_age.py",
    "step8":  "step8_age_conditioned_disease_scgpt_CRC.py",
    "step9":  "step9_visualizations_scgpt_CRC_age.py",
}

for tag, ild_fname in script_names.items():
    ild_path = f"{BASE_ILD}/{ild_fname}"
    crc_path = f"{BASE_CRC}/{out_names[tag]}"
    with open(ild_path) as f:
        src = f.read()
    adapted = adapt(src)
    # step3a: silhouette_batch compat fix
    if tag == "step3a":
        adapted = adapted.replace("silhouette_batch=True,", "")
        adapted = adapted.replace("silhouette_batch=False,", "")
    with open(crc_path, "w") as f:
        f.write(adapted)
    print(f"Written: {out_names[tag]}  ({len(adapted.splitlines())} lines)")

# step3d — verify validation embedding
step3d = f'''#!/usr/bin/env python3
"""STEP 3d — Verify External Validation Embedding (CRC AGE, scGPT)"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE    = pathlib.Path("{BASE_CRC}")
EMB_KEY = "X_scGPT"

candidates = [
    BASE / "{VAL_FILE}_scgpt.h5ad",
    BASE / "step2a_embeddings" / "{VAL_FILE}_scgpt.h5ad",
]

print("STEP 3d -- Verify CRC AGE scGPT validation embedding", flush=True)

OUT_FILE = None
for c in candidates:
    if c.exists():
        OUT_FILE = c
        break

if OUT_FILE is None:
    print("  ERROR: Validation embedding not found. Run step2a first.", flush=True)
    sys.exit(1)

ad = sc.read_h5ad(OUT_FILE)
if EMB_KEY not in ad.obsm:
    print(f"  ERROR: {{EMB_KEY}} missing", flush=True)
    sys.exit(1)

n_unique = len(np.unique(ad.obsm[EMB_KEY], axis=0))
print(f"  Cells: {{ad.n_obs:,}}  |  Unique embedding rows: {{n_unique:,}}", flush=True)
if n_unique <= 10:
    print("  ERROR: Degenerate embedding.", flush=True)
    sys.exit(1)

root_copy = BASE / "{VAL_FILE}_scgpt.h5ad"
if not root_copy.exists():
    shutil.copy(str(OUT_FILE), str(root_copy))
    print(f"  Copied to root: {{root_copy.name}}", flush=True)

print("  OK -- validation embedding valid.", flush=True)
print("STEP 3d COMPLETE", flush=True)
'''
with open(f"{BASE_CRC}/{out_names['step3d']}", "w") as f:
    f.write(step3d)
print(f"Written: {out_names['step3d']}")

# slurm files
ENV    = "scgpt310"
MODULE = "miniforge3/25.3.0-3"

slurm_specs = [
    ("step4",  out_names["step4"],  "batch", "32G", "4:00:00"),
    ("step4a", out_names["step4a"], "batch", "64G", "6:00:00"),
    ("step4b", out_names["step4b"], "batch", "64G", "6:00:00"),
    ("step5",  out_names["step5"],  "batch", "64G", "6:00:00"),
    ("step6",  out_names["step6"],  "batch", "64G", "6:00:00"),
    ("step7",  out_names["step7"],  "batch", "64G", "4:00:00"),
    ("step8",  out_names["step8"],  "batch", "32G", "2:00:00"),
    ("step9",  out_names["step9"],  "batch", "64G", "4:00:00"),
]

for tag, script, partition, mem, tlimit in slurm_specs:
    slurm = f"""#!/bin/bash
#SBATCH --job-name=CRC_age_scgpt_{tag}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={tlimit}
#SBATCH --output={BASE_CRC}/logs/CRC_age_scgpt_{tag}_%j.out
#SBATCH --error={BASE_CRC}/logs/CRC_age_scgpt_{tag}_%j.err

module purge
module load {MODULE}
source activate {ENV}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export LD_PRELOAD=/data/.conda/envs/scgpt310/lib/libstdc++.so.6
cd {BASE_CRC}
python {script}
echo "Exit: $?"
"""
    with open(f"{BASE_CRC}/{tag}_CRC_age_scgpt.slurm", "w") as f:
        f.write(slurm)
    print(f"Written: {tag}_CRC_age_scgpt.slurm")

# submit script
submit = f"""#!/bin/bash
# Submit CRC age scGPT steps 4-9 with dependencies.
# Steps 3a, 3b, 3d must be run manually first.
# Usage: bash submit_CRC_age_scGPT_steps.sh

BASE={BASE_CRC}
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {{
  local name=$1; local deps=$2
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${{deps}}"
  local id=$(sbatch $dep_flag $BASE/${{name}}_CRC_age_scgpt.slurm | awk '{{print $NF}}')
  echo "$name: $id" >&2
  echo $id
}}

J4=$(submit_job  "step4"  "")
J4A=$(submit_job "step4a" "")
J4B=$(submit_job "step4b" "")
J5=$(submit_job  "step5"  "")
J6=$(submit_job  "step6"  "")
J7=$(submit_job  "step7"  "")
J8=$(submit_job  "step8"  "")
J9=$(submit_job  "step9"  "${{J4}}:${{J4A}}:${{J4B}}:${{J5}}:${{J6}}:${{J7}}:${{J8}}")

echo ""
echo "Job chain:"
echo "  step4  : $J4    step4a : $J4A   step4b : $J4B"
echo "  step5  : $J5    step6  : $J6    step7  : $J7    step8 : $J8"
echo "  step9  : $J9  (after all)"
echo "Monitor: squeue -u $USER"
"""
with open(f"{BASE_CRC}/submit_CRC_age_scGPT_steps.sh", "w") as f:
    f.write(submit)
print(f"\nWritten: submit_CRC_age_scGPT_steps.sh")

# spot check
print("\nSpot-check step3b and step8:")
for tag in ["step3b", "step8"]:
    path = f"{BASE_CRC}/{out_names[tag]}"
    result = subprocess.run(
        ["grep", "-n", r"OUTPUT_BASE\|DATASETS\|UNDERREP\|BASE =\|Proportional\|520\|1999\|99Each\|9402\|KNOWN_GROUP\|EMBDIR\|VALIDATION"],
        stdin=open(path), capture_output=True, text=True)
    print(f"\n--- {out_names[tag]} ---")
    print(result.stdout[:1000])

print("\nAll done. Run:")
print(f"  bash {BASE_CRC}/submit_CRC_age_scGPT_steps.sh")
