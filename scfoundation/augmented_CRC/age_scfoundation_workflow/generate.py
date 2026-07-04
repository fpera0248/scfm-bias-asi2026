#!/usr/bin/env python3
"""
Generates steps 3a-9 for CRC age scFoundation workflow.
Run on Oscar: python3 generate_CRC_age_scFoundation_steps3to9.py
"""
import os, subprocess

BASE_ILD = "/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow"
BASE_CRC = "/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow"
OUTPUT_BASE = "CRC_Age_Pilot"
VAL_FILE    = "CRC_Age_External_Validation_9402"

os.makedirs(BASE_CRC, exist_ok=True)
os.makedirs(f"{BASE_CRC}/logs", exist_ok=True)
os.makedirs(f"{BASE_CRC}/step3b_labeled", exist_ok=True)

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
        .replace(
            'VALIDATION_FILE = BASE / "step2a_embeddings" / "ILD_Age_External_Validation_10500_scfoundation.h5ad"',
            f'VALIDATION_FILE = BASE / "step2a_embeddings" / "{VAL_FILE}_scfoundation.h5ad"\n'
            f'if not VALIDATION_FILE.exists():\n'
            f'    VALIDATION_FILE = BASE / "{VAL_FILE}_scfoundation.h5ad"')
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
        # step9 DATASET_ORDER
        .replace(
            '    "Proportional_2495",\n    "BalancedAugmented_1262Each",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "Proportional_1999",\n    "BalancedAugmented_520Each",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        .replace(
            '    "BalancedAugmented_1262Each",\n    "Proportional_2495",\n    "BalancedUpsampled_1262Each",\n    "Downsampled_25Each",',
            '    "BalancedAugmented_520Each",\n    "Proportional_1999",\n    "BalancedUpsampled_520Each",\n    "Downsampled_99Each",')
        # step9 UMAP stems
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
        # step9 UMAP display
        .replace('"Proportional (2,495 cells — real only)"',
                 '"Proportional (1,999 cells — real only)"')
        .replace('"Balanced Augmented (1,262/bin — scDesign3)"',
                 '"Balanced Augmented (520/bin — scDesign3)"')
        .replace('"Balanced Upsampled (1,262/bin — real only)"',
                 '"Balanced Upsampled (520/bin — real only)"')
        .replace('"Downsampled (25/bin — real only)"',
                 '"Downsampled (99/bin — real only)"')
        # kNN mixing notes
        .replace("random expected kNN mixing is ~0.86",
                 "random expected kNN mixing is ~0.80")
        .replace("(6 out of 7 neighbours expected from other bins by chance).",
                 "(4 out of 5 neighbours expected from other bins by chance).")
        .replace("Values well below 0.86 indicate age-specific clustering.",
                 "Values well below 0.80 indicate age-specific clustering.")
        .replace("NOTE: With 7 age bins, random expected kNN mixing is ~0.86",
                 "NOTE: With 5 age bins, random expected kNN mixing is ~0.80")
        .replace("with 7 age bins, random expected ~0.857 (6/7 neighbours from other bins).",
                 "with 5 age bins, random expected ~0.800 (4/5 neighbours from other bins).")
        # JAX cpu env
        .replace('os.environ["JAX_PLATFORM_NAME"] = "cpu"\n\n', '')
        # silhouette_batch compat
        .replace("silhouette_batch=True,", "")
        .replace("silhouette_batch=False,", "")
        # titles
        .replace("(AGE, scFoundation)", "(CRC AGE, scFoundation)")
        .replace("(AGE, Geneformer)", "(CRC AGE, scFoundation)")
        .replace("scFoundation AGE", "CRC scFoundation AGE")
    )

script_names = {
    "step3a": "step3a_benchmark_scfoundation_age.py",
    "step3b": "step3b_label_propagation_scfoundation_age.py",
    "step4":  "step4_external_validation_scfoundation_age.py",
    "step4a": "step4a_downstream_scfoundation_age.py",
    "step4b": "step4b_robustness_scfoundation_age.py",
    "step5":  "step5_fairness_scfoundation_age.py",
    "step6":  "step6_per_age_diagnostics_scfoundation.py",
    "step7":  "step7_representation_diagnostics_scfoundation_age.py",
    "step8":  "step8_age_conditioned_disease_scfoundation.py",
    "step9":  "step9_visualizations_scfoundation_age.py",
}

out_names = {
    "step3a": "step3a_benchmark_scfoundation_CRC_age.py",
    "step3b": "step3b_label_propagation_scfoundation_CRC_age.py",
    "step3d": "step3d_verify_validation_scfoundation_CRC_age.py",
    "step4":  "step4_external_validation_scfoundation_CRC_age.py",
    "step4a": "step4a_downstream_scfoundation_CRC_age.py",
    "step4b": "step4b_robustness_scfoundation_CRC_age.py",
    "step5":  "step5_fairness_scfoundation_CRC_age.py",
    "step6":  "step6_per_age_diagnostics_scfoundation_CRC.py",
    "step7":  "step7_representation_diagnostics_scfoundation_CRC_age.py",
    "step8":  "step8_age_conditioned_disease_scfoundation_CRC.py",
    "step9":  "step9_visualizations_scfoundation_CRC_age.py",
}

for tag, ild_fname in script_names.items():
    ild_path = f"{BASE_ILD}/{ild_fname}"
    crc_path = f"{BASE_CRC}/{out_names[tag]}"
    with open(ild_path) as f:
        src = f.read()
    adapted = adapt(src)
    with open(crc_path, "w") as f:
        f.write(adapted)
    print(f"Written: {out_names[tag]}  ({len(adapted.splitlines())} lines)")

# step3d
step3d = f'''#!/usr/bin/env python3
"""STEP 3d — Verify External Validation Embedding (CRC AGE, scFoundation)"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE    = pathlib.Path("{BASE_CRC}")
EMB_KEY = "X_scfoundation"

candidates = [
    BASE / "step2a_embeddings" / "{VAL_FILE}_scfoundation.h5ad",
    BASE / "{VAL_FILE}_scfoundation.h5ad",
]

print("STEP 3d -- Verify CRC AGE scFoundation validation embedding", flush=True)

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

# ensure copy in both locations
for dest in [BASE / "step2a_embeddings" / "{VAL_FILE}_scfoundation.h5ad",
             BASE / "{VAL_FILE}_scfoundation.h5ad"]:
    if not dest.exists():
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(str(OUT_FILE), str(dest))
        print(f"  Copied to: {{dest.name}}", flush=True)

print("  OK -- validation embedding valid.", flush=True)
print("STEP 3d COMPLETE", flush=True)
'''
with open(f"{BASE_CRC}/{out_names['step3d']}", "w") as f:
    f.write(step3d)
print(f"Written: {out_names['step3d']}")

# slurm files
ENV    = "scfoundation_gpu"
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
#SBATCH --job-name=CRC_age_scf_{tag}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={tlimit}
#SBATCH --output={BASE_CRC}/logs/CRC_age_scf_{tag}_%j.out
#SBATCH --error={BASE_CRC}/logs/CRC_age_scf_{tag}_%j.err

module purge
module load {MODULE}
source activate {ENV}
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd {BASE_CRC}
python {script}
echo "Exit: $?"
"""
    with open(f"{BASE_CRC}/{tag}_CRC_age_scf.slurm", "w") as f:
        f.write(slurm)
    print(f"Written: {tag}_CRC_age_scf.slurm")

# submit script
submit = f"""#!/bin/bash
# Submit CRC age scFoundation steps 4-9.
# Steps 3a, 3b, 3d must be run manually first.
# Usage: bash submit_CRC_age_scFoundation_steps.sh

BASE={BASE_CRC}
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {{
  local name=$1; local deps=$2
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${{deps}}"
  local id=$(sbatch $dep_flag $BASE/${{name}}_CRC_age_scf.slurm | awk '{{print $NF}}')
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
echo "Monitor: squeue -u fperalta"
"""
with open(f"{BASE_CRC}/submit_CRC_age_scFoundation_steps.sh", "w") as f:
    f.write(submit)
print(f"\nWritten: submit_CRC_age_scFoundation_steps.sh")

# spot check
print("\nSpot-check step3b and step8:")
for tag in ["step3b", "step8"]:
    path = f"{BASE_CRC}/{out_names[tag]}"
    result = subprocess.run(
        ["grep", "-n", r"OUTPUT_BASE\|DATASETS\|UNDERREP\|BASE =\|Proportional\|520\|1999\|99Each\|9402\|KNOWN_GROUP\|EMBDIR\|VALIDATION"],
        stdin=open(path), capture_output=True, text=True)
    print(f"\n--- {out_names[tag]} ---")
    print(result.stdout[:800])

print("\nAll done. Run:")
print(f"  python3 {BASE_CRC}/{out_names['step3a']}  # manually with JAX_PLATFORMS=cpu")
print(f"  python3 {BASE_CRC}/{out_names['step3b']}  # manually")
print(f"  python3 {BASE_CRC}/{out_names['step3d']}  # manually")
print(f"  bash {BASE_CRC}/submit_CRC_age_scFoundation_steps.sh")
