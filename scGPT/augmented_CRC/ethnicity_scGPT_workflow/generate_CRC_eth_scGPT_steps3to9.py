#!/usr/bin/env python3
"""
Generates steps 3a-9 for CRC ethnicity scGPT workflow.
Run on Oscar: python3 generate_CRC_eth_scGPT_steps3to9.py
"""
import os, subprocess

BASE_ILD = "/oscar/home/fperalta/data/fperalta/scGPT/ethnicity_scGPT_workflow"
BASE_CRC = "/oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/ethnicity_scGPT_workflow"
OUTPUT_BASE = "CRC_Eth_Pilot"
VAL_FILE    = "CRC_Eth_External_Validation_8572"

os.makedirs(BASE_CRC, exist_ok=True)
os.makedirs(f"{BASE_CRC}/logs", exist_ok=True)
os.makedirs(f"{BASE_CRC}/step3b_labeled", exist_ok=True)

def adapt(src: str) -> str:
    return (src
        # paths
        .replace(BASE_ILD, BASE_CRC)
        # INDIR: ILD uses step2a_embeddings; CRC files are in root
        .replace('INDIR  = BASE / "step2a_embeddings"', 'INDIR  = BASE')
        .replace('INDIR = BASE / "step2a_embeddings"',  'INDIR = BASE')
        .replace('EMB_DIR    = BASE / "step2a_embeddings"', 'EMB_DIR    = BASE')
        # validation file
        .replace(
            'VALIDATION_FILE = BASE / "step2a_embeddings" / "ILD_Ethnicity_External_Validation_12500_scgpt.h5ad"',
            f'VALIDATION_FILE = BASE / "{VAL_FILE}_scgpt.h5ad"')
        # output base
        .replace("ILD_Ethnicity_Pilot", OUTPUT_BASE)
        # dataset size keys
        .replace("BalancedAugmented_2143Each", "BalancedAugmented_1504Each")
        .replace("BalancedUpsampled_2143Each", "BalancedUpsampled_1504Each")
        .replace("Proportional_2497",          "Proportional_1998")
        .replace("Downsampled_48Each",          "Downsampled_90Each")
        # underrep group: ILD = native american, CRC = african american
        .replace('UNDERREP_GROUP = "native american"', 'UNDERREP_GROUP = "african american"')
        # KNOWN_GROUPS in step4b
        .replace(
            'KNOWN_GROUPS = ["asian", "european american", "hispanic or latin", "native american"]',
            'KNOWN_GROUPS = ["asian", "european american", "hispanic or latin", "african american"]')
        # DISEASE_GROUPS: CRC has all 4 groups (ILD only had 2)
        .replace(
            'DISEASE_GROUPS = {"african american", "european american"}',
            'DISEASE_GROUPS = {"african american", "asian", "european american", "hispanic or latin"}')
        # step9 short labels
        .replace('"Proportional_2497":          "Proportional\\n(2,497 cells)"',
                 '"Proportional_1998":          "Proportional\\n(1,998 cells)"')
        .replace('"BalancedAugmented_2143Each": "scDesign3\\nAugmented\\n(2,143/group)"',
                 '"BalancedAugmented_1504Each": "scDesign3\\nAugmented\\n(1,504/group)"')
        .replace('"BalancedUpsampled_2143Each": "Upsampled\\n(2,143/group)"',
                 '"BalancedUpsampled_1504Each": "Upsampled\\n(1,504/group)"')
        .replace('"Downsampled_48Each":         "Downsampled\\n(48/group)"',
                 '"Downsampled_90Each":         "Downsampled\\n(90/group)"')
        # step9 palette
        .replace('"Proportional_2497":          "#4C72B0"',
                 '"Proportional_1998":          "#4C72B0"')
        .replace('"BalancedAugmented_2143Each": "#DD8452"',
                 '"BalancedAugmented_1504Each": "#DD8452"')
        .replace('"BalancedUpsampled_2143Each": "#55A868"',
                 '"BalancedUpsampled_1504Each": "#55A868"')
        .replace('"Downsampled_48Each":         "#C44E52"',
                 '"Downsampled_90Each":         "#C44E52"')
        # step9 DATASET_ORDER
        .replace(
            '    "Proportional_2497", "BalancedAugmented_2143Each",\n    "BalancedUpsampled_2143Each", "Downsampled_48Each",',
            '    "Proportional_1998", "BalancedAugmented_1504Each",\n    "BalancedUpsampled_1504Each", "Downsampled_90Each",')
        .replace(
            '    "Proportional_2497",\n    "BalancedAugmented_2143Each",\n    "BalancedUpsampled_2143Each",\n    "Downsampled_48Each",',
            '    "Proportional_1998",\n    "BalancedAugmented_1504Each",\n    "BalancedUpsampled_1504Each",\n    "Downsampled_90Each",')
        # step9 UMAP_DATASETS stems
        .replace(
            '    "Proportional_2497":          "ILD_Ethnicity_Pilot_Proportional_2497_scgpt"',
            '    "Proportional_1998":          "CRC_Eth_Pilot_Proportional_1998_ETH_scgpt"')
        .replace(
            '    "BalancedAugmented_2143Each": "ILD_Ethnicity_Pilot_BalancedAugmented_2143Each_scgpt"',
            '    "BalancedAugmented_1504Each": "CRC_Eth_Pilot_BalancedAugmented_1504Each_ETH_scgpt"')
        .replace(
            '    "BalancedUpsampled_2143Each": "ILD_Ethnicity_Pilot_BalancedUpsampled_2143Each_scgpt"',
            '    "BalancedUpsampled_1504Each": "CRC_Eth_Pilot_BalancedUpsampled_1504Each_ETH_scgpt"')
        .replace(
            '    "Downsampled_48Each":         "ILD_Ethnicity_Pilot_Downsampled_48Each_scgpt"',
            '    "Downsampled_90Each":         "CRC_Eth_Pilot_Downsampled_90Each_ETH_scgpt"')
        # step9 UMAP_DISPLAY
        .replace('"Proportional_2497":          "Proportional (2,497 cells — real only)"',
                 '"Proportional_1998":          "Proportional (1,998 cells — real only)"')
        .replace('"BalancedAugmented_2143Each": "Balanced Augmented (2,143/group — scDesign3)"',
                 '"BalancedAugmented_1504Each": "Balanced Augmented (1,504/group — scDesign3)"')
        .replace('"BalancedUpsampled_2143Each": "Balanced Upsampled (2,143/group — real only)"',
                 '"BalancedUpsampled_1504Each": "Balanced Upsampled (1,504/group — real only)"')
        .replace('"Downsampled_48Each":         "Downsampled (48/group — real only)"',
                 '"Downsampled_90Each":         "Downsampled (90/group — real only)"')
        # step8 underrep text
        .replace(
            '"Fig 7C — Native American Accuracy Delta vs Proportional (scGPT)"',
            '"Fig 7C — African American Accuracy Delta vs Proportional (CRC scGPT)"')
        .replace(
            '"step8_fig7_native_american_delta.png"',
            '"step8_fig7_african_american_delta.png"')
        .replace(
            '"Native Am. Acc"', '"African Am. Acc"')
        .replace(
            '"Native Am. Delta"', '"African Am. Delta"')
        .replace(
            'na   = df8b[df8b["ethnicity"].str.lower().str.strip() == UNDERREP_GROUP]',
            'na   = df8b[df8b["ethnicity"].str.lower().str.strip() == UNDERREP_GROUP]')
        # titles
        .replace("(ETHNICITY, scGPT)", "(CRC ETHNICITY, scGPT)")
        .replace("scGPT ETHNICITY", "CRC scGPT ETHNICITY")
        .replace("(scGPT)", "(CRC scGPT)")
    )

script_names = {
    "step3a": "step3a_benchmark_scgpt_ethnicity.py",
    "step3b": "step3b_label_propagation_scgpt_ethnicity.py",
    "step4":  "step4_external_validation_scgpt_ethnicity.py",
    "step4a": "step4a_downstream_scgpt_ethnicity.py",
    "step4b": "step4b_robustness_scgpt_ethnicity.py",
    "step5":  "step5_fairness_scgpt_ethnicity.py",
    "step6":  "step6_per_ethnicity_diagnostics_scgpt.py",
    "step7":  "step7_representation_diagnostics_scgpt_ethnicity.py",
    "step8":  "step8_eth_conditioned_disease_scgpt.py",
    "step9":  "step9_visualizations_scgpt_ethnicity.py",
}

out_names = {
    "step3a": "step3a_benchmark_scgpt_CRC_eth.py",
    "step3b": "step3b_label_propagation_scgpt_CRC_eth.py",
    "step3d": "step3d_verify_validation_scgpt_CRC_eth.py",
    "step4":  "step4_external_validation_scgpt_CRC_eth.py",
    "step4a": "step4a_downstream_scgpt_CRC_eth.py",
    "step4b": "step4b_robustness_scgpt_CRC_eth.py",
    "step5":  "step5_fairness_scgpt_CRC_eth.py",
    "step6":  "step6_per_ethnicity_diagnostics_scgpt_CRC.py",
    "step7":  "step7_representation_diagnostics_scgpt_CRC_eth.py",
    "step8":  "step8_eth_conditioned_disease_scgpt_CRC.py",
    "step9":  "step9_visualizations_scgpt_CRC_eth.py",
}

for tag, ild_fname in script_names.items():
    ild_path = f"{BASE_ILD}/{ild_fname}"
    crc_path = f"{BASE_CRC}/{out_names[tag]}"
    with open(ild_path) as f:
        src = f.read()
    adapted = adapt(src)
    if tag == "step3a":
        adapted = adapted.replace("silhouette_batch=True,", "")
        adapted = adapted.replace("silhouette_batch=False,", "")
    with open(crc_path, "w") as f:
        f.write(adapted)
    print(f"Written: {out_names[tag]}  ({len(adapted.splitlines())} lines)")

# step3d
step3d = f'''#!/usr/bin/env python3
"""STEP 3d — Verify External Validation Embedding (CRC ETHNICITY, scGPT)"""
import sys, pathlib, shutil
import numpy as np
import scanpy as sc

BASE    = pathlib.Path("{BASE_CRC}")
EMB_KEY = "X_scGPT"

candidates = [
    BASE / "{VAL_FILE}_scgpt.h5ad",
    BASE / "step2a_embeddings" / "{VAL_FILE}_scgpt.h5ad",
]

print("STEP 3d -- Verify CRC ETH scGPT validation embedding", flush=True)

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
#SBATCH --job-name=CRC_eth_scgpt_{tag}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={tlimit}
#SBATCH --output={BASE_CRC}/logs/CRC_eth_scgpt_{tag}_%j.out
#SBATCH --error={BASE_CRC}/logs/CRC_eth_scgpt_{tag}_%j.err

module purge
module load {MODULE}
source activate {ENV}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt310/lib/libstdc++.so.6
cd {BASE_CRC}
python {script}
echo "Exit: $?"
"""
    with open(f"{BASE_CRC}/{tag}_CRC_eth_scgpt.slurm", "w") as f:
        f.write(slurm)
    print(f"Written: {tag}_CRC_eth_scgpt.slurm")

# submit script
submit = f"""#!/bin/bash
# Submit CRC ethnicity scGPT steps 4-9 with dependencies.
# Steps 3a, 3b, 3d must be run manually first.
# Usage: bash submit_CRC_eth_scGPT_steps.sh

BASE={BASE_CRC}
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {{
  local name=$1; local deps=$2
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${{deps}}"
  local id=$(sbatch $dep_flag $BASE/${{name}}_CRC_eth_scgpt.slurm | awk '{{print $NF}}')
  echo "$name: $id" >&2
  echo $id
}}

# steps 4, 4a, 4b, 5, 6, 7, 8 run independently (3b already done)
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
with open(f"{BASE_CRC}/submit_CRC_eth_scGPT_steps.sh", "w") as f:
    f.write(submit)
print(f"\nWritten: submit_CRC_eth_scGPT_steps.sh")

# spot check
print("\nSpot-check step3b and step8:")
for tag in ["step3b", "step8"]:
    path = f"{BASE_CRC}/{out_names[tag]}"
    result = subprocess.run(
        ["grep", "-n", r"OUTPUT_BASE\|DATASETS\|UNDERREP\|BASE =\|Proportional\|1504\|1998\|90Each\|8572\|KNOWN_GROUP\|INDIR\|DISEASE_GROUPS\|VALIDATION"],
        stdin=open(path), capture_output=True, text=True)
    print(f"\n--- {out_names[tag]} ---")
    print(result.stdout[:1000])

print("\nAll done. Run:")
print(f"  bash {BASE_CRC}/submit_CRC_eth_scGPT_steps.sh")
