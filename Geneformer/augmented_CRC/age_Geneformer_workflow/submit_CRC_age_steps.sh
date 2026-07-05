#!/bin/bash
# Submit CRC age Geneformer steps 3a-9 with dependencies.
# Steps 4a-9 wait for step 3b to complete.
# Step 9 waits for all of 4a, 4b, 5, 6, 7, 8.
# Usage: bash submit_CRC_age_steps.sh

BASE=/data/Geneformer/augmented_CRC/age_Geneformer_workflow
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {
  local name=$1; local script=$2; local mem=$3; local time=$4; local deps=$5
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${deps}"
  local slurm=$BASE/${name}_CRC_age_gf.slurm
  local id=$(sbatch $dep_flag $slurm | awk '{print $NF}')
  echo "$name: $id" >&2
  echo $id
}

# step 3a (scIB) — no deps
J3A=$(submit_job "step3a" "step3a_benchmark_geneformer_CRC_age.py" "64G" "4:00:00" "")

# step 3b (label propagation) — no deps (parallel with 3a)
J3B=$(submit_job "step3b" "step3b_label_propagation_geneformer_CRC_age.py" "32G" "2:00:00" "")

# step 3d (verify validation) — no deps
J3D=$(submit_job "step3d" "step3d_verify_validation_CRC_age.py" "16G" "0:30:00" "")

# step 4 (external validation) — needs 3b + 3d
J4=$(submit_job "step4" "step4_external_validation_geneformer_CRC_age.py" "32G" "4:00:00" "${J3B}:${J3D}")

# steps 4a, 4b, 5, 6, 7, 8 — each needs step 3b
J4A=$(submit_job "step4a" "step4a_downstream_geneformer_CRC_age.py" "64G" "6:00:00" "${J3B}")
J4B=$(submit_job "step4b" "step4b_robustness_geneformer_CRC_age.py" "64G" "6:00:00" "${J3B}")
J5=$(submit_job  "step5"  "step5_fairness_geneformer_CRC_age.py"  "64G" "6:00:00" "${J3B}")
J6=$(submit_job  "step6"  "step6_per_age_diagnostics_geneformer_CRC.py"  "64G" "6:00:00" "${J3B}")
J7=$(submit_job  "step7"  "step7_representation_diagnostics_geneformer_CRC_age.py"  "64G" "4:00:00" "${J3B}")
J8=$(submit_job  "step8"  "step8_age_conditioned_disease_geneformer_CRC.py"  "32G" "2:00:00" "${J3B}")

# step 9 — needs all prior analysis steps
ALL="${J3A}:${J4}:${J4A}:${J4B}:${J5}:${J6}:${J7}:${J8}"
J9=$(submit_job "step9" "step9_visualizations_geneformer_CRC_age.py" "64G" "4:00:00" "${ALL}")

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
echo "Monitor: squeue -u $USER"
