#!/bin/bash
# Submit CRC ethnicity scGPT steps 3a-9 with dependencies.
# Usage: bash submit_CRC_eth_scGPT_steps.sh

BASE=/oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/ethnicity_scGPT_workflow
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {
  local name=$1; local deps=$2
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${deps}"
  local id=$(sbatch $dep_flag $BASE/${name}_CRC_eth_scgpt.slurm | awk '{print $NF}')
  echo "$name: $id" >&2
  echo $id
}

J3A=$(submit_job "step3a" "")
J3B=$(submit_job "step3b" "")
J3D=$(submit_job "step3d" "")
J4=$(submit_job  "step4"  "${J3B}:${J3D}")
J4A=$(submit_job "step4a" "${J3B}")
J4B=$(submit_job "step4b" "${J3B}")
J5=$(submit_job  "step5"  "${J3B}")
J6=$(submit_job  "step6"  "${J3B}")
J7=$(submit_job  "step7"  "${J3B}")
J8=$(submit_job  "step8"  "${J3B}")
J9=$(submit_job  "step9"  "${J3A}:${J4}:${J4A}:${J4B}:${J5}:${J6}:${J7}:${J8}")

echo ""
echo "Job chain:"
echo "  step3a : $J3A    step3b : $J3B    step3d : $J3D"
echo "  step4  : $J4  (after 3b+3d)"
echo "  step4a : $J4A   step4b : $J4B   step5 : $J5"
echo "  step6  : $J6    step7  : $J7    step8 : $J8"
echo "  step9  : $J9  (after all)"
echo "Monitor: squeue -u fperalta"
