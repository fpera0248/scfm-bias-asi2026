#!/bin/bash
# Submit CRC ethnicity scFoundation steps 4-9.
# Steps 3a, 3b, 3d must be run manually first.
# Usage: bash submit_CRC_eth_scFoundation_steps.sh

BASE=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow
LOGS=$BASE/logs
mkdir -p $LOGS

submit_job() {
  local name=$1; local deps=$2
  local dep_flag=""
  [[ -n "$deps" ]] && dep_flag="--dependency=afterok:${deps}"
  local id=$(sbatch $dep_flag $BASE/${name}_CRC_eth_scf.slurm | awk '{print $NF}')
  echo "$name: $id" >&2
  echo $id
}

J4=$(submit_job  "step4"  "")
J4A=$(submit_job "step4a" "")
J4B=$(submit_job "step4b" "")
J5=$(submit_job  "step5"  "")
J6=$(submit_job  "step6"  "")
J7=$(submit_job  "step7"  "")
J8=$(submit_job  "step8"  "")
J9=$(submit_job  "step9"  "${J4}:${J4A}:${J4B}:${J5}:${J6}:${J7}:${J8}")

echo ""
echo "Job chain:"
echo "  step4  : $J4    step4a : $J4A   step4b : $J4B"
echo "  step5  : $J5    step6  : $J6    step7  : $J7    step8 : $J8"
echo "  step9  : $J9  (after all)"
echo "Monitor: squeue -u fperalta"
