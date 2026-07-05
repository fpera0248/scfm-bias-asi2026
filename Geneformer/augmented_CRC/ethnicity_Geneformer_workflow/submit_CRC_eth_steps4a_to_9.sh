#!/bin/bash
# Submit CRC ethnicity Geneformer steps 4a-9 with dependencies.
# Step 9 runs only after all prior steps succeed.
# Usage: bash submit_CRC_eth_steps4a_to_9.sh

BASE=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
LOGS=$BASE/logs
mkdir -p $LOGS

ENV=scfoundation_gpu
MODULE="miniforge3/25.3.0-3"

submit_step() {
  local name=$1
  local script=$2
  local partition=$3
  local mem=$4
  local time=$5
  local deps=$6

  local dep_flag=""
  if [[ -n "$deps" ]]; then
    dep_flag="--dependency=afterok:${deps}"
  fi

  local gpu_line=""
  if [[ "$partition" == "gpu" ]]; then
    gpu_line="#SBATCH --gres=gpu:1"
  fi

  local slurm_file=$BASE/${name}_CRC_eth_gf.slurm

  cat > $slurm_file << SEOF
#!/bin/bash
#SBATCH --job-name=CRC_eth_${name}
#SBATCH --partition=${partition}
#SBATCH --nodes=1
${gpu_line}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${LOGS}/CRC_eth_${name}_%j.out
#SBATCH --error=${LOGS}/CRC_eth_${name}_%j.err

module purge
module load ${MODULE}
source activate ${ENV}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd ${BASE}
python ${script}
echo "Exit: \$?"
SEOF

  local job_id
  job_id=$(sbatch $dep_flag $slurm_file | awk '{print $NF}')
  echo "$name submitted: job $job_id" >&2
  echo $job_id
}

JOB_4A=$(submit_step "step4a" \
  "step4a_downstream_results_eth_AR_EOS_geneformer.py" \
  "batch" "64G" "6:00:00" "")

JOB_4B=$(submit_step "step4b" \
  "step4b_overfitting_stress_tests_geneformer.py" \
  "batch" "64G" "6:00:00" "")

JOB_5=$(submit_step "step5" \
  "step5_fairness_eth_geneformer.py" \
  "batch" "64G" "6:00:00" "")

JOB_6=$(submit_step "step6" \
  "step6_per_ethnicity_diagnostics_geneformer.py" \
  "batch" "64G" "6:00:00" "")

JOB_7=$(submit_step "step7" \
  "step7_representation_diagnostics_geneformer.py" \
  "batch" "64G" "4:00:00" "")

JOB_8=$(submit_step "step8" \
  "step8_eth_conditioned_disease_geneformer.py" \
  "batch" "32G" "2:00:00" "")

ALL_DEPS="${JOB_4A}:${JOB_4B}:${JOB_5}:${JOB_6}:${JOB_7}:${JOB_8}"
JOB_9=$(submit_step "step9" \
  "step9_visualizations_geneformer_ethnicity.py" \
  "batch" "64G" "4:00:00" "$ALL_DEPS")

echo ""
echo "Job chain:"
echo "  step4a : $JOB_4A"
echo "  step4b : $JOB_4B"
echo "  step5  : $JOB_5"
echo "  step6  : $JOB_6"
echo "  step7  : $JOB_7"
echo "  step8  : $JOB_8"
echo "  step9  : $JOB_9  (runs after all above succeed)"
echo ""
echo "Monitor with: squeue -u $USER"
