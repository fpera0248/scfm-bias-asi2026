#!/bin/bash
#SBATCH --job-name=age_s0b_stage2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/data/scfoundation/augmentedv4/age_scfoundation_workflow/logs/age_stage2_%x_%j.out
#SBATCH --error=/data/scfoundation/augmentedv4/age_scfoundation_workflow/logs/age_stage2_%x_%j.err

set -euo pipefail

echo "================================================="
echo " STAGE 2: scDesign3 AUGMENTATION (AGE)"
echo " Job ID:        $SLURM_JOB_ID"
echo " Job Name:      $SLURM_JOB_NAME"
echo " Bin (env):     $BIN_LABEL_ENCODED"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " Start:         $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scdesign3_env

cd /data/scfoundation/augmentedv4/age_scfoundation_workflow
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript step0b_stage2_augment.R
EXIT=$?
echo "Stage 2 ($BIN_LABEL_ENCODED) finished at: $(date) | Exit: $EXIT"
exit $EXIT
