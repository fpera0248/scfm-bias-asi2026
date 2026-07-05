#!/bin/bash
#SBATCH --job-name=sex_s0b_stage3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_stage3_%j.out
#SBATCH --error=/data/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_stage3_%j.err

set -euo pipefail

echo "================================================="
echo " STAGE 3: AGGREGATE + WRITE FAIRNESS (SEX)"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $(hostname)"
echo " Start:      $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate scdesign3_env
set -u

cd /data/scfoundation/augmentedv4/sex_scfoundation_workflow
mkdir -p logs

Rscript step0b_stage3_aggregate.R
EXIT=$?
echo "Stage 3 finished at: $(date) | Exit: $EXIT"
exit $EXIT
