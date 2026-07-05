#!/bin/bash
#SBATCH --job-name=s0b_stage1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

set -euo pipefail
cd /data/scfoundation/augmentedv4/ethnicity_scfoundation_workflow
mkdir -p logs

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scdesign3_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export R_MAX_VSIZE=120Gb

echo "=== STAGE 1 START $(date -Iseconds) ==="
Rscript --vanilla step0b_stage1_setup.R
echo "=== STAGE 1 END $(date -Iseconds) ==="
