#!/bin/bash
#SBATCH --job-name=s0b_stage2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2_%x_%j.out
#SBATCH --error=logs/stage2_%x_%j.err

set -euo pipefail
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/ethnicity_scfoundation_workflow
mkdir -p logs

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scdesign3_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export R_MAX_VSIZE=60Gb

GROUP="${BIN_LABEL_ENCODED:?BIN_LABEL_ENCODED env var required}"
GROUP="${GROUP//__SP__/ }"
echo "=== STAGE 2 START group=${GROUP} $(date -Iseconds) ==="
Rscript --vanilla step0b_stage2_augment.R "${GROUP}"
echo "=== STAGE 2 END group=${GROUP} $(date -Iseconds) ==="
