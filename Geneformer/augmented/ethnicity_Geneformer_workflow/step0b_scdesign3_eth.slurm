#!/bin/bash
#SBATCH --job-name=eth_step0b_v5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/step0b_eth_%j.out
#SBATCH --error=logs/step0b_eth_%j.err

set -euo pipefail

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"
cd "$BASE_DIR"
mkdir -p logs

if [ ! -f "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad" ]; then
  echo "ERROR: step0a output not found."
  exit 1
fi

if [ ! -f "ILD_Ethnicity_External_Validation_12500.h5ad" ]; then
  echo "ERROR: step0c output not found. Run step0c before step0b."
  exit 1
fi

module purge
module load miniforge3/25.3.0-3
source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
conda activate scdesign3_env

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Conda activation failed"
  exit 1
fi

which R
R --version
echo "Conda env: $CONDA_PREFIX"

THREADS=1
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS

echo "Memory before run:"
free -h || true

echo "Running step0b ethnicity augmentation v5..."
date
Rscript step0b_scdesign3_ethnicity_augmentation.R
EXIT_CODE=$?

echo "Memory after run:"
free -h || true
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit "$EXIT_CODE"