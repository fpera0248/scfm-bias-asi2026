#!/bin/bash
#SBATCH --job-name=sex_aug_v5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/scdesign3_sex_%j.out
#SBATCH --error=logs/scdesign3_sex_%j.err

set -euo pipefail

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented/sex_Geneformer_workflow"
cd "$BASE_DIR"
mkdir -p logs

if [ ! -f "InterstitialLungDisease_RawCounts_SEX.h5ad" ]; then
  echo "ERROR: step0a output not found."
  exit 1
fi

if [ ! -f "ILD_Sex_External_Validation_5000.h5ad" ]; then
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

echo "Running step0b sex augmentation v5..."
date
Rscript step0b_scdesign3_sex_augmentation.R
EXIT_CODE=$?

echo "Memory after run:"
free -h || true
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit "$EXIT_CODE"