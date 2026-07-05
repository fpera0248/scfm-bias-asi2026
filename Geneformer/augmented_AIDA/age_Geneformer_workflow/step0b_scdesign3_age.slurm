#!/bin/bash
#SBATCH --job-name=scdesign3_age_aug_v5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/scdesign3_age_%j.out
#SBATCH --error=logs/scdesign3_age_%j.err

set -euo pipefail

echo "================================================="
echo " STEP 0b -- scDesign3 Age Augmentation v5"
echo " Job ID: ${SLURM_JOB_ID:-unknown}"
echo " Node: $(hostname)"
echo " CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo " Mem per node: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo " Mem per CPU:  ${SLURM_MEM_PER_CPU:-unknown} MB"
echo " Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo " Start time: $(date)"
echo "================================================="

# ---------------- modules ----------------
module purge
module load miniforge3/25.3.0-3

# ---------------- conda ----------------
source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
conda activate scdesign3_env

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Conda activation failed"
  exit 1
fi

echo "Environment check"
which R
R --version
echo "Conda env: $CONDA_PREFIX"

# ---------------- working directory ----------------
BASE_DIR="/data/Geneformer/augmented_AIDA/age_Geneformer_workflow"

if [ ! -d "$BASE_DIR" ]; then
  echo "ERROR: BASE_DIR does not exist: $BASE_DIR"
  exit 1
fi

cd "$BASE_DIR"
mkdir -p logs

echo "Working directory:"
pwd
ls -lh

# ---------------- sanity checks ----------------
if [ ! -f "step0b_scdesign3_age_augmentation.R" ]; then
  echo "ERROR: step0b_scdesign3_age_augmentation.R not found"
  exit 1
fi

if [ ! -f "AIDA_RawCounts_AGE.h5ad" ]; then
  echo "ERROR: AIDA_RawCounts_AGE.h5ad not found"
  exit 1
fi

if [ ! -f "AIDA_Age_External_Validation_10000.h5ad" ]; then
  echo "ERROR: AIDA_Age_External_Validation_10000.h5ad not found"
  exit 1
fi

# ---------------- thread control ----------------
THREADS=1
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
echo "Thread control set to $THREADS thread (serial)"

# ---------------- memory snapshot ----------------
echo "Memory before run:"
free -h || true

# ---------------- run R script ----------------
echo "Running scDesign3 age augmentation v5..."
date

Rscript step0b_scdesign3_age_augmentation.R

EXIT_CODE=$?

# ---------------- memory snapshot after ----------------
echo "Memory after run:"
free -h || true

echo "================================================="
echo " Job finished at: $(date)"
echo " Exit code: $EXIT_CODE"
echo "================================================="

exit "$EXIT_CODE"