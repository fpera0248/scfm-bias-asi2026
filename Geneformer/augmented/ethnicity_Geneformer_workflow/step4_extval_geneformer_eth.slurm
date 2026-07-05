#!/bin/bash
#SBATCH --job-name=eth_step4_extval_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step4_extval_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step4_extval_gf_%j.err

set -euo pipefail

echo "================================================="
echo " STEP 4 -- External Validation (Geneformer ETHNICITY)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $(hostname)"
echo " CPUs:        $SLURM_CPUS_PER_TASK"
echo " Start time:  $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate geneformer310
set -u
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Environment check"
which python
python --version
echo "Conda env: $CONDA_PREFIX"

BASE_DIR="/data/Geneformer/augmented/ethnicity_Geneformer_workflow"
LOGDIR="/data/Geneformer/augmented/ethnicity_Geneformer_workflow/logs"

mkdir -p "$LOGDIR"
mkdir -p "$BASE_DIR/step4_external_validation_geneformer"

cd "$BASE_DIR" || { echo "Failed to cd into $BASE_DIR"; exit 1; }
pwd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify required input files exist
python - << 'PYEOF'
import sys, pathlib
BASE = pathlib.Path("/data/Geneformer/augmented/ethnicity_Geneformer_workflow")
LABELED = BASE / "step3b_labeled"
pilot_files = [
    "ILD_Ethnicity_Pilot_BalancedAugmented_2143Each_labeled_geneformer.h5ad",
    "ILD_Ethnicity_Pilot_Proportional_2497_labeled_geneformer.h5ad",
    "ILD_Ethnicity_Pilot_BalancedUpsampled_2143Each_labeled_geneformer.h5ad",
    "ILD_Ethnicity_Pilot_Downsampled_48Each_labeled_geneformer.h5ad",
]
val_file = BASE / "ILD_Ethnicity_External_Validation_12500_geneformer.h5ad"

ok = True
for f in pilot_files:
    p = LABELED / f
    status = "OK" if p.exists() else "MISSING"
    print(f"{status}: {f}")
    if not p.exists(): ok = False

vstatus = "OK" if val_file.exists() else "MISSING"
print(f"{vstatus}: {val_file.name}")
if not val_file.exists(): ok = False

if not ok:
    print("\nFATAL: required files missing.")
    sys.exit(1)
print("\nAll required files verified.")
PYEOF

echo ""
echo "[RUNNING STEP 4 -- External Validation (Geneformer)]"
date

python step4_external_validation_geneformer_ethnicity.py

EXIT_CODE=$?

echo "------------------------------------------"
echo " Job finished at: $(date)"
echo " Exit code: $EXIT_CODE"
echo "------------------------------------------"

exit $EXIT_CODE
