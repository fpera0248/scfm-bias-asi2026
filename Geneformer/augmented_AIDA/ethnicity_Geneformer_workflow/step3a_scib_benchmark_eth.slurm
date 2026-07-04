#!/bin/bash
#SBATCH --job-name=gf_eth_scib_step3a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=96:00:00
#SBATCH --output=/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/gf_eth_scib_step3a_%j.out
#SBATCH --error=/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/gf_eth_scib_step3a_%j.err

set -euo pipefail

echo "================================================="
echo " Geneformer Step 3a -- scIB Benchmark (ETHNICITY)"
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
set -u
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Environment check"
which python
python --version
echo "Conda env: $CONDA_PREFIX"

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow"
LOGDIR="$BASE_DIR/logs"

mkdir -p "$LOGDIR"
cd "$BASE_DIR" || { echo "Failed to cd into $BASE_DIR"; exit 1; }
pwd

python - << 'EOF'
import sys, scanpy as sc, scib_metrics
print("Python:      ", sys.executable)
print("scanpy:      ", sc.__version__)
print("scib_metrics:", scib_metrics.__version__)
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python - << 'EOF'
import scanpy as sc, numpy as np, sys, pathlib

BASE        = pathlib.Path("/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow")
OUTPUT_BASE = "AIDA_Ethnicity_Pilot"
EMB_KEY     = "X_geneformer"

files = [
    f"{OUTPUT_BASE}_BalancedAugmented_779Each_ETHNICITY_geneformer.h5ad",
    f"{OUTPUT_BASE}_Proportional_2500_ETHNICITY_geneformer.h5ad",
    f"{OUTPUT_BASE}_BalancedUpsampled_779Each_ETHNICITY_geneformer.h5ad",
    f"{OUTPUT_BASE}_Downsampled_92Each_ETHNICITY_geneformer.h5ad",
]

all_ok = True
for f in files:
    path = BASE / f
    if not path.exists():
        print(f"MISSING: {f}"); all_ok = False; continue
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm:
        print(f"NO EMBEDDING: {f}"); all_ok = False; continue
    n_unique = len(np.unique(np.array(ad.obsm[EMB_KEY]), axis=0))
    print(f"OK  {ad.n_obs:>6,} cells  {n_unique:>6,} unique emb rows  {f}")

if not all_ok:
    print("\nFATAL: one or more files missing. Run step2a first.")
    sys.exit(1)
print("\nAll files verified.")
EOF

echo ""
echo "RUNNING STEP 3a -- scIB BENCHMARKING (ETHNICITY, Geneformer)"
date

python step3a_benchmark_geneformer_ethnicity.py

EXIT_CODE=$?

echo "-------------------------------------------------"
echo " Job finished at: $(date)"
echo " Exit code:      $EXIT_CODE"
echo "-------------------------------------------------"

exit $EXIT_CODE