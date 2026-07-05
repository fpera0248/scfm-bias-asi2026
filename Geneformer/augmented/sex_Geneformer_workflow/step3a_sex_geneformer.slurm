#!/bin/bash
#SBATCH --job-name=sex_step3a_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step3a_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step3a_gf_%j.err

set -euo pipefail

echo "================================================="
echo " STEP 3a -- scIB Benchmark (SEX, Geneformer)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $(hostname)"
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

cd /data/Geneformer/augmented/sex_Geneformer_workflow
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python step3a_benchmark_geneformer_sex.py
EXIT=$?
echo "Step 3a finished at: $(date) | Exit: $EXIT"
exit $EXIT
