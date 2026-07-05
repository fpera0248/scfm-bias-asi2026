#!/bin/bash
#SBATCH --job-name=eth_step4a_gf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/data/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step4a_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step4a_gf_%j.err

set -e
set -o pipefail

echo "================================================="
echo " STEP 4a — Downstream Prediction (ETHNICITY, Geneformer)"
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

BASE_DIR="/data/Geneformer/augmented/ethnicity_Geneformer_workflow"
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/step4a_downstream_ethnicity"
cd "$BASE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""

python step4a_downstream_results_eth_AR_EOS_geneformer.py

echo "Job finished at: $(date) | Exit: $?"