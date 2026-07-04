#!/bin/bash
#SBATCH --job-name=gf_eth_labelprop_step3b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/gf_eth_labelprop_step3b_%j.out
#SBATCH --error=/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/gf_eth_labelprop_step3b_%j.err

set -euo pipefail

echo "================================================="
echo " Geneformer Step 3b -- Label Propagation (ETHNICITY)"
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

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow"
mkdir -p "$BASE_DIR/logs"
cd "$BASE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python step3b_label_propagation_geneformer_ethnicity.py
EXIT_CODE=$?
echo "Job finished at: $(date) | Exit: $EXIT_CODE"
exit $EXIT_CODE
