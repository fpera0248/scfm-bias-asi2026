#!/bin/bash
#SBATCH --job-name=gf_eth_step9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/gf_eth_step9_%j.out
#SBATCH --error=/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/gf_eth_step9_%j.err

set -euo pipefail

echo "================================================="
echo " Geneformer Step 9 -- Combine UMAPs (ETHNICITY)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Start time:  $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate geneformer310
set -u

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"
mkdir -p "$BASE_DIR/logs"
cd "$BASE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run both step9 scripts (UMAPs + learning curves)
python step9_visualizations_geneformer_ethnicity.py
EX1=$?


EXIT_CODE=$EX1
echo "Job finished at: $(date) | step9 exit: $EXIT_CODE"
exit $EXIT_CODE
