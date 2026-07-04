#!/bin/bash
#SBATCH --job-name=eth_step7b_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/eth_step7b_gf_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow/logs/eth_step7b_gf_%j.err

set -euo pipefail

echo "================================================="
echo " Geneformer Step 7b -- External validation probe (ETHNICITY)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $(hostname)"
echo " Start time:  $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/ethnicity_Geneformer_workflow

python step7b_external_val_probe_geneformer.py

echo ""
echo "Job finished at: $(date) | Exit: $?"
