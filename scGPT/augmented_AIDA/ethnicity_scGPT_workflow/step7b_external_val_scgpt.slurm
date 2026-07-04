#!/bin/bash
#SBATCH --job-name=eth_step7b_scgpt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/ethnicity_scGPT_workflow/logs/eth_step7b_scgpt_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/ethnicity_scGPT_workflow/logs/eth_step7b_scgpt_%j.err

set -euo pipefail

echo "================================================="
echo " scGPT Step 7b -- External validation probe (ETHNICITY)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $(hostname)"
echo " Start time:  $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310

export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)
export PYTHONNOUSERSITE=1

cd /oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/ethnicity_scGPT_workflow

python step7b_external_val_probe_scgpt.py

echo ""
echo "Job finished at: $(date) | Exit: $?"
