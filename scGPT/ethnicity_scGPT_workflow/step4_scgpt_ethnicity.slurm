#!/bin/bash
#SBATCH --job-name=eth_step4_scgpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/ethnicity_scGPT_workflow/logs/eth_step4_scgpt_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/ethnicity_scGPT_workflow/logs/eth_step4_scgpt_%j.err

set -e
set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)

export PYTHONNOUSERSITE=1

cd /oscar/home/fperalta/data/fperalta/scGPT/ethnicity_scGPT_workflow
python step4_external_validation_scgpt_ethnicity.py
echo "Job finished at: $(date) | Exit: $?"
