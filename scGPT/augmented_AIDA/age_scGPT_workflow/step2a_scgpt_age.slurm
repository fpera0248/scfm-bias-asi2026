#!/bin/bash
#SBATCH --job-name=scgpt_age_2a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_2a_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_2a_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310

export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)
export PYTHONNOUSERSITE=1

cd /oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow
python step2a_embed_scgpt_age.py
echo "Exit: $?"
