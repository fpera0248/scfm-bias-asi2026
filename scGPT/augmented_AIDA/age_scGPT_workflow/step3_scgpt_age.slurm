#!/bin/bash
#SBATCH --job-name=scgpt_age_3a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_3_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_3_%j.err

set -euo pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310

export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

cd /oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow
python step3a_benchmark_scgpt_age.py
echo "Exit: $?"
