#!/bin/bash
#SBATCH --job-name=scgpt_age_7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_7_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow/logs/scgpt_age_7_%j.err

BASE=/oscar/home/fperalta/data/fperalta/scGPT/augmented_AIDA/age_scGPT_workflow
module load miniforge3/25.3.0-3
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step7_representation_diagnostics_scgpt_age.py
echo "Exit: $?"
