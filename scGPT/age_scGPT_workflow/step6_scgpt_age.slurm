#!/bin/bash
#SBATCH --job-name=scgpt_age_6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_6_%j.out
#SBATCH --error=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_6_%j.err

BASE=/data/scGPT/age_scGPT_workflow
module load miniforge3/25.3.0-3
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step6_per_age_diagnostics_scgpt.py
echo "Exit: $?"
