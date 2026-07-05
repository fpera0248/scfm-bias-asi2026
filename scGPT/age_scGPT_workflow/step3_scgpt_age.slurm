#!/bin/bash
#SBATCH --job-name=scgpt_age_3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_3_%j.out
#SBATCH --error=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_3_%j.err

BASE=/data/scGPT/age_scGPT_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
cd $BASE && python step3a_benchmark_scgpt_age.py && python step3b_label_propagation_scgpt_age.py
echo "Exit: $?"
