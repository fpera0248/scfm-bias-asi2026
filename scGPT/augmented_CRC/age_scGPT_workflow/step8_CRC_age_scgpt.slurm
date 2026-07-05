#!/bin/bash
#SBATCH --job-name=CRC_age_scgpt_step8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/data/scGPT/augmented_CRC/age_scGPT_workflow/logs/CRC_age_scgpt_step8_%j.out
#SBATCH --error=/data/scGPT/augmented_CRC/age_scGPT_workflow/logs/CRC_age_scgpt_step8_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scgpt310
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export LD_PRELOAD=/data/.conda/envs/scgpt310/lib/libstdc++.so.6
cd /data/scGPT/augmented_CRC/age_scGPT_workflow
python step8_age_conditioned_disease_scgpt_CRC.py
echo "Exit: $?"
