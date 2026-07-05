#!/bin/bash
#SBATCH --job-name=CRC_2a_scgpt_eth
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow/logs/CRC_2a_scgpt_eth_%j.out
#SBATCH --error=/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow/logs/CRC_2a_scgpt_eth_%j.err

set -eo pipefail

module load miniforge3/25.3.0-3
source activate scgpt310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/data/.conda/envs/scgpt310/lib/libstdc++.so.6
cd /data/scGPT/augmented_CRC/ethnicity_scGPT_workflow
python step2a_embed_scgpt_CRC_eth.py
echo "Exit: $?"
