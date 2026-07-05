#!/bin/bash
#SBATCH --job-name=scgpt_age_2a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_2a_%j.out
#SBATCH --error=/data/scGPT/age_scGPT_workflow/logs/scgpt_age_2a_%j.err

BASE=/data/scGPT/age_scGPT_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/data/.conda/envs/scgpt310/lib/libstdc++.so.6
cd $BASE && python step2a_embed_scgpt_age.py
echo "Exit: $?"
