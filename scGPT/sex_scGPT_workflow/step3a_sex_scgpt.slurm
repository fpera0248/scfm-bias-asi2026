#!/bin/bash
#SBATCH --job-name=sex_step3a_scgpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/data/scGPT/sex_scGPT_workflow/logs/sex_step3a_scgpt_%j.out
#SBATCH --error=/data/scGPT/sex_scGPT_workflow/logs/sex_step3a_scgpt_%j.err

set -e; set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310

export OMP_NUM_THREADS=$SLURM_NTASKS
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

cd /data/scGPT/sex_scGPT_workflow
python step3a_benchmark_scgpt_sex.py
echo "Job finished at: $(date) | Exit: $?"
