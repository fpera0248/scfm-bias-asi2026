#!/bin/bash
#SBATCH --job-name=sex_step2a_sf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/data/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_step2a_sf_%j.out
#SBATCH --error=/data/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_step2a_sf_%j.err

set -e; set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu

export OMP_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0

cd /data/scfoundation/augmentedv4/sex_scfoundation_workflow
python step2a_embed_scfoundation_sex.py
echo "Job finished at: $(date) | Exit: $?"
