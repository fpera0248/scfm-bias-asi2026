#!/bin/bash
#SBATCH --job-name=sex_step2a_scgpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/sex_scGPT_workflow/logs/sex_step2a_scgpt_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/sex_scGPT_workflow/logs/sex_step2a_scgpt_%j.err

set -e; set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0

cd /oscar/home/fperalta/data/fperalta/scGPT/sex_scGPT_workflow
python step2a_embed_scgpt_sex.py
echo "Job finished at: $(date) | Exit: $?"
