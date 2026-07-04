#!/bin/bash
#SBATCH --job-name=scf_age_2a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_2a_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_2a_%j.err

BASE=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd $BASE && python step2a_embed_scfoundation_age.py
echo "Exit: $?"
