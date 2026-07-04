#!/bin/bash
#SBATCH --job-name=gf_embed_age
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_embed_age_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_embed_age_%j.err

BASE=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/geneformer310/lib/libstdc++.so.6
cd $BASE
python step2a_embed_geneformer_age.py
echo "Exit: $?"
