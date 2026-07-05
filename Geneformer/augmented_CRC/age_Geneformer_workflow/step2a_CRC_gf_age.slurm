#!/bin/bash
#SBATCH --job-name=CRC_2a_gf_age
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_2a_gf_age_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_2a_gf_age_%j.err

set -eo pipefail

module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1

cd /data/Geneformer/augmented_CRC/age_Geneformer_workflow
python step2a_embed_geneformer_CRC_age.py
echo "Exit: $?"
