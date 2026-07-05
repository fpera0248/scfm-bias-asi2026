#!/bin/bash
#SBATCH --job-name=CRC_age_step3d
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_age_step3d_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_age_step3d_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd /data/Geneformer/augmented_CRC/age_Geneformer_workflow
python step3d_verify_validation_CRC_age.py
echo "Exit: $?"
