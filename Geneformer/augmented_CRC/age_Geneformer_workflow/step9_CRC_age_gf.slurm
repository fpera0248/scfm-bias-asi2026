#!/bin/bash
#SBATCH --job-name=CRC_age_step9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_age_step9_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/CRC_age_step9_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd /data/Geneformer/augmented_CRC/age_Geneformer_workflow
python step9_visualizations_geneformer_CRC_age.py
echo "Exit: $?"
