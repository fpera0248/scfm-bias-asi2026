#!/bin/bash
#SBATCH --job-name=gf_age_8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_8_%j.out
#SBATCH --error=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_8_%j.err

BASE=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow
module load miniforge3/25.3.0-3
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd $BASE && python step8_age_conditioned_disease_geneformer.py
echo "Job finished at: $(date) | Exit: $?"
