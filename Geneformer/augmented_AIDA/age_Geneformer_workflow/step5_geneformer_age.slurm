#!/bin/bash
#SBATCH --job-name=gf_age_5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_5_%j.out
#SBATCH --error=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_5_%j.err

BASE=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow
module load miniforge3/25.3.0-3
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd $BASE && python step5_fairness_geneformer_age.py
echo "Job finished at: $(date) | Exit: $?"
