#!/bin/bash
#SBATCH --job-name=gf_age_6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_6_%j.out
#SBATCH --error=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_6_%j.err

set -eo pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

cd /data/Geneformer/augmented_AIDA/age_Geneformer_workflow
python step6_per_age_diagnostics_geneformer.py
echo "Exit: $?"
