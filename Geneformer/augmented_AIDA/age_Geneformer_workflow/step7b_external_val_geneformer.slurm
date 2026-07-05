#!/bin/bash
#SBATCH --job-name=age_step7b_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/age_step7b_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/age_step7b_gf_%j.err

set -eo pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
cd /data/Geneformer/augmented_AIDA/age_Geneformer_workflow
python step7b_external_val_probe_geneformer.py
echo "Job finished at: $(date) | Exit: $?"
