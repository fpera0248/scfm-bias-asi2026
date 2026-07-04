#!/bin/bash
#SBATCH --job-name=gf_age_3a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_3a_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow/logs/gf_age_3a_%j.err

set -eo pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow
python step3a_benchmark_geneformer_age.py
echo "Exit: $?"
