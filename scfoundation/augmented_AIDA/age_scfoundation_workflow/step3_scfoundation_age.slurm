#!/bin/bash
#SBATCH --job-name=scf_age_3a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_3_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_3_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu

cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow
python step3a_benchmark_scfoundation_age.py
echo "Exit: $?"
