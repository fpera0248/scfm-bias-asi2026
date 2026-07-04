#!/bin/bash
#SBATCH --job-name=age_step7b_sf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/age_step7b_sf_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/age_step7b_sf_%j.err

set -euo pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow
python step7b_external_val_probe_scfoundation.py
echo "Job finished at: $(date) | Exit: $?"
