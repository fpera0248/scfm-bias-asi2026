#!/bin/bash
#SBATCH --job-name=eth_step7b_sf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/step7b_%j.out
#SBATCH --error=logs/step7b_%j.err

set -euo pipefail
cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow
mkdir -p logs

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

python step7b_external_val_probe_scfoundation.py
