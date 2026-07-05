#!/bin/bash
#SBATCH --job-name=step2a_eth_scf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step2a_%j.out
#SBATCH --error=logs/step2a_%j.err

set -euo pipefail
echo "=== step2a START $(date -Iseconds) node=$(hostname) job=${SLURM_JOB_ID} ==="

cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

python3 step2a_embed_scfoundation_ethnicity.py

echo "=== step2a END $(date -Iseconds) ==="
