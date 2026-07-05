#!/bin/bash
#SBATCH --job-name=eth_step4a_sf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step4a_ethnicity_scfoundation_%j.out
#SBATCH --error=logs/step4a_ethnicity_scfoundation_%j.err

set -e; set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu

export OMP_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow
python step4a_downstream_scfoundation_ethnicity.py
echo "Job finished at: $(date) | Exit: $?"
