#!/bin/bash
#SBATCH --job-name=eth_step3b_scgpt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scGPT/augmented_AIDA/ethnicity_scGPT_workflow/logs/eth_step3b_scgpt_%j.out
#SBATCH --error=/data/scGPT/augmented_AIDA/ethnicity_scGPT_workflow/logs/eth_step3b_scgpt_%j.err

set -e
set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export LD_PRELOAD=$(find $CONDA_PREFIX/lib -name 'libstdc++.so.6*' -not -type l | head -1)

export OMP_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1

cd /data/scGPT/augmented_AIDA/ethnicity_scGPT_workflow
python step3b_label_propagation_scgpt_ethnicity.py
echo "Job finished at: $(date) | Exit: $?"
