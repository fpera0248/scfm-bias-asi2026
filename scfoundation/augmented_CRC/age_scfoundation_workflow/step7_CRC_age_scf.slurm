#!/bin/bash
#SBATCH --job-name=CRC_age_scf_step7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step7_%j.out
#SBATCH --error=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step7_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd /data/scfoundation/augmented_CRC/age_scfoundation_workflow
python step7_representation_diagnostics_scfoundation_CRC_age.py
echo "Exit: $?"
