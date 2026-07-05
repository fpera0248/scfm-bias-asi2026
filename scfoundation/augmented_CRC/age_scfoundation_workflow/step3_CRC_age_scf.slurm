#!/bin/bash
#SBATCH --job-name=scf_CRC_age_3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/scf_CRC_age_3_%j.out
#SBATCH --error=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/scf_CRC_age_3_%j.err

set -eo pipefail

BASE=/data/scfoundation/augmented_CRC/age_scfoundation_workflow

module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu

cd $BASE && python step3a_benchmark_scfoundation_CRC_age.py && python step3b_label_propagation_scfoundation_CRC_age.py
echo "Exit: $?"
