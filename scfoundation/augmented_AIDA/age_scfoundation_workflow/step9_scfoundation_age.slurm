#!/bin/bash
#SBATCH --job-name=scf_age_9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/data/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_9_%j.out
#SBATCH --error=/data/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_9_%j.err

BASE=/data/scfoundation/augmented_AIDA/age_scfoundation_workflow
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step9_visualizations_scfoundation_age.py
echo "Exit: $?"
