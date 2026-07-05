#!/bin/bash
#SBATCH --job-name=scf_age_7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_7_%j.out
#SBATCH --error=/data/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_7_%j.err

BASE=/data/scfoundation/augmentedv4/age_scfoundation_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step7_representation_diagnostics_scfoundation_age.py
echo "Exit: $?"
