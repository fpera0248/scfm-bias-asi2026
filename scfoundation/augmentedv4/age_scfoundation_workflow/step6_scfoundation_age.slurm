#!/bin/bash
#SBATCH --job-name=scf_age_6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_6_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow/logs/scf_age_6_%j.err

BASE=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/age_scfoundation_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step6_per_age_diagnostics_scfoundation.py
echo "Exit: $?"
