#!/bin/bash
#SBATCH --job-name=scf_age_4a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_4a_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow/logs/scf_age_4a_%j.err

BASE=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
cd $BASE && python step4a_downstream_scfoundation_age.py
echo "Exit: $?"
