#!/bin/bash
#SBATCH --job-name=CRC_age_scf_step8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step8_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step8_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow
python step8_age_conditioned_disease_scfoundation_CRC.py
echo "Exit: $?"
