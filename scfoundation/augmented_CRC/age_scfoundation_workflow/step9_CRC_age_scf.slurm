#!/bin/bash
#SBATCH --job-name=CRC_age_scf_step9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step9_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_age_scf_step9_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow
python step9_visualizations_scfoundation_CRC_age.py
echo "Exit: $?"
