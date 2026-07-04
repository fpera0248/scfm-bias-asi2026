#!/bin/bash
#SBATCH --job-name=CRC_eth_scf_step4b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow/logs/CRC_eth_scf_step4b_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow/logs/CRC_eth_scf_step4b_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow
python step4b_robustness_scfoundation_CRC_eth.py
echo "Exit: $?"
