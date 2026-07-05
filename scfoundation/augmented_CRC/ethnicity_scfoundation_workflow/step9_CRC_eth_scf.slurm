#!/bin/bash
#SBATCH --job-name=CRC_eth_scf_step9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow/logs/CRC_eth_scf_step9_%j.out
#SBATCH --error=/data/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow/logs/CRC_eth_scf_step9_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd /data/scfoundation/augmented_CRC/ethnicity_scfoundation_workflow
python step9_visualizations_scfoundation_CRC_eth.py
echo "Exit: $?"
