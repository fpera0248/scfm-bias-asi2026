#!/bin/bash
#SBATCH --job-name=scgpt_CRC_eth_3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow/logs/scgpt_CRC_eth_3_%j.out
#SBATCH --error=/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow/logs/scgpt_CRC_eth_3_%j.err

set -eo pipefail

BASE=/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow

module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu

cd $BASE && python step3a_benchmark_scgpt_CRC_eth.py && python step3b_label_propagation_scgpt_CRC_eth.py
echo "Exit: $?"
