#!/bin/bash
#SBATCH --job-name=CRC_eth_step3a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step3a_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step3a_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1
cd /data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step3a_benchmark_geneformer_ethnicity.py
echo "Exit: $?"
