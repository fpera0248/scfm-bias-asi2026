#!/bin/bash
#SBATCH --job-name=CRC_eth_step7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step7_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step7_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1
cd /data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step7_representation_diagnostics_geneformer.py
echo "Exit: $?"
