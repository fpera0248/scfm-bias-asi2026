#!/bin/bash
#SBATCH --job-name=CRC_eth_step3b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step3b_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step3b_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1
cd /data/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step3b_label_propagation_geneformer_ethnicity.py
echo "Exit: $?"
