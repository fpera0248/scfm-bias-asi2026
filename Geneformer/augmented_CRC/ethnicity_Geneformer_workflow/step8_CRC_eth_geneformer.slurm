#!/bin/bash
#SBATCH --job-name=CRC_eth_step8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step8_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step8_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1
cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step8_eth_conditioned_disease_geneformer.py
echo "Exit: $?"
