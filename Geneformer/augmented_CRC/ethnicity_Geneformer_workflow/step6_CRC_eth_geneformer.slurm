#!/bin/bash
#SBATCH --job-name=CRC_eth_step6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step6_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_eth_step6_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1
cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step6_per_ethnicity_diagnostics_geneformer.py
echo "Exit: $?"
