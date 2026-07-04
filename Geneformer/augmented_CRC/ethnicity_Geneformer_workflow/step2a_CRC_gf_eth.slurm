#!/bin/bash
#SBATCH --job-name=CRC_2a_gf_eth
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_2a_gf_eth_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow/logs/CRC_2a_gf_eth_%j.err

set -eo pipefail

module load miniforge3/25.3.0-3
source activate geneformer310
export PYTHONNOUSERSITE=1

cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/ethnicity_Geneformer_workflow
python step2a_embed_geneformer_CRC_eth.py
echo "Exit: $?"
