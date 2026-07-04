#!/bin/bash
#SBATCH --job-name=CRC_0b_sex
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/sex_Geneformer_workflow/logs/step0b_CRC_sex_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/sex_Geneformer_workflow/logs/step0b_CRC_sex_%j.err

module purge
module load miniforge3/25.3.0-3
source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
conda activate scdesign3_env
cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented_CRC/sex_Geneformer_workflow
Rscript step0b_scdesign3_CRC_sex.R
echo "Exit: $?"
