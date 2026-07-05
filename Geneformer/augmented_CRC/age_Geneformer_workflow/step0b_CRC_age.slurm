#!/bin/bash
#SBATCH --job-name=CRC_0b_age
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/step0b_CRC_age_%j.out
#SBATCH --error=/data/Geneformer/augmented_CRC/age_Geneformer_workflow/logs/step0b_CRC_age_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
conda activate scdesign3_env
cd /data/Geneformer/augmented_CRC/age_Geneformer_workflow
Rscript step0b_scdesign3_CRC_age.R
echo "Exit: $?"
