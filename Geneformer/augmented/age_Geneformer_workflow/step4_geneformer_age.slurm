#!/bin/bash
#SBATCH --job-name=gf_age_4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow/logs/gf_age_4_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow/logs/gf_age_4_%j.err
BASE=/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow
module load miniforge3/25.3.0-3
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate geneformer310
export PYTHONNOUSERSITE=1
cd $BASE && python step4_external_validation_geneformer_age.py
echo "Exit: $?"
