#!/bin/bash
#SBATCH --job-name=eth_step8_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=30G
#SBATCH --time=64:00:00
#SBATCH --output=/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step8_gf_%j.out
#SBATCH --error=/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow/logs/eth_step8_gf_%j.err

set -e
set -o pipefail

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate geneformer310
set -u

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"
mkdir -p "$BASE_DIR/logs"
cd "$BASE_DIR"

export OMP_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export NUMEXPR_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

python step8_eth_conditioned_disease_geneformer.py

echo "Job finished at: $(date) | Exit: $?"