#!/bin/bash
#SBATCH --job-name=sex_step5_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step5_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step5_gf_%j.err

set -e
set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate geneformer310
set -u

mkdir -p /data/Geneformer/augmented/sex_Geneformer_workflow/logs
cd /data/Geneformer/augmented/sex_Geneformer_workflow

export OMP_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

python step5_fairness_geneformer_sex.py
echo "Job finished at: $(date) | Exit: $?"
