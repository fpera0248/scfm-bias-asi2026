#!/bin/bash
#SBATCH --job-name=sex_step2a_gf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/data/rsingh47/fperalta/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step2a_gf_%j.out
#SBATCH --error=/oscar/data/rsingh47/fperalta/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step2a_gf_%j.err

set -euo pipefail

echo "================================================="
echo " STEP 2a -- Geneformer Embedding (SEX)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $(hostname)"
echo " Start time:  $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate geneformer310
set -u

cd /oscar/home/fperalta/data/fperalta/Geneformer/augmented/sex_Geneformer_workflow
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python step2a_embed_geneformer_sex.py
EXIT=$?
echo "Step 2a finished at: $(date) | Exit: $EXIT"
exit $EXIT
