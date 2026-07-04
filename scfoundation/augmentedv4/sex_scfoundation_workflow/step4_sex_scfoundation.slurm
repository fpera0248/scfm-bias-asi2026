#!/bin/bash
#SBATCH --job-name=sex_step4_sf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_step4_sf_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_step4_sf_%j.err

set -e; set -o pipefail
module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu

export OMP_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow
python step4_external_validation_scfoundation_sex.py
echo "Job finished at: $(date) | Exit: $?"
