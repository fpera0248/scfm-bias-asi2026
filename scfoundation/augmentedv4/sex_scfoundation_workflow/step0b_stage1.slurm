#!/bin/bash
#SBATCH --job-name=sex_s0b_stage1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_stage1_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow/logs/sex_stage1_%j.err

set -euo pipefail

echo "================================================="
echo " STAGE 1: SETUP + HVG + ANCHOR CACHE (SEX)"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $(hostname)"
echo " CPUs:       $SLURM_CPUS_PER_TASK"
echo " Start:      $(date)"
echo "================================================="

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
set +u
conda activate scdesign3_env
set -u

cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript step0b_stage1_setup.R
EXIT=$?
echo "Stage 1 finished at: $(date) | Exit: $EXIT"
exit $EXIT
