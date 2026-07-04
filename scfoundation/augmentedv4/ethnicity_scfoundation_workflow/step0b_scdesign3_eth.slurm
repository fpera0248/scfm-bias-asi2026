#!/bin/bash
#SBATCH --job-name=step0b_eth_v6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/step0b_scdesign3_eth_%j.out
#SBATCH --error=logs/step0b_scdesign3_eth_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fernando_peralta_castro@brown.edu

set -euo pipefail

echo "=== JOB START ==="
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK}"
echo "mem=${SLURM_MEM_PER_NODE:-unknown}"
echo "start=$(date -Iseconds)"

cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/ethnicity_scfoundation_workflow

module purge
module load miniforge3/25.3.0-3
source activate scdesign3_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export R_MAX_VSIZE=240Gb

Rscript --vanilla step0b_scdesign3_ethnicity_augmentation.R

echo "=== JOB END ==="
echo "end=$(date -Iseconds)"
