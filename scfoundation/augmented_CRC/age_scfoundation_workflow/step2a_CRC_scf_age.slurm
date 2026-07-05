#!/bin/bash
#SBATCH --job-name=CRC_2a_scf_age
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_2a_scf_age_%j.out
#SBATCH --error=/data/scfoundation/augmented_CRC/age_scfoundation_workflow/logs/CRC_2a_scf_age_%j.err

set -eo pipefail

module load miniforge3/25.3.0-3
source activate scfoundation_gpu
export PYTHONNOUSERSITE=1

cd /data/scfoundation/augmented_CRC/age_scfoundation_workflow
python step2a_embed_scfoundation_CRC_age.py
echo "Exit: $?"
