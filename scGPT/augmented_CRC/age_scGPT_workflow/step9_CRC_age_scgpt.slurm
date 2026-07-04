#!/bin/bash
#SBATCH --job-name=CRC_age_scgpt_step9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/age_scGPT_workflow/logs/CRC_age_scgpt_step9_%j.out
#SBATCH --error=/oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/age_scGPT_workflow/logs/CRC_age_scgpt_step9_%j.err

set -eo pipefail

module purge
module load miniforge3/25.3.0-3
source activate scgpt310
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt310/lib/libstdc++.so.6
cd /oscar/home/fperalta/data/fperalta/scGPT/augmented_CRC/age_scGPT_workflow
python step9_visualizations_scgpt_CRC_age.py
echo "Exit: $?"
