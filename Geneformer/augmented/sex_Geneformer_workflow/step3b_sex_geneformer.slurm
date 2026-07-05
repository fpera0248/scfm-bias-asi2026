#!/bin/bash
#SBATCH --job-name=sex_step3b_gf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step3b_gf_%j.out
#SBATCH --error=/data/Geneformer/augmented/sex_Geneformer_workflow/logs/sex_step3b_gf_%j.err

set -euo pipefail

echo "================================================="
echo " STEP 3b -- Label Propagation (SEX, Geneformer)"
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

cd /data/Geneformer/augmented/sex_Geneformer_workflow
mkdir -p logs

python step3b_label_propagation_geneformer_sex.py
EXIT=$?
echo "Step 3b finished at: $(date) | Exit: $EXIT"
exit $EXIT
