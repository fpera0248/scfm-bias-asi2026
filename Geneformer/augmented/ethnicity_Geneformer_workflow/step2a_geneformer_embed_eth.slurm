#!/bin/bash
#SBATCH --job-name=gf_eth_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gf_eth_embed_%j.out
#SBATCH --error=logs/gf_eth_embed_%j.err

echo "=========================================="
echo " Geneformer V2-316M ETHNICITY Embedding"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node(s):     $SLURM_NODELIST"
echo " Start Time:  $(date)"
echo "=========================================="

BASE_DIR="/oscar/data/rsingh47/fperalta/Geneformer/augmented/ethnicity_Geneformer_workflow"

cd "$BASE_DIR" || {
    echo "Failed to cd into project directory"
    exit 1
}

mkdir -p logs

echo "Working directory:"
pwd

echo "GPU visibility check"
nvidia-smi || {
    echo "GPU not visible"
    exit 1
}

module load miniforge3/25.3.0-3 || exit 1

source "$(dirname $(which conda))/../etc/profile.d/conda.sh" || {
    echo "Failed to source conda.sh"
    exit 1
}

set +u
conda activate geneformer310
set -u || {
    echo "Failed to activate scfoundation_gpu"
    exit 1
}

echo "Python diagnostics:"
python - << 'EOF'
import sys, torch
print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
    total_gb = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[RUNNING STEP 2a -- Geneformer V2-316M embedding (ETHNICITY)]"
date

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python step2a_ethnicity_embed_geneformer.py

EXIT_CODE=$?

echo "------------------------------------------"
echo " Job finished at: $(date)"
echo " Exit code:      $EXIT_CODE"
echo "------------------------------------------"

exit $EXIT_CODE