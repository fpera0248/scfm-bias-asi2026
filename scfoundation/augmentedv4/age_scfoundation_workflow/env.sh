module purge
module load miniforge3/25.3.0-3

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
