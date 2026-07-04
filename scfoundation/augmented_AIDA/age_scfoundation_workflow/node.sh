interact -q batch -n 8 -m 100g -t 64:00:00
module purge
module load miniforge3/25.3.0-3

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu