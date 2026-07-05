#!/bin/bash
set -euo pipefail
cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow

if [[ $# -eq 1 ]]; then
    DEP="--dependency=afterok:$1"
    echo "Chaining downstream after job $1"
else
    DEP=""
    echo "Launching downstream immediately (stage3 should be complete)"
fi

J2A=$(sbatch --parsable $DEP step2a_ethnicity_scfoundation.slurm)
echo "step2a (embed) -> $J2A"

J3A=$(sbatch --parsable --dependency=afterok:$J2A step3a_ethnicity_scfoundation.slurm)
echo "step3a (scIB)  -> $J3A"

J7=$(sbatch --parsable --dependency=afterok:$J3A step7_ethnicity_scfoundation.slurm)
echo "step7 (cell_type classification per ethnicity) -> $J7"

squeue -u $USER
