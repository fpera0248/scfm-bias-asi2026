#!/bin/bash
set -euo pipefail
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow

ETHNICITIES=("indian" "japanese" "korean" "singaporean chinese" "singaporean indian" "singaporean malay" "thai")
S2_JIDS=""
for g in "${ETHNICITIES[@]}"; do
    encoded_g="${g// /__SP__}"
    safe_g=$(echo "$g" | tr ' ' '_')
    jid=$(sbatch --parsable \
        --job-name="s2_${safe_g}" \
        --export=ALL,BIN_LABEL_ENCODED="$encoded_g" \
        step0b_stage2.slurm)
    echo "Stage 2 [$g] -> $jid"
    S2_JIDS="${S2_JIDS}:${jid}"
done
S2_JIDS="${S2_JIDS#:}"

S3=$(sbatch --parsable --dependency=afterok:${S2_JIDS} step0b_stage3.slurm)
echo "Stage 3 -> $S3"
echo ""
squeue -u $USER
