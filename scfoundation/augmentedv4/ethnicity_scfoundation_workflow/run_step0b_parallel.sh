#!/bin/bash
set -euo pipefail
cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/ethnicity_scfoundation_workflow
mkdir -p logs

CHAIN_LOG="logs/chain_$(date +%Y%m%d_%H%M%S).txt"
echo "chain started $(date -Iseconds)" | tee -a "$CHAIN_LOG"

# Clear stale stage state (DO NOT touch existing checkpoints_ethnicity_v6/)
rm -f stage1_shared_state.rds stage1_sce_*.rds stage2_synth_*.rds stage1_sce_full.rds stage1_sce_detect.rds

# Stage 1
S1=$(sbatch --parsable step0b_stage1.slurm)
echo "Stage 1 setup -> job $S1" | tee -a "$CHAIN_LOG"

# Stage 2: 5 parallel jobs (one per group)
GROUPS=()
GROUPS+=("african american")
GROUPS+=("asian")
GROUPS+=("european american")
GROUPS+=("hispanic or latin")
GROUPS+=("native american")
S2_JIDS=""
for g in "${GROUPS[@]}"; do
  safe_g=$(echo "$g" | tr ' ' '_')
  # Encode spaces with __SP__ since slurm --export breaks on spaces
  encoded_g="${g// /__SP__}"
  jid=$(sbatch --parsable \
    --dependency=afterok:$S1 \
    --job-name="s2_${safe_g}" \
    --export=ALL,BIN_LABEL_ENCODED="$encoded_g" \
    step0b_stage2.slurm)
  echo "Stage 2 [$g] -> job $jid (afterok $S1)" | tee -a "$CHAIN_LOG"
  S2_JIDS="${S2_JIDS}:${jid}"
done
S2_JIDS="${S2_JIDS#:}"

# Stage 3: aggregate
S3=$(sbatch --parsable --dependency=afterok:${S2_JIDS} step0b_stage3.slurm)
echo "Stage 3 aggregate -> job $S3 (afterok all stage 2)" | tee -a "$CHAIN_LOG"

# Then chain the rest of the pipeline
PREV=$S3
for s in step2a step3a step3b step4 step4a step4b step5 step6 step7 step8 step9; do
  slurm="${s}_ethnicity_scfoundation.slurm"
  if [[ ! -f "$slurm" ]]; then
    echo "SKIP missing: $slurm" | tee -a "$CHAIN_LOG"
    continue
  fi
  jid=$(sbatch --parsable --dependency=afterok:$PREV "$slurm")
  echo "$s -> job $jid (afterok $PREV)" | tee -a "$CHAIN_LOG"
  PREV=$jid
done

# Final consolidator runs even on failure
AGG=$(sbatch --parsable --dependency=afterany:$PREV --partition=batch \
  --time=00:10:00 --mem=4G --cpus-per-task=1 \
  --output=logs/consolidate_%j.out \
  --wrap="cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/ethnicity_scfoundation_workflow; { echo '==== CHAIN CONSOLIDATED LOG ===='; echo \"generated: \$(date -Iseconds)\"; for f in \$(ls -tr logs/stage*.out logs/stage*.err logs/step*.out logs/step*.err 2>/dev/null); do echo; echo '================================================================'; echo \"FILE: \$f\"; echo '================================================================'; cat \"\$f\"; done; } > logs/chain_consolidated_\$(date +%Y%m%d_%H%M%S).txt")
echo "Consolidator -> job $AGG (afterany $PREV)" | tee -a "$CHAIN_LOG"

echo | tee -a "$CHAIN_LOG"
echo "monitor:  squeue -u \$USER" | tee -a "$CHAIN_LOG"
echo "consolidated: logs/chain_consolidated_*.txt" | tee -a "$CHAIN_LOG"
