#!/bin/bash
# Submit all pipeline steps with afterok dependency chain.
# Each step only runs if the previous finished successfully.
# All .out/.err files land in logs/, plus one consolidated log.
set -euo pipefail
cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow
mkdir -p logs

CHAIN_LOG="logs/chain_$(date +%Y%m%d_%H%M%S).txt"
echo "chain started $(date -Iseconds)" | tee -a "$CHAIN_LOG"

STEPS=(
  step0b_scdesign3_eth.slurm
  step2a_ethnicity_scfoundation.slurm
  step3a_ethnicity_scfoundation.slurm
  step3b_ethnicity_scfoundation.slurm
  step4_ethnicity_scfoundation.slurm
  step4a_ethnicity_scfoundation.slurm
  step4b_ethnicity_scfoundation.slurm
  step5_ethnicity_scfoundation.slurm
  step6_ethnicity_scfoundation.slurm
  step7_ethnicity_scfoundation.slurm
  step8_ethnicity_scfoundation.slurm
  step9_ethnicity_scfoundation.slurm
)

PREV_JID=""
for step in "${STEPS[@]}"; do
  if [[ ! -f "$step" ]]; then
    echo "SKIP missing: $step" | tee -a "$CHAIN_LOG"
    continue
  fi
  if [[ -z "$PREV_JID" ]]; then
    JID=$(sbatch --parsable "$step")
  else
    JID=$(sbatch --parsable --dependency=afterok:"$PREV_JID" "$step")
  fi
  echo "submitted $step -> job $JID  (depends on ${PREV_JID:-none})" | tee -a "$CHAIN_LOG"
  PREV_JID="$JID"
done

# Submit a final aggregator job that runs after the whole chain.
# This concatenates every step's .out and .err into one consolidated file.
AGG_SCRIPT="logs/_aggregate_$(date +%Y%m%d_%H%M%S).slurm"
cat > "$AGG_SCRIPT" <<AGGEOF
#!/bin/bash
#SBATCH --job-name=chain_aggregate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/_aggregate_%j.out
#SBATCH --error=logs/_aggregate_%j.err
#SBATCH --dependency=afterany:$PREV_JID

set -euo pipefail
cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow

OUT="logs/chain_consolidated_\$(date +%Y%m%d_%H%M%S).txt"
{
  echo "==================== CHAIN CONSOLIDATED LOG ===================="
  echo "generated: \$(date -Iseconds)"
  echo "directory: \$(pwd)"
  echo "jobs in order:"
  sacct --format=JobID,JobName%30,State,ExitCode,Start,End,Elapsed -j $PREV_JID -X 2>/dev/null || true
  echo
  for f in \$(ls -tr logs/step*_*.out logs/step*_*.err 2>/dev/null); do
    echo
    echo "================================================================"
    echo "FILE: \$f"
    echo "================================================================"
    cat "\$f"
  done
} > "\$OUT"
echo "consolidated log written to \$OUT"
AGGEOF

AGG_JID=$(sbatch --parsable "$AGG_SCRIPT")
echo "submitted aggregator -> job $AGG_JID  (runs after chain ends, success or fail)" | tee -a "$CHAIN_LOG"

echo | tee -a "$CHAIN_LOG"
echo "chain submitted. monitor with:  squeue -u \$USER" | tee -a "$CHAIN_LOG"
echo "final consolidated log will be in logs/chain_consolidated_*.txt" | tee -a "$CHAIN_LOG"
