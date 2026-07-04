#!/bin/bash
# ============================================================
# Submit step0b sex v6 chain
# Stage 1 -> 2 parallel Stage 2 jobs -> Stage 3
# ============================================================
set -euo pipefail

cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow
mkdir -p logs

EMAIL=your_email@example.com

# 2 sex bins
B1="female"
B2="male"

CHAIN_LOG="logs/chain_sex_step0b_$(date +%Y%m%d_%H%M%S).txt"
echo "Sex step0b v6 chain submitted $(date -Iseconds)" | tee "$CHAIN_LOG"

# Stage 1
S1=$(sbatch --parsable \
  --mail-type=BEGIN,FAIL \
  --mail-user=$EMAIL \
  step0b_stage1.slurm)
echo "Stage 1 -> $S1" | tee -a "$CHAIN_LOG"

# Stage 2
S2_IDS=""
submit_s2() {
  local bin_label=$1
  local job_name="sex_s2_${bin_label}"
  local jid
  jid=$(sbatch --parsable \
    --dependency=afterok:$S1 \
    --job-name=$job_name \
    --export=ALL,BIN_LABEL_ENCODED=$bin_label \
    --mail-type=FAIL \
    --mail-user=$EMAIL \
    step0b_stage2.slurm)
  echo "Stage 2 [$bin_label] -> $jid (afterok $S1)" | tee -a "$CHAIN_LOG"
  if [[ -z "$S2_IDS" ]]; then S2_IDS="$jid"; else S2_IDS="$S2_IDS:$jid"; fi
}

submit_s2 "$B1"
submit_s2 "$B2"

# Stage 3
S3=$(sbatch --parsable \
  --dependency=afterok:$S2_IDS \
  --mail-type=END,FAIL \
  --mail-user=$EMAIL \
  step0b_stage3.slurm)
echo "Stage 3 -> $S3 (afterok $S2_IDS)" | tee -a "$CHAIN_LOG"

# Watchdog
WATCHDOG=$(sbatch --parsable \
  --dependency=afterany:$S3 \
  --partition=batch --time=00:05:00 --mem=2G --cpus-per-task=1 \
  --mail-type=END --mail-user=$EMAIL \
  --job-name=sex_s0b_watchdog \
  --output=logs/watchdog_%j.out \
  --wrap="cd /oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow
echo '=== SEX step0b CHAIN STATUS ==='
echo \"Watchdog ran at: \$(date -Iseconds)\"
echo
sacct -j $S1,$S2_IDS,$S3 -X -o JobID%12,JobName%30,State,ExitCode,Elapsed | head -20
echo
ls -la ILD_Sex_Pilot_*_SEX.h5ad 2>/dev/null
echo
ls -la stage1_outputs/ 2>/dev/null
echo
ls -la stage2_outputs/ 2>/dev/null")
echo "Watchdog -> $WATCHDOG (afterany $S3)" | tee -a "$CHAIN_LOG"

echo "" | tee -a "$CHAIN_LOG"
echo "================================================" | tee -a "$CHAIN_LOG"
echo "SEX STEP0B v6 CHAIN SUBMITTED" | tee -a "$CHAIN_LOG"
echo "================================================" | tee -a "$CHAIN_LOG"
echo "monitor: squeue -u fperalta" | tee -a "$CHAIN_LOG"
echo "log: $CHAIN_LOG" | tee -a "$CHAIN_LOG"
