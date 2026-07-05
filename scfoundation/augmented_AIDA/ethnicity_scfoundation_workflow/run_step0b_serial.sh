#!/bin/bash
set -e

cd /data/scfoundation/augmented_AIDA/ethnicity_scfoundation_workflow
mkdir -p logs

CHAIN_LOG="logs/chain_$(date +%Y%m%d_%H%M%S).txt"
echo "started" | tee "$CHAIN_LOG"

rm -f stage1_shared_state.rds stage1_sce_*.rds stage2_synth_*.rds

S1=$(sbatch --parsable step0b_stage1.slurm)
echo "stage1 $S1" | tee -a "$CHAIN_LOG"

J1=$(sbatch --parsable --dependency=afterok:$S1 --job-name=s2_aa --export=ALL,BIN_LABEL_ENCODED=african__SP__american step0b_stage2.slurm)
echo "stage2_aa $J1" | tee -a "$CHAIN_LOG"

J2=$(sbatch --parsable --dependency=afterok:$S1 --job-name=s2_as --export=ALL,BIN_LABEL_ENCODED=asian step0b_stage2.slurm)
echo "stage2_as $J2" | tee -a "$CHAIN_LOG"

J3=$(sbatch --parsable --dependency=afterok:$S1 --job-name=s2_ea --export=ALL,BIN_LABEL_ENCODED=european__SP__american step0b_stage2.slurm)
echo "stage2_ea $J3" | tee -a "$CHAIN_LOG"

J4=$(sbatch --parsable --dependency=afterok:$S1 --job-name=s2_hi --export=ALL,BIN_LABEL_ENCODED=hispanic__SP__or__SP__latin step0b_stage2.slurm)
echo "stage2_hi $J4" | tee -a "$CHAIN_LOG"

J5=$(sbatch --parsable --dependency=afterok:$S1 --job-name=s2_na --export=ALL,BIN_LABEL_ENCODED=native__SP__american step0b_stage2.slurm)
echo "stage2_na $J5" | tee -a "$CHAIN_LOG"

S3=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3:$J4:$J5 step0b_stage3.slurm)
echo "stage3 $S3" | tee -a "$CHAIN_LOG"

PREV=$S3
J=$(sbatch --parsable --dependency=afterok:$PREV step2a_ethnicity_scfoundation.slurm); echo "step2a $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step3a_ethnicity_scfoundation.slurm); echo "step3a $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step3b_ethnicity_scfoundation.slurm); echo "step3b $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step4_ethnicity_scfoundation.slurm); echo "step4 $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step4a_ethnicity_scfoundation.slurm); echo "step4a $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step4b_ethnicity_scfoundation.slurm); echo "step4b $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step5_ethnicity_scfoundation.slurm); echo "step5 $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step6_ethnicity_scfoundation.slurm); echo "step6 $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step7_ethnicity_scfoundation.slurm); echo "step7 $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step8_ethnicity_scfoundation.slurm); echo "step8 $J" | tee -a "$CHAIN_LOG"; PREV=$J
J=$(sbatch --parsable --dependency=afterok:$PREV step9_ethnicity_scfoundation.slurm); echo "step9 $J" | tee -a "$CHAIN_LOG"; PREV=$J

echo "done. tail -f $CHAIN_LOG"
