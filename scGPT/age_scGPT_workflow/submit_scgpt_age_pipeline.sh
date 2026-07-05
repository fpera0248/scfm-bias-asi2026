#!/bin/bash
BASE=/data/scGPT/age_scGPT_workflow

J2=$(sbatch --parsable $BASE/step2a_scgpt_age.slurm)
J3=$(sbatch --parsable --dependency=afterok:$J2 $BASE/step3_scgpt_age.slurm)

J4=$(sbatch  --parsable --dependency=afterok:$J3 $BASE/step4_scgpt_age.slurm)
J4a=$(sbatch --parsable --dependency=afterok:$J3 $BASE/step4a_scgpt_age.slurm)
J4b=$(sbatch --parsable --dependency=afterok:$J3 $BASE/step4b_scgpt_age.slurm)
J5=$(sbatch  --parsable --dependency=afterok:$J3 $BASE/step5_scgpt_age.slurm)
J6=$(sbatch  --parsable --dependency=afterok:$J3 $BASE/step6_scgpt_age.slurm)
J7=$(sbatch  --parsable --dependency=afterok:$J3 $BASE/step7_scgpt_age.slurm)
J8=$(sbatch  --parsable --dependency=afterok:$J3 $BASE/step8_scgpt_age.slurm)

J9=$(sbatch --parsable \
    --dependency=afterok:$J4:$J4a:$J4b:$J5:$J6:$J7:$J8 \
    $BASE/step9_scgpt_age.slurm)

echo "Submitted:"
echo "  step2a : $J2"
echo "  step3  : $J3 (depends on 2a)"
echo "  step4  : $J4  (depends on 3)"
echo "  step4a : $J4a (depends on 3)"
echo "  step4b : $J4b (depends on 3)"
echo "  step5  : $J5  (depends on 3)"
echo "  step6  : $J6  (depends on 3)"
echo "  step7  : $J7  (depends on 3)"
echo "  step8  : $J8  (depends on 3)"
echo "  step9  : $J9  (depends on 4-8)"
