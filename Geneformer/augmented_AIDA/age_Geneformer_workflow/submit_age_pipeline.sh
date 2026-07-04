#!/bin/bash
BASE=/oscar/home/fperalta/data/fperalta/Geneformer/augmented_AIDA/age_Geneformer_workflow

J4=$(sbatch  --parsable $BASE/step4_geneformer_age.slurm)
J4a=$(sbatch --parsable $BASE/step4a_geneformer_age.slurm)
J4b=$(sbatch --parsable $BASE/step4b_geneformer_age.slurm)
J5=$(sbatch  --parsable $BASE/step5_geneformer_age.slurm)
J6=$(sbatch  --parsable $BASE/step6_geneformer_age.slurm)
J7=$(sbatch  --parsable $BASE/step7_geneformer_age.slurm)
J8=$(sbatch  --parsable $BASE/step8_geneformer_age.slurm)

J9=$(sbatch --parsable \
    --dependency=afterok:$J4:$J4a:$J4b:$J5:$J6:$J7:$J8 \
    $BASE/step9_geneformer_age.slurm)

echo "Submitted:"
echo "  step4  : $J4"
echo "  step4a : $J4a"
echo "  step4b : $J4b"
echo "  step5  : $J5"
echo "  step6  : $J6"
echo "  step7  : $J7"
echo "  step8  : $J8"
echo "  step9  : $J9 (depends on all above)"
