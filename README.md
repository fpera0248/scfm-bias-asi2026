# Demographic Bias in Single-Cell Foundation Models

Code to reproduce *Demographic Bias in Single-Cell Foundation Models: Evaluation and Mitigation Across Models and Cohorts* (ACM-BCB ASI 2026 workshop). We measure how well scFoundation, Geneformer, and scGPT recover cell types for underrepresented demographic groups, and we test scDesign3 synthetic augmentation as a mitigation.

Cohorts: interstitial lung disease (ILD, ethnicity imbalance 127:1), colorectal cancer (CRC, 17:1), and the Asian Immune Diversity Atlas (AIDA, 8.4:1). Demographic axes: ethnicity (main analysis), sex and age (supplementary). Metrics: iLISI for embedding mixing, worst-group cell-type macro-F1 for recovery.

## What this repo contains

This repo holds the pipeline code and the pinned conda environment specs for the three models plus the scDesign3 augmentation step. It does not host the datasets or the model weights, which each carry their own license and access terms. Reproducing the numbers needs a Linux machine with an NVIDIA GPU, the three datasets, and the three model weight sets.

## Start here

This repo serves two uses. To reproduce every number in the paper, you need all 24 workflows, since each one corresponds to a specific reported result. To learn how the pipeline works and adapt it to your own data, follow one workflow end to end.

The reference workflow is `scfoundation/augmentedv4/ethnicity_scfoundation_workflow`. It runs scFoundation on the ILD cohort along the ethnicity axis, the main-text setup. Read "The pipeline, stage by stage" below, then open that folder and read its `step*.py` scripts in the order listed under Running a workflow. Every stage of the analysis lives there.

The demographic axis changes little across the three. The ethnicity workflow is the reference and groups cells by `self_reported_ethnicity`. The sex workflow is identical, grouping by `sex`. The age workflow adds one step, `step0a`, which extracts raw counts and bins age; ethnicity and sex reuse the existing raw counts and skip it.

The remaining workflows run the same pipeline for the other two models and the other two cohorts. Only the embedding stage differs by model, and the data files and disease-column setup differ by cohort. Use them for full reproduction. You do not need to read all of them to understand the method.

## Requirements

- Linux with an NVIDIA GPU. The original runs used Brown's Oscar HPC cluster.
- conda or miniforge. The original runs used miniforge3/25.3.0-3.
- The three datasets (see Data).
- The three model weight sets (see Model setup).

## The pipeline, stage by stage

Every workflow runs the same stages. Read this section to understand what each stage measures and why it matters, then use Running a workflow for the commands.

**1. Load and label.** Read the `.h5ad`, assign each cell to its demographic group from the `obs` column, and drop cells whose label is missing or a placeholder (`"nan"`, `"unknown"`). The analysis compares groups, so a pooled or mislabeled group corrupts every downstream number.

**2. Build the four rebalancing conditions.** From the loaded data, construct four versions:
- **Proportional:** the native class balance, untouched. Shows the bias as it exists.
- **BalancedAugmented:** scDesign3 generates synthetic minority cells until each group reaches the majority count.
- **BalancedUpsampled:** duplicate existing minority cells to that same target count.
- **Downsampled:** subsample every group down to the smallest group's count.

Each condition isolates a different mechanism: proportional measures the problem, augmentation adds new cells, upsampling adds copies, downsampling removes cells. Comparing them separates a real fix from an artifact.

**3. Generate synthetic cells with scDesign3.** Fit per-gene marginal distributions and the gene-gene correlation structure on the real minority cells, then simulate new cells in expression space. The synthetic cells pass through the same frozen encoder as real cells, so they have to look like real expression, not like points shifted in embedding space. Fitting the correlation structure is what keeps them from collapsing to noise.

**4. Embed with the frozen foundation model.** Pass each condition through scFoundation, Geneformer, or scGPT without fine-tuning, and keep the cell embeddings. These models get deployed as-is and reused across tasks, so freezing them measures the bias a downstream user actually inherits. This is the only stage that differs across the three models.

**5. Measure mixing with iLISI.** For each cell's local neighborhood in the embedding, measure how well demographic groups interleave: 1 is full mixing, 0 is full separation. A minority group sitting in its own region means the model encoded demographic structure, which is the bias signal.

**6. Validate the synthetic cells.** Run KS tests comparing synthetic against real cells on library size and gene sparsity. Augmentation only counts as a fix if the synthetic cells match real ones. If they differ, a high downstream score is memorization of an artifact, not recovery.

**7. Classify cell type on the frozen embeddings.** Train logistic regression and random forest on the embeddings to predict cell type. Running both matters: a result that holds under only one classifier is a classifier artifact, and one that holds under both is a property of the embedding.

**8. Trace learning curves.** Sweep the training-set size and record validation macro-F1 at each size. A curve that climbs steadily is genuine learning. A curve that jumps and then flatlines is memorization of duplicated cells. This is how upsampling's inflated internal score gets exposed.

**9. Run external validation and worst-group macro-F1.** Evaluate on a shared held-out set the model never trained on, compute macro-F1 for the smallest minority group, and bootstrap it over 1000 resamples for a confidence interval. Internal cross-validation leaks duplicated cells, and an overall score hides the group that fails, so the worst-group score on held-out data is the honest measure of who the model leaves behind.

**10. Classify disease (ILD and CRC only).** As a secondary check, predict normal versus disease and compute worst-group disease accuracy. This confirms the model ranking is not specific to cell-type labels. The smallest minorities have too few cells for a reliable per-group disease number, so treat it as directional, not a headline result. AIDA donors are all healthy, so this stage does not run on AIDA.

## Repo layout

One folder per (model, cohort, demographic) combination. The ILD root differs by model, so the pattern is not uniform. The exact roots:

**ILD**
```
scfoundation/augmentedv4/{demographic}_scfoundation_workflow
Geneformer/augmented/{demographic}_Geneformer_workflow
scGPT/{demographic}_scGPT_workflow
```

**CRC**
```
{model}/augmented_CRC/{demographic}_{model}_workflow
```

**AIDA**
```
{model}/augmented_AIDA/{demographic}_{model}_workflow
```

`{demographic}` is `age`, `ethnicity`, or `sex`. `{model}` is `scfoundation`, `Geneformer`, or `scGPT`. That gives 24 workflows: three models times three cohorts times three demographics, minus the three AIDA sex workflows.

AIDA has no sex workflow for any model. AIDA is sex-balanced, so the sex analysis does not apply and the paper does not report it. This is by design, not a missing file.

Ethnicity workflows produce the main-text results. Sex and age workflows produce the supplementary results.

## Model setup

Four conda environments are used: one per model, plus one for the scDesign3 synthetic-cell generation (the R `step0b`). Their dependencies conflict, so keep them separate.

Each environment ships as two files:

- `environment_<name>.yml` — a slim, curated spec listing the top-level dependencies. Solves fastest; recommended starting point.
- `environment_<name>.full.yml` — the exact `conda env export` with every transitive dependency and build string pinned. Use it to reproduce the original Linux/CUDA environment down to the build. It is platform-specific and may not solve on non-Linux machines.

For cross-checking, `piplist_<name>.txt` records the exact `pip list` output for each environment, capturing pip-installed packages that the conda export may not fully pin.

Create from the slim spec (swap in the matching `.full.yml` for exact pinning):

scFoundation
```
conda env create -f environment_scfoundation_gpu.yml
```
Model code and weights: [FILL: scFoundation source and checkpoint version]

Geneformer
```
conda env create -f environment_geneformer310.yml
```
Model code and weights: [FILL: Geneformer source and checkpoint version]

scGPT
```
conda env create -f environment_scgpt310.yml
```
Model code and weights: [FILL: scGPT source and checkpoint version]

scDesign3 (synthetic augmentation, the R `step0b`; R 4.3)
```
conda env create -f environment_scdesign3_env.yml
```
scDesign3 itself is installed from source into this environment with `devtools`: [FILL: scDesign3 source and version].

Pin the exact checkpoint for each model. A different checkpoint produces different embeddings and will not reproduce the reported numbers.

## Data

The three cohorts come from CZ CELLxGENE. This repo ships download commands, not the files, since CELLxGENE redistributes them under their own licenses. `fetch_data.sh` pulls all three into `data/`.

- ILD: Natri et al. 2024, Nature Genetics. GEO GSE227136, doi 10.1038/s41588-024-01702-0.
  https://datasets.cellxgene.cziscience.com/c3d9262e-0dc5-4eca-bf20-56e6d96d0306.h5ad
- CRC: Moorman et al. 2024, Nature. doi 10.1038/s41586-024-08560-0, epithelial compartment.
  https://datasets.cellxgene.cziscience.com/66cadf3b-4c71-4930-8add-fa748745704d.h5ad
- AIDA: Kock et al. 2025, Cell. Phase 1 Data Freeze v2.
  https://datasets.cellxgene.cziscience.com/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad

These URLs point at the raw published objects. The pipeline filters them, so the raw cell counts do not match the post-QC counts in the paper. The raw AIDA v2 object holds 1,265,624 cells; the pipeline uses 1,165,872. Each dataset keeps its own CELLxGENE license, so check the terms before redistributing or publishing derived data.

## Input format

Each workflow reads an AnnData `.h5ad`. To run on your own data, the file needs:

- an expression matrix in `X`,
- an `obs` cell-type column (`cell_type`),
- an `obs` demographic column for the axis you run: `self_reported_ethnicity`, `sex`, or the age field.

Two rules the pipeline depends on:

- Synthetic cells carry the literal string `"nan"` in `self_reported_ethnicity`, `disease`, and `sex`. Exclude `"nan"` from every per-group count. Do not fold it into a real group.
- The age workflow uses `"unknown"` as an ethnicity value. Exclude it the same way.

Each model expects its own gene panel and tokenization. Preprocess your `.h5ad` to the model's expected input before the embedding stage, or the model will not produce valid representations.

## Running a workflow

Run the `step*.py` scripts in numerical order from inside a workflow folder. The sequence:

- `step0a`: extract raw counts. Present only in age workflows; the other axes reuse its output.
- `step0b`: scDesign3 synthetic generation, written in R, run as stages 1 through 3.
- `step2a`: embed the cells with the frozen foundation model.
- `step3a`: benchmark.
- `step3b`: label propagation.
- `step4`: external validation. Computes worst-group macro-F1 on the held-out set.
- `step4a`: downstream classification.
- `step4b`: robustness stress tests.
- `step5`: learning curves.
- `step6`: per-group diagnostics.
- `step7`: representation diagnostics.
- `step8`: disease classification (ILD and CRC).
- `step9`: visualizations.

This is the sequence in the reference workflow; the core pipeline uses `step2a`, `step3a`, and `step3b` rather than standalone `step2` or `step3`. Some other workflows carry additional diagnostic or validation scripts around this core (for example `step0c` external validation in the age and sex workflows, `step1a` integrity checks, `step3d`, and `step7b`). They sort into place by name and are optional extras, not new required stages.

Set the input path before running. Each step script hardcodes an absolute Oscar path near the top (the `BASE` variable). Edit it to point at your data location. This release does not include a config-driven version that sets the path once.

Seeding uses `numpy.random.default_rng` with a fixed `RANDOM_STATE` defined near the top of each script.

## Reproducing across the three models

The stages above are identical across scFoundation, Geneformer, and scGPT. Only stage 4, the embedding, differs. To reproduce all three, run the matching workflow for each. The iLISI, classification, learning-curve, and worst-group scripts run the same logic on whatever embeddings stage 4 produced.

## Auditing a new foundation model

The evaluation stages (5 through 10) operate on cell embeddings and do not depend on which model produced them, so the audit generalizes to any frozen single-cell model that maps cells to a fixed-length embedding.

The scripts are per-model copies, not a plugin interface. To audit a new model today:

1. Copy an existing workflow folder.
2. Replace the embedding stage (`step2a`) with your model's `h5ad` to embedding call, keeping the same output format (one embedding row per cell, aligned to `obs`).
3. Run the downstream stages unchanged.

## Data-quality notes

- Ethnicity values are cased inconsistently across cohorts. Lowercase-normalize before any per-group count, or a group splits into two buckets.
- `"nan"` (synthetic cells) and `"unknown"` (age workflow) are always exclusions, never a group.
