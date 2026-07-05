# Environments

The pipeline uses four separate conda environments. Their dependencies conflict
(different Python versions, CUDA vs. R), so they cannot share one environment.

| Environment       | Used for                                   | Runtime                              |
|-------------------|--------------------------------------------|--------------------------------------|
| `scfoundation_gpu`| scFoundation embedding (`step2a`)          | Python 3.10                          |
| `geneformer310`   | Geneformer embedding (`step2a`)            | Python 3.10                          |
| `scgpt310`        | scGPT embedding (`step2a`)                 | Python 3.9 + `scgpt==0.2.1` (pip)    |
| `scdesign3_env`   | scDesign3 synthetic generation (`step0b`)  | R 4.3 + `scDesign3` 1.5.0 (source)   |

## Quick start

```
bash install.sh          # build all four from the slim specs (recommended)
bash install.sh --full   # build from the fully pinned specs (exact Linux/CUDA reproduction)
```

Then activate whichever environment a step needs, e.g. `conda activate scfoundation_gpu`.

## Prerequisites

- **conda or miniforge/mamba.** The original runs used miniforge3/25.3.0-3. If you
  don't have conda, install Miniforge: https://github.com/conda-forge/miniforge
  (`mamba` gives a much faster solve and `install.sh` uses it automatically if present).
- **Linux with an NVIDIA GPU** for the three model environments. `scdesign3_env`
  is CPU/R only.

## Spec files

Each environment ships two specs:

- `environment_<name>.yml` — slim, curated top-level dependencies. Solves fastest;
  the recommended starting point.
- `environment_<name>.full.yml` — full `conda env export`, every transitive
  dependency and build string pinned. Reproduces the original Linux/CUDA environment
  down to the build; it is platform-specific and may not solve off Linux.

For cross-checking, `piplist_<name>.txt` is the exact `pip list` for each environment,
and `Rpackages_scdesign3_env.csv` is the exact R package set for `scdesign3_env`.

## Manual build (what `install.sh` automates)

Swap `environment_<name>.yml` for `environment_<name>.full.yml` in any command below
to build from the fully pinned spec instead.

### 1. scfoundation_gpu
```
conda env create -f environment_scfoundation_gpu.yml
```

### 2. geneformer310
```
conda env create -f environment_geneformer310.yml
```

### 3. scgpt310 — needs a pip step
scGPT is not on conda. Create the environment, then install scGPT from pip:
```
conda env create -f environment_scgpt310.yml
conda run -n scgpt310 pip install scgpt==0.2.1
```

### 4. scdesign3_env — needs an R source install
scDesign3 is not on conda. Create the environment (R 4.3 and its dependencies),
then install scDesign3 from source with `devtools`:
```
conda env create -f environment_scdesign3_env.yml
conda run -n scdesign3_env Rscript -e 'devtools::install_github("SONGDONGYUAN1994/scDesign3")'
```
The pinned version is scDesign3 1.5.0; see `Rpackages_scdesign3_env.csv` for the full
R package set the original runs used.

## Model weights

The environments provide the **code** to run each model. The pretrained **weights**
are separate downloads with their own licenses and are not in this repo. See the
**Model setup** section of `README.md` for the checkpoint each environment expects.
