# Container: reproducible environment, bring your own data

A single image bakes all four conda environments and the model code so you can mount
your own `.h5ad` data and run any workflow without building anything. GPU comes from
the **host driver** at run time (`--gpus all` / `--nv`); the conda envs ship their own
CUDA userspace, so there is no system CUDA to match.

- **Docker image:** `ghcr.io/fpera0248/scfm-bias-asi2026:latest`
- **Size:** ~18–28 GB (four PyTorch/CUDA + R environments). Pull once.
- **Needs:** an NVIDIA GPU + driver on the host; `nvidia-container-toolkit` for Docker,
  or Apptainer with `--nv`. `scdesign3_env` (augmentation) is CPU-only.

## Pull and run

### Docker
```
docker pull ghcr.io/fpera0248/scfm-bias-asi2026:latest
docker run --gpus all -it \
    -v /path/to/your/data:/data \
    ghcr.io/fpera0248/scfm-bias-asi2026:latest
```

### Apptainer (HPC)
```
apptainer pull scfm.sif docker://ghcr.io/fpera0248/scfm-bias-asi2026:latest
apptainer run --nv -B /path/to/your/data:/data scfm.sif
```

Inside the container, each stage runs in its env:
```
conda run -n scfoundation_gpu python step2a_embed_scfoundation_ethnicity.py
conda run -n scdesign3_env    Rscript step0b_stage2_augment.R
```

## Build it yourself

```
# Docker
docker build -t ghcr.io/fpera0248/scfm-bias-asi2026:latest .
docker build --build-arg ... .            # (see Dockerfile for --full pinning note)

# Apptainer, from the built/published image
apptainer build scfm.sif Apptainer.def
```

`Dockerfile` runs `install.sh` (the four envs) then `fetch_weights.sh` (model code +
weights). To pin every transitive dependency to the original Linux/CUDA build, change
the `install.sh` call in the Dockerfile to `install.sh --full`.

## Model weights

- **scFoundation** — no action. `modelgenerator` pulls `genbio-ai/scFoundation`
  (`models.ckpt`, ~1.43 GB) from the HF Hub; the build pre-warms that cache.
- **Geneformer** — `fetch_weights.sh` clones `ctheodoris/Geneformer` @ `fcd26c4`
  (git-lfs) and editable-installs it; the 104M token dictionary ships in that repo.
- **scGPT** — the whole-human checkpoint (`best_model.pt`, `vocab.json`, `args.json`)
  is a separate download from the authors. Provide it one of two ways:
  - `--build-arg`/env **`SCGPT_CKPT_URL`** = a direct/Zenodo tarball URL (recommended
    for reproducibility — see Archival below), or
  - leave it unset and `fetch_weights.sh` uses `gdown` on the Google Drive link from
    the scGPT README. **Verify that link** — author-hosted links move.

## Bring your own data

Mount your data at `/data`. Each `.h5ad` needs (see README "Input format"): an
expression matrix in `X`, an `obs` cell-type column (`cell_type`), and the demographic
column for your axis (`self_reported_ethnicity`, `sex`, or the age field).

**Heads-up — hardcoded paths.** The step scripts hardcode absolute Oscar `BASE` paths
(555 files; see README "Running a workflow"). Two ways to deal with it:

1. **Symlink shim (turnkey, best-effort).** The image symlinks the original Oscar roots
   (`/oscar/data/rsingh47/fperalta`, `/oscar/home/fperalta`, `/users/fperalta`) to
   `/data`, so mounting your data at `/data` makes many hardcoded paths resolve as-is.
   Verify the `BASE` value in the script you run points under one of those roots.
2. **Edit `BASE`.** Open the step script and set `BASE` to `/data/...`. This release has
   no config-driven single-path option.

## Hosting

- **GHCR** (convenience): tag as `ghcr.io/fpera0248/scfm-bias-asi2026:<version>`,
  `docker push`. Public images are free and Apptainer pulls `docker://…` directly.
  Also tag an immutable `:asi2026` or `:v1.0` alongside `:latest`.
- **Zenodo** (archival, for the paper): `apptainer build scfm.sif …`, upload the `.sif`
  to Zenodo for a DOI so the exact container is citable and permanent. If you also
  archive the scGPT checkpoint there, point `SCGPT_CKPT_URL` at that record so the
  build no longer depends on the authors' Drive link.

## Notes / caveats

- First GPU run inside the container needs a host driver new enough for CUDA 11.8
  (driver ≥ 520). `--nv` / `--gpus all` injects it.
- scGPT runs without flash-attention (not a dependency of `scgpt==0.2.1`).
- `scdesign3_env` installs scDesign3 from GitHub `main` (~v1.5.0); pin a commit in
  `install.sh` if you need bit-exact R reproducibility.
