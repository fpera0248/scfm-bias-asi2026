# Container: reproduce on your own GPU

The models are published as **four public Docker images, one per model**. Anyone with
their own NVIDIA GPU can pull the one they want, mount their own `.h5ad` data, and run —
no build step, no access to any HPC cluster.

| Image | Model | Approx size |
|-------|-------|-------------|
| `ghcr.io/fpera0248/scfm-scfoundation` | scFoundation embedding | ~7 GB |
| `ghcr.io/fpera0248/scfm-geneformer`   | Geneformer embedding   | ~7 GB |
| `ghcr.io/fpera0248/scfm-scgpt`        | scGPT embedding        | ~6 GB |
| `ghcr.io/fpera0248/scfm-scdesign3`    | scDesign3 augmentation (CPU) | ~3 GB |

## What a user needs

- **Docker** + **NVIDIA Container Toolkit** (`nvidia-container-toolkit`) on their machine.
- **An NVIDIA GPU + driver ≥ 520** (images are CUDA 11.8; the driver is the only host-side
  CUDA piece). `scfm-scdesign3` is CPU-only and needs no GPU.
- Their own `.h5ad` (see README "Input format").

## Run one

```
docker pull ghcr.io/fpera0248/scfm-scgpt:latest
docker run --gpus all -it -v /path/to/your/data:/data \
    ghcr.io/fpera0248/scfm-scgpt:latest

# inside the container, run a stage in its env:
conda run -n scgpt310 python step2a_embed_scgpt_ethnicity.py
```

On HPC (no Docker) the same images run under Apptainer:
```
apptainer pull scfm-scgpt.sif docker://ghcr.io/fpera0248/scfm-scgpt:latest
apptainer run --nv -B /path/to/your/data:/data scfm-scgpt.sif
```

## Bring your own data

Mount your data at `/data`. The images symlink the scripts' hardcoded Oscar roots
(`/oscar/data/rsingh47/fperalta`, `/oscar/home/fperalta`, `/users/fperalta`) to `/data`,
so many hardcoded `BASE` paths resolve automatically. If a script's `BASE` points
elsewhere, edit it to `/data/...` (see README "Running a workflow").

---

# Publishing (maintainer)

The images are built and pushed **automatically by GitHub Actions** — no build machine,
no GPU, and Oscar is never involved. Building an image needs only CPU + internet + disk,
all of which the GitHub runner has.

**One-time setup:**

1. **scGPT weights → a public link.** Geneformer and scFoundation weights already live on
   public servers (HF Hub); only scGPT's checkpoint doesn't. Tar the `scGPT_human/` you
   have (`best_model.pt`, `vocab.json`, `args.json`) and attach it to a **GitHub Release**
   in this repo:
   ```
   tar -czf scGPT_human.tar.gz -C /path/to scGPT_human
   # then drag scGPT_human.tar.gz onto a GitHub Release (repo → Releases → draft)
   ```
   Copy the asset's download URL and set it as an **Actions variable**:
   repo → Settings → Secrets and variables → Actions → Variables → New →
   `SCGPT_CKPT_URL` = that URL.

2. **Enable GHCR.** The workflow already has `packages: write`. After the first run, open
   each package (repo → Packages) and set its visibility to **Public** so anyone can pull.

**Publish / update:** cut a release (or run the `build-images` workflow manually via
"Run workflow"). GitHub builds all four images and pushes `:latest` and `:<release-tag>`
to GHCR. Done.

**Smoke test before announcing:** on any GPU box (or Oscar via `apptainer pull`), run one
model's `step2a` on a few cells to confirm GPU + weights + env all work end to end.

---

# All-in-one image (optional, for local/HPC single-image use)

The root `Dockerfile` and `Apptainer.def` build a single image with **all four** envs
(~18–28 GB). Useful if you want everything in one artifact for your own HPC runs, but it's
large for public distribution — that's why the public images are split per model.

```
docker build -t scfm-all .
bash install.sh --full        # inside: build all four, fully pinned
```

# Notes

- scGPT runs without flash-attention (not a dependency of `scgpt==0.2.1`).
- scFoundation weights auto-download via `modelgenerator` (HF `genbio-ai/scFoundation`);
  the build pre-warms that cache.
- `scdesign3_env` installs scDesign3 from GitHub `main` (~v1.5.0); pin a commit in
  `install.sh` for bit-exact R reproducibility.
