# Container: reproduce our results, turnkey

Everything runs from a prebuilt container — no environment setup, no path editing. Pull
an image and run one command per workflow:

```
reproduce <model> <cohort> <demographic>
```

`reproduce` downloads the cohort from CZ CELLxGENE, wires the baked model checkpoint into
the paths the scripts expect, and runs the full chain (extract → scDesign3 augment → embed
→ benchmark → all downstream metrics/figures), writing outputs under `/data`.

## Images

Each image bundles its model env **and** scDesign3 (so it runs a full workflow — including
step0b augmentation — on its own). Pick the model you want, or the all-in-one.

| Image | Contents | Use for |
|-------|----------|---------|
| `ghcr.io/fpera0248/scfm-scfoundation` | scFoundation + scDesign3 | `reproduce scfoundation …` |
| `ghcr.io/fpera0248/scfm-geneformer`   | Geneformer + scDesign3   | `reproduce geneformer …` |
| `ghcr.io/fpera0248/scfm-scgpt`        | scGPT + scDesign3        | `reproduce scgpt …` |
| `ghcr.io/fpera0248/scfm-all`          | all three + scDesign3    | any workflow (largest) |

## What you need

- **Docker** + **NVIDIA Container Toolkit**, or **Apptainer** on HPC.
- **An NVIDIA GPU + driver ≥ 520** (images are CUDA 11.8; the driver is the only host-side
  CUDA piece). Only the embedding step uses the GPU; augmentation + downstream are CPU.
- Free space for the image + the cohort download (AIDA is ~14 GB).

## Reproduce a workflow

```
docker pull ghcr.io/fpera0248/scfm-geneformer:latest
docker run --gpus all -v "$PWD/data":/data \
    ghcr.io/fpera0248/scfm-geneformer:latest \
    reproduce geneformer aida ethnicity
```
- `<model>` — `scfoundation` | `geneformer` | `scgpt`
- `<cohort>` — `ild` | `crc` | `aida`
- `<demographic>` — `ethnicity` | `sex` | `age`  *(AIDA & CRC are sex-balanced — no `sex`)*

On HPC with Apptainer (note `--writable-tmpfs`, explained below):
```
apptainer pull scfm-geneformer.sif docker://ghcr.io/fpera0248/scfm-geneformer:latest
apptainer run --nv --writable-tmpfs -B "$PWD/data":/data scfm-geneformer.sif reproduce geneformer aida ethnicity
```
`--writable-tmpfs` gives an ephemeral (RAM) overlay so the R↔AnnData bridge (basilisk) can
write its lockfile — the `.sif` is read-only and its baked conda env otherwise can't be
locked. Docker images are writable, so `docker run` needs no equivalent flag.

### Running on the right resource (`[stage]`)

`reproduce` takes an optional 4th argument so each phase runs where it belongs — only the
embedding needs a GPU:

| stage | does | resource |
|-------|------|----------|
| `prep` | fetch + step0a/step0c + step0b scDesign3 augmentation | CPU (long) |
| `embed` | step2a embedding | **GPU** |
| `down` | step3a…step9 downstream (iLISI, classification, figures) | CPU |
| `all` *(default)* | everything in one process | GPU box |

On a local GPU box just omit it (`reproduce scfoundation ild ethnicity` runs `all`). On a
cluster, submit `... prep` and `... down` to a CPU partition and `... embed` to a GPU
partition (chain them with `sbatch --dependency=afterok:`), so the GPU is held only for the
minutes of embedding, not the hours of augmentation.

The image is also robust to your shell: if you have conda/mamba active on the host, those
env vars no longer leak in and hijack the container's own conda — `reproduce` resets to the
image's `/opt/conda` itself, so no `--cleanenv` is needed.

## What's turnkey vs. what needs new code

**Verified turnkey — ethnicity across all three cohorts × all three models (36 conditions).**
`reproduce <model> <cohort> ethnicity` runs end to end for `ild` / `crc` / `aida` and each
reproduces the paper's iLISI (the four in-scope conditions: Proportional / BalancedAugmented /
BalancedUpsampled / Downsampled) with nothing to edit. This is the path that was validated
end-to-end. The stochastic BalancedAugmented condition is made exact by the shipped pilots in
`pilots/` (scDesign3 synthesis is not bit-reproducible — see the Notes below); every other
condition regenerates deterministically.

**Not yet verified turnkey — the `sex` / `age` demographics and the CRC *full-prep* path.**
The step scripts for these combinations exist, but they weren't validated in the container:
the `sex`/`age` workflows use a staged (multi-part) step0b the driver doesn't yet wire, and
CRC's `step0a` is a shared top-level extractor rather than a per-model one, so a default
full-cohort `reproduce … crc …` run reaches step0b without its RawCounts input. Running these
needs driver wiring plus a live prep run to confirm; treat them as "needs new code" for now.

**Downstream classification needs full data, not tiny pilots.** The iLISI/integration metrics
(the paper's headline result) run on the pilots and reproduce exactly. The *downstream*
learning-curve / robustness stages (`step4a`/`step4b`+) train per-group classifiers, which can
raise `ValueError: this solver needs samples of at least 2 classes` when a rebalanced pilot is
small enough that a group collapses to one class. That is post-iLISI and does not affect any
reported integration number — run those stages on the full cohort outputs, not the shipped
pilots.

**Needs new code — this is a reproduction harness for these 3 datasets and 3 models, not a
general plug-in framework:**
- A **new dataset** needs its own `step0a` extractor (its `obs` schema, raw-count location,
  and validation split are dataset-specific), a workflow, and a driver-manifest entry.
- A **new model** needs its own conda env and `step2a` embedding call (each model tokenizes
  and loads differently), plus checkpoint wiring.
- The **downstream** stages (iLISI, classification, learning curves) *are* model- and
  dataset-agnostic — they run on embeddings + labels — so those generalize. See the README's
  "Auditing a new foundation model".

---

*The sections below are for running stages by hand or bringing your own data — you don't
need them if you're using `reproduce`.*

## Bring your own data

Mount your data at `/data`, then set each step script's `BASE` variable to `/data`
(the scripts hardcode an absolute input path near the top — see README "Running a
workflow"). No cluster-specific paths are baked into the images.

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
- The `scgpt310` env pins `numpy==1.26.4`: its torch (2.1.2) was built against numpy 1.x,
  so a numpy 2.x would break `torch.from_numpy` with `_ARRAY_API not found`. The
  `geneformer310` env similarly pins `tokenizers<0.22` + `huggingface-hub<1.0` (transformers
  4.49 requires them). These pins live in `install.sh` and are baked into the images.
- scFoundation weights auto-download via `modelgenerator` (HF `genbio-ai/scFoundation`);
  the build pre-warms that cache.
- `scdesign3_env` installs scDesign3 pinned to the exact commit used for the paper
  (`SONGDONGYUAN1994/scDesign3@4370074c`, version 1.5.0). Version `1.5.0` spans many
  commits; the seeded scDesign3 augmentation only reproduces bit-for-bit at this SHA.

## Runtime gotchas (Apptainer)

- **Cache dirs are handled for you:** scientific tools (scanpy/numba, matplotlib) and the
  HuggingFace hub write caches that fail on the read-only image. `reproduce` already points
  `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, and `HF_HOME` at writable paths under `/data`, so imports
  work in any run mode (`--no-home`, `--containall`, no `--writable-tmpfs`) with nothing to
  set yourself. (`XDG_CACHE_HOME` is deliberately *not* redirected — it would send HF's cache
  off the baked scFoundation weights.) If you invoke a step script by hand outside `reproduce`,
  set those three yourself, or redirect a read-only/over-quota HOME with
  `--home /some/writable/dir:/root` (Apptainer refuses `--env HOME=...`).
- **scdesign3 / basilisk:** the R `zellkonverter` bridge uses a basilisk-managed conda env.
  The image bakes it at `/opt/basilisk` and `reproduce` already points `BASILISK_EXTERNAL_DIR`
  there — **do not set it yourself.** The only thing you must add is **`--writable-tmpfs`** on
  the `apptainer run` line, so basilisk can write its lockfile (the `.sif` is read-only, so
  it needs an ephemeral overlay; the env itself still comes from the baked, valid
  `/opt/basilisk` — no download, no internet, no rebuild). Pointing `BASILISK_EXTERNAL_DIR` at
  an empty scratch dir is the classic mistake: basilisk then can't find its env and, offline
  on a compute node, can't build one — the run dies with `use_condaenv … Unable to locate`.
- **step0b is CPU-only and long:** scDesign3 fits per-gene GAMs + a copula serially
  (`N_CORES=1`, deliberate for memory safety). On a full cohort it can run many hours and
  peak RAM is cohort-dependent: ~125 GB measured on ILD (and similar on CRC), while the
  largest cohort (AIDA, ~1.2 M cells) transiently peaks near 190 GB during the step0a
  raw-count extraction that precedes it. Budget ~128 GB for ILD/CRC and ~300 GB for AIDA.
  Run it on a CPU partition with a long walltime; reserve the GPU only for the
  embedding/downstream steps.
