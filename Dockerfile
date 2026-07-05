# syntax=docker/dockerfile:1
#
# Reproducible environment for "Demographic Bias in Single-Cell Foundation Models".
# Bakes all four conda envs + model code so a user can bring their own data and run
# any workflow. GPU is provided by the HOST driver at run time (the conda envs ship
# their own CUDA userspace), so no system CUDA toolkit is baked here.
#
#   Build:  docker build -t ghcr.io/fpera0248/scfm-bias-asi2026:latest .
#   Run:    docker run --gpus all -it \
#             -v /path/to/your/data:/data \
#             ghcr.io/fpera0248/scfm-bias-asi2026:latest
#
# See CONTAINER.md for Apptainer/HPC, weights, and bring-your-own-data details.

FROM condaforge/miniforge3:24.11.3-2

# Let the NVIDIA container runtime (Docker --gpus / Apptainer --nv) inject the host driver.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_DIR=/opt/models \
    SCFM_HOME=/opt/scfm

# System deps: git-lfs (Geneformer HF repo), build tools (R/pip source installs), curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scfm
COPY . /opt/scfm

# Heavy layer 1: build all four conda envs from the slim specs (mamba solver).
# Use --full to pin every transitive dep to the original Linux/CUDA build instead.
RUN bash install.sh

# Heavy layer 2: model code + weights (Geneformer clone+install, scFoundation cache,
# scGPT checkpoint). scGPT weights need SCGPT_CKPT_URL or a reachable Drive link;
# if unset the build still succeeds and scGPT weights can be added at run time.
RUN bash fetch_weights.sh || echo "WARN: fetch_weights.sh incomplete; see CONTAINER.md."

# Mount your data at /data and set each step script's BASE to /data
# (the scripts hardcode an absolute input path — see README "Running a workflow").
# No cluster-specific paths are baked into the image.
RUN mkdir -p /data

CMD ["/bin/bash"]
