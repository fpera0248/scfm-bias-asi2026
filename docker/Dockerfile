# syntax=docker/dockerfile:1
#
# Per-model image (one conda env + the pipeline code + that model's weights).
# Built by .github/workflows/build-images.yml, one image per model, pushed to GHCR.
# GPU comes from the host driver at run time (--gpus all / --nv); the env ships its
# own CUDA userspace, so no system CUDA toolkit is baked.
#
#   docker build -f docker/Dockerfile \
#       --build-arg MODEL=scgpt --build-arg ENV_NAME=scgpt310 \
#       -t ghcr.io/fpera0248/scfm-scgpt:latest .

FROM condaforge/miniforge3:24.11.3-2
# MODEL: scfoundation | geneformer | scgpt | scdesign3   (drives fetch_weights.sh)
# ENV_NAME: matching conda env name                       (drives install.sh)
ARG MODEL
ARG ENV_NAME
ARG SCGPT_CKPT_URL=""

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_DIR=/opt/models \
    SCFM_HOME=/opt/scfm \
    SCGPT_CKPT_URL=${SCGPT_CKPT_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scfm
COPY . /opt/scfm

# Build just this model's env (from the fully pinned spec), then fetch just its weights.
RUN bash install.sh --full "${ENV_NAME}"
RUN bash fetch_weights.sh "${MODEL}" || echo "WARN: weights step incomplete for ${MODEL}; see CONTAINER.md."

# Mount your data at /data and point each step script's BASE at /data
# (the scripts hardcode an absolute input path — see README "Running a workflow").
# No cluster-specific paths are baked into the image.
RUN mkdir -p /data

LABEL org.opencontainers.image.source="https://github.com/fpera0248/scfm-bias-asi2026" \
      description="Reproducible env (${MODEL}) for Demographic Bias in Single-Cell Foundation Models (ACM-BCB ASI 2026)"

CMD ["/bin/bash"]
