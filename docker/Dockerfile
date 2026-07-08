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
    BASILISK_EXTERNAL_DIR=/opt/basilisk \
    SCGPT_CKPT_URL=${SCGPT_CKPT_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scfm
COPY . /opt/scfm

# Build this model's env PLUS scdesign3_env (step0b augmentation) so the image can run
# a full workflow start-to-finish on its own, then fetch this model's weights.
RUN bash install.sh --full "${ENV_NAME}" scdesign3_env
RUN bash fetch_weights.sh "${MODEL}" || echo "WARN: weights step incomplete for ${MODEL}; see CONTAINER.md."

# Turnkey entrypoint: `reproduce <model> <cohort> <demographic>` on PATH.
# The driver wires baked weights to the /data paths scripts expect and runs the chain.
RUN chmod +x /opt/scfm/reproduce.sh && ln -sf /opt/scfm/reproduce.sh /usr/local/bin/reproduce
RUN mkdir -p /data

LABEL org.opencontainers.image.source="https://github.com/fpera0248/scfm-bias-asi2026" \
      description="Reproducible env (${MODEL} + scdesign3) for Demographic Bias in Single-Cell Foundation Models (ACM-BCB ASI 2026)"

CMD ["/bin/bash"]
