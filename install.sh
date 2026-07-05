#!/usr/bin/env bash
# One-pass build of the four conda environments this repo uses.
# See ENVIRONMENTS.md for the manual, step-by-step version, prerequisites,
# and troubleshooting.
#
# Usage:
#   bash install.sh          # build from the slim specs (environment_<name>.yml)
#   bash install.sh --full   # build from the fully pinned specs (environment_<name>.full.yml)
set -euo pipefail

SUFFIX=""
if [ "${1:-}" = "--full" ]; then
  SUFFIX=".full"
elif [ -n "${1:-}" ]; then
  echo "Unknown argument: $1 (use --full or no argument)" >&2
  exit 2
fi

# Prefer mamba (much faster solver) if available, else conda.
if command -v mamba >/dev/null 2>&1; then
  SOLVER=mamba
elif command -v conda >/dev/null 2>&1; then
  SOLVER=conda
else
  echo "ERROR: need conda or mamba on PATH. See ENVIRONMENTS.md (Prerequisites)." >&2
  exit 1
fi

env_exists () { conda env list | awk '{print $1}' | grep -qx "$1"; }

create_env () {
  local name="$1" spec="environment_$1${SUFFIX}.yml"
  echo "=== env: $name  (spec: $spec) ==="
  if [ ! -f "$spec" ]; then
    echo "ERROR: spec not found: $spec" >&2
    exit 1
  fi
  if env_exists "$name"; then
    echo "  '$name' already exists; skipping. Remove it first (conda env remove -n $name) to rebuild."
  else
    "$SOLVER" env create -f "$spec"
  fi
}

create_env scfoundation_gpu
create_env geneformer310
create_env scgpt310
create_env scdesign3_env

# Extra step 1: scGPT is not on conda — install it into scgpt310 from pip.
echo "=== installing scgpt==0.2.1 into scgpt310 ==="
conda run -n scgpt310 pip install "scgpt==0.2.1"

# Extra step 2: scDesign3 is not on conda — install it into scdesign3_env from source.
echo "=== installing scDesign3 (1.5.0) into scdesign3_env ==="
conda run -n scdesign3_env Rscript -e \
  'if (!requireNamespace("scDesign3", quietly=TRUE)) devtools::install_github("SONGDONGYUAN1994/scDesign3")'

echo
echo "Done. All four environments are built."
echo "Activate one with, e.g.:  conda activate scfoundation_gpu"
echo "Model weights are separate downloads — see the Model setup section of README.md."
