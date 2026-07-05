#!/usr/bin/env bash
# Build the conda environments this repo uses.
# See ENVIRONMENTS.md for prerequisites and the manual, step-by-step version.
#
# Usage:
#   bash install.sh                       # build all four envs (slim specs)
#   bash install.sh --full                # all four, fully pinned specs
#   bash install.sh scgpt310              # build only this env (for per-model images)
#   bash install.sh --full geneformer310  # one env, fully pinned
set -euo pipefail

SUFFIX=""
ENVS=()
for arg in "$@"; do
  case "$arg" in
    --full) SUFFIX=".full" ;;
    -*) echo "Unknown flag: $arg (use --full)" >&2; exit 2 ;;
    *) ENVS+=("$arg") ;;
  esac
done
# Default: all four.
if [ "${#ENVS[@]}" -eq 0 ]; then
  ENVS=(scfoundation_gpu geneformer310 scgpt310 scdesign3_env)
fi

# Prefer mamba (faster solver) if available, else conda.
if command -v mamba >/dev/null 2>&1; then SOLVER=mamba
elif command -v conda >/dev/null 2>&1; then SOLVER=conda
else echo "ERROR: need conda or mamba on PATH. See ENVIRONMENTS.md." >&2; exit 1; fi

want () { for e in "${ENVS[@]}"; do [ "$e" = "$1" ] && return 0; done; return 1; }
env_exists () { conda env list | awk '{print $1}' | grep -qx "$1"; }

create_env () {
  local name="$1" spec="environment_$1${SUFFIX}.yml" build_spec
  echo "=== env: $name  (spec: $spec) ==="
  [ -f "$spec" ] || { echo "ERROR: spec not found: $spec" >&2; exit 1; }
  if env_exists "$name"; then
    echo "  '$name' already exists; skipping. Remove it (conda env remove -n $name) to rebuild."
    return
  fi
  build_spec="$spec"
  if [ "$SUFFIX" = ".full" ]; then
    # The .full specs come from `conda env export`, which isn't directly replayable:
    #   - the pip section pins torch==...+cu118 wheels hosted on PyTorch's index, and
    #   - geneformer is a source/editable install (fetch_weights.sh), not on PyPI.
    # Sanitize into a temp spec: add PyTorch's extra index, drop the geneformer self-pin.
    build_spec="/tmp/spec_${name}.yml"
    awk '
      /^[[:space:]]*-[[:space:]]*geneformer(==| @)/ { next }
      { print }
      /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ {
        print "      - --extra-index-url https://download.pytorch.org/whl/cu118"
      }
    ' "$spec" > "$build_spec"
  fi
  "$SOLVER" env create -f "$build_spec"
}

for e in "${ENVS[@]}"; do create_env "$e"; done

# Extra step 1: scGPT is not on conda — install it into scgpt310 from pip.
if want scgpt310; then
  echo "=== installing scgpt==0.2.1 into scgpt310 ==="
  conda run -n scgpt310 pip install "scgpt==0.2.1"
fi

# Extra step 2: scDesign3 is not on conda — install it into scdesign3_env from source.
if want scdesign3_env; then
  echo "=== installing scDesign3 (1.5.0) into scdesign3_env ==="
  conda run -n scdesign3_env Rscript -e \
    'if (!requireNamespace("scDesign3", quietly=TRUE)) devtools::install_github("SONGDONGYUAN1994/scDesign3")'
fi

echo
echo "Done. Built: ${ENVS[*]}"
echo "Activate one with, e.g.:  conda activate ${ENVS[0]}"
