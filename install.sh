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
  if [ "$SUFFIX" != ".full" ]; then
    "$SOLVER" env create -f "$spec"
    return
  fi

  # --- Robust replay of a `conda env export` (.full spec) ---
  # The pip section is a complete, flattened, exact-pinned list, but its pins are
  # internally inconsistent (the env was built up incrementally on the cluster with a
  # permissive resolver), and pip 25.3 has no legacy resolver to force an inconsistent
  # set. So: build the conda layer, then pip-install the exact pinned set with --no-deps
  # (install every version as-is, no re-resolution -> no conflicts). torch==...+cu118
  # wheels come from PyTorch's index; geneformer is installed from source separately
  # (fetch_weights.sh), so drop its self-reference here.
  local conda_yml="/tmp/conda_${name}.yml" pip_reqs="/tmp/pip_${name}.txt"
  awk '/^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ { exit } { print }' "$spec" > "$conda_yml"
  # (|| true: an env with no pip section yields empty input, and grep -v exits 1 on that,
  #  which would otherwise trip `set -e -o pipefail`.)
  awk '/^[[:space:]]*-[[:space:]]*pip:/ { p=1; next }
       p && /^[[:space:]]+-[[:space:]]/ { sub(/^[[:space:]]+-[[:space:]]*/, ""); print }' "$spec" \
     | { grep -vE '^geneformer(==| @)' || true; } > "$pip_reqs"
  "$SOLVER" env create -f "$conda_yml"
  if [ -s "$pip_reqs" ]; then
    conda run -n "$name" pip install --no-deps \
      --extra-index-url https://download.pytorch.org/whl/cu118 -r "$pip_reqs"
  fi
}

for e in "${ENVS[@]}"; do create_env "$e"; done

# Extra step 1: scGPT is not on conda — install it into scgpt310 from pip.
# (Skipped for --full builds: scgpt is already in the pinned pip set installed above.)
if want scgpt310 && [ "$SUFFIX" != ".full" ]; then
  echo "=== installing scgpt==0.2.1 into scgpt310 ==="
  conda run -n scgpt310 pip install "scgpt==0.2.1"
fi

# Extra step 2: scDesign3 (source) + zellkonverter (Bioconductor) into scdesign3_env.
# zellkonverter provides the R<->AnnData (.h5ad) bridge that step0b uses; it relies on a
# basilisk-managed Python env, which we pre-create here so it's baked into the image and
# found at runtime via BASILISK_EXTERNAL_DIR (set in the Dockerfile). Without this the
# container silently depended on the user's personal ~/.conda env — not reproducible.
if want scdesign3_env; then
  echo "=== installing scDesign3 + zellkonverter into scdesign3_env ==="
  : "${BASILISK_EXTERNAL_DIR:=/opt/basilisk}"; export BASILISK_EXTERNAL_DIR
  conda run -n scdesign3_env Rscript -e '
    if (!requireNamespace("BiocManager", quietly=TRUE))
      install.packages("BiocManager", repos="https://cloud.r-project.org")
    # Pinned to the exact commit used for the paper (version 1.5.0 spans many
    # commits; the seeded augmentation is only bit-reproducible at this SHA).
    if (!requireNamespace("scDesign3", quietly=TRUE))
      devtools::install_github("SONGDONGYUAN1994/scDesign3@4370074cc5392ddd7821e66e1e1c1d1181f21d3d")
    bioc <- c("zellkonverter", "scran", "scuttle", "SingleCellExperiment", "BiocParallel")
    need <- bioc[!vapply(bioc, requireNamespace, logical(1), quietly=TRUE)]
    if (length(need)) BiocManager::install(need, update=FALSE, ask=FALSE)
    # Pre-build the basilisk Python env so it is cached inside the image.
    library(basilisk); library(zellkonverter)
    cl <- basiliskStart(zellkonverterAnnDataEnv()); basiliskStop(cl)
    cat("scDesign3 + zellkonverter + basilisk env ready\n")
  '
fi

echo
echo "Done. Built: ${ENVS[*]}"
echo "Activate one with, e.g.:  conda activate ${ENVS[0]}"
