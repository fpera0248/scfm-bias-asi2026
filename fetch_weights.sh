#!/usr/bin/env bash
# Fetch model CODE and WEIGHTS into $MODELS_DIR. Run AFTER install.sh.
#
# Usage:
#   bash fetch_weights.sh                 # all models
#   bash fetch_weights.sh geneformer      # only this model (for per-model images)
#   bash fetch_weights.sh scgpt scfoundation
#
#   scFoundation : modelgenerator pulls genbio-ai/scFoundation from HF Hub; pre-warm here.
#   Geneformer   : clone HF repo (git-lfs) at a pinned commit + editable install.
#   scGPT        : whole-human checkpoint. Set SCGPT_CKPT_URL to a direct/GitHub-release
#                  tarball (recommended), or SCGPT_GDRIVE to the scGPT README Drive folder.
set -euo pipefail

MODELS=("$@")
[ "${#MODELS[@]}" -eq 0 ] && MODELS=(scfoundation geneformer scgpt)
want () { for m in "${MODELS[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

MODELS_DIR="${MODELS_DIR:-/opt/models}"
mkdir -p "$MODELS_DIR"

# --- Geneformer: HF repo @ pinned commit, editable install into geneformer310 ---
if want geneformer; then
  GF_DIR="$MODELS_DIR/Geneformer"; GF_COMMIT="fcd26c4"
  echo "=== Geneformer: $GF_DIR @ $GF_COMMIT ==="
  if [ ! -d "$GF_DIR/.git" ]; then
    git lfs install
    git clone https://huggingface.co/ctheodoris/Geneformer "$GF_DIR"
  fi
  git -C "$GF_DIR" fetch --all --quiet || true
  git -C "$GF_DIR" checkout "$GF_COMMIT"
  # --no-deps: geneformer's deps are already installed at their pinned versions.
  conda run -n geneformer310 pip install --no-deps -e "$GF_DIR"
  echo "  token dictionary: $GF_DIR/geneformer/token_dictionary_gc104M.pkl"
fi

# --- scFoundation: pre-warm HF Hub cache ---
if want scfoundation; then
  echo "=== scFoundation: pre-warming genbio-ai/scFoundation cache ==="
  conda run -n scfoundation_gpu python - <<'PY' || echo "  WARN: pre-warm failed; modelgenerator will fetch at first run."
from huggingface_hub import snapshot_download
p = snapshot_download("genbio-ai/scFoundation",
                      revision="cb434153a1acfacd215eefc956ea445f7cc39c3")
print(f"  cached at {p}")
PY
fi

# --- scGPT: whole-human checkpoint ---
if want scgpt; then
  SCGPT_DIR="$MODELS_DIR/scGPT_human"
  echo "=== scGPT: $SCGPT_DIR ==="
  if [ -f "$SCGPT_DIR/best_model.pt" ]; then
    echo "  already present; skipping."
  elif [ -n "${SCGPT_CKPT_URL:-}" ]; then
    mkdir -p "$SCGPT_DIR"
    curl -fL --retry 3 -o "$SCGPT_DIR/scGPT_human.tar.gz" "$SCGPT_CKPT_URL"
    tar -xzf "$SCGPT_DIR/scGPT_human.tar.gz" -C "$SCGPT_DIR" --strip-components=1
    rm -f "$SCGPT_DIR/scGPT_human.tar.gz"
  elif [ -n "${SCGPT_GDRIVE:-}" ]; then
    mkdir -p "$SCGPT_DIR"
    conda run -n scgpt310 pip install --quiet gdown
    conda run -n scgpt310 gdown --folder "$SCGPT_GDRIVE" -O "$SCGPT_DIR" \
      || echo "  WARN: gdown failed. Download scGPT_human manually into $SCGPT_DIR."
  else
    echo "  SKIP: set SCGPT_CKPT_URL (direct/GitHub-release tarball) or SCGPT_GDRIVE."
    echo "        Expected files in $SCGPT_DIR: best_model.pt, vocab.json, args.json"
  fi
fi

echo
echo "Weights staged under $MODELS_DIR for: ${MODELS[*]}"
