#!/usr/bin/env bash
# Fetch model CODE and WEIGHTS into $MODELS_DIR.
# Run AFTER install.sh (the four conda envs must already exist).
#
#   scFoundation : no manual file — modelgenerator pulls genbio-ai/scFoundation
#                  from the HF Hub; we optionally pre-warm the cache here.
#   Geneformer   : clone the HF repo (git-lfs) at a pinned commit and editable-install it.
#   scGPT        : the whole-human checkpoint is a separate download from the authors.
#                  Set SCGPT_CKPT_URL to a direct/Zenodo URL, or the script uses gdown
#                  with the Google Drive link from the scGPT README (verify it — author
#                  links move). See CONTAINER.md.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/models}"
mkdir -p "$MODELS_DIR"

# --- Geneformer: HF repo @ pinned commit, editable install into geneformer310 ---
GF_DIR="$MODELS_DIR/Geneformer"
GF_COMMIT="fcd26c4"
echo "=== Geneformer: $GF_DIR @ $GF_COMMIT ==="
if [ ! -d "$GF_DIR/.git" ]; then
  git lfs install
  git clone https://huggingface.co/ctheodoris/Geneformer "$GF_DIR"
fi
git -C "$GF_DIR" fetch --all --quiet || true
git -C "$GF_DIR" checkout "$GF_COMMIT"
conda run -n geneformer310 pip install -e "$GF_DIR"
echo "  token dictionary: $GF_DIR/geneformer/token_dictionary_gc104M.pkl"

# --- scFoundation: pre-warm HF Hub cache (modelgenerator fetches at runtime otherwise) ---
echo "=== scFoundation: pre-warming genbio-ai/scFoundation cache ==="
conda run -n scfoundation_gpu python - <<'PY' || echo "  WARN: pre-warm failed; modelgenerator will download at first run."
from huggingface_hub import snapshot_download
p = snapshot_download("genbio-ai/scFoundation",
                      revision="cb434153a1acfacd215eefc956ea445f7cc39c3")
print(f"  cached at {p}")
PY

# --- scGPT: whole-human checkpoint (best_model.pt, vocab.json, args.json) ---
SCGPT_DIR="$MODELS_DIR/scGPT_human"
echo "=== scGPT: $SCGPT_DIR ==="
if [ -f "$SCGPT_DIR/best_model.pt" ]; then
  echo "  already present; skipping."
elif [ -n "${SCGPT_CKPT_URL:-}" ]; then
  mkdir -p "$SCGPT_DIR"
  # Expect an archive or direct files at SCGPT_CKPT_URL (e.g. a Zenodo record).
  curl -fL --retry 3 -o "$SCGPT_DIR/scGPT_human.tar.gz" "$SCGPT_CKPT_URL"
  tar -xzf "$SCGPT_DIR/scGPT_human.tar.gz" -C "$SCGPT_DIR" --strip-components=1
  rm -f "$SCGPT_DIR/scGPT_human.tar.gz"
elif [ -n "${SCGPT_GDRIVE:-}" ]; then
  # SCGPT_GDRIVE = the whole-human Google Drive folder link from the scGPT README.
  mkdir -p "$SCGPT_DIR"
  conda run -n scgpt310 pip install --quiet gdown
  echo "  downloading whole-human checkpoint via gdown from $SCGPT_GDRIVE"
  conda run -n scgpt310 gdown --folder "$SCGPT_GDRIVE" -O "$SCGPT_DIR" \
    || echo "  WARN: gdown failed. Download scGPT_human manually into $SCGPT_DIR."
else
  echo "  SKIP: no scGPT checkpoint source set."
  echo "        Set SCGPT_CKPT_URL (a direct/Zenodo tarball) or SCGPT_GDRIVE (the"
  echo "        whole-human Google Drive folder from the scGPT README), then re-run."
  echo "        Expected files in $SCGPT_DIR: best_model.pt, vocab.json, args.json"
fi

echo
echo "Model code + weights staged under $MODELS_DIR:"
echo "  Geneformer  -> $GF_DIR (editable-installed into geneformer310)"
echo "  scFoundation-> HF Hub cache (auto via modelgenerator)"
echo "  scGPT       -> $SCGPT_DIR"
echo "Point the step scripts' model/BASE paths at these locations (see CONTAINER.md)."
