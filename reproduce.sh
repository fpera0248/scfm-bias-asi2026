#!/usr/bin/env bash
# Turnkey reproduction driver.
#
#   reproduce.sh <model> <cohort> <demographic> [stage]
#     model       : scfoundation | geneformer | scgpt
#     cohort      : ild | crc | aida
#     demographic : ethnicity | sex | age
#     stage       : all (default) | prep | embed | down
#
# Runs one workflow end to end inside the container: fetches the raw data, wires up
# the model checkpoint + working paths so the (path-hardcoded) step scripts run
# unmodified, then executes step0a -> step0b (scDesign3, CPU) -> step2a (embed, GPU)
# -> step3a..step9 (downstream, CPU). Outputs land under $DATA_ROOT.
#
# The optional stage lets each phase run on the right resource (GPU only for embed):
# on HPC, submit `... prep` to a CPU partition and `... embed` to a GPU partition.
# Omit it (or pass `all`) to run everything in one process, e.g. for local Docker.
#
# Env knobs:
#   DATA_ROOT   where data/outputs live and get bind-mounted as /data (default /data)
#   SCFM_HOME   where the repo lives in the image (default /opt/scfm)
#   MODELS_DIR  where baked weights live (default /opt/models)
set -euo pipefail

# --- conda hygiene ----------------------------------------------------------
# Apptainer/Docker forward the caller's shell environment into the container by
# default. A user who has conda or mamba active on the host thus leaks CONDA_EXE,
# MAMBA_ROOT_PREFIX, CONDA_PREFIX, CONDA_SHLVL, ... into the image, where they
# hijack `conda run` and send it to a host path that does not exist inside the
# container (breaking every step). Drop that inherited state and pin to the
# image's own conda so `reproduce` works from any shell, with no --cleanenv needed.
unset CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV \
      CONDA_PROMPT_MODIFIER MAMBA_ROOT_PREFIX MAMBA_EXE _CE_CONDA _CE_M 2>/dev/null || true
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  export PATH="/opt/conda/bin:$PATH"
  . /opt/conda/etc/profile.d/conda.sh
fi

# --- basilisk / zellkonverter -----------------------------------------------
# step0b reads .h5ad files into R via zellkonverter, which runs a private conda env
# managed by basilisk. The image bakes that env at /opt/basilisk, so default there and
# the reviewer never has to provision or point at it. NOTE: /opt is read-only in a .sif,
# so basilisk needs somewhere writable for its lockfile — run the container with
# `apptainer run --writable-tmpfs ...` (an ephemeral overlay; see CONTAINER.md).
export BASILISK_EXTERNAL_DIR="${BASILISK_EXTERNAL_DIR:-/opt/basilisk}"

MODEL="${1:?usage: reproduce.sh <model> <cohort> <demographic> [stage]}"
COHORT="${2:?usage: reproduce.sh <model> <cohort> <demographic> [stage]}"
DEMO="${3:?usage: reproduce.sh <model> <cohort> <demographic> [stage]}"
# Optional stage so phases can run on the right partition (GPU only where needed):
#   prep  = fetch + step0a/step0c/step0b augmentation   (CPU, long)
#   embed = step2a                                       (GPU)
#   down  = step3a..step9 downstream                     (CPU)
#   all   = everything in one process (default)
STAGE="${4:-all}"
case "$STAGE" in all|prep|embed|down) ;; *) echo "unknown stage '$STAGE' (all|prep|embed|down)" >&2; exit 1 ;; esac
DATA_ROOT="${DATA_ROOT:-/data}"
SCFM_HOME="${SCFM_HOME:-/opt/scfm}"
MODELS_DIR="${MODELS_DIR:-/opt/models}"

say(){ echo; echo "########## $* ##########"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
want(){ [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# ---- 1. model -> conda env + repo dir name + checkpoint wiring -----------------
case "$MODEL" in
  scfoundation) ENV=scfoundation_gpu; MDIR=scfoundation ;;
  geneformer)   ENV=geneformer310;    MDIR=Geneformer ;;
  scgpt)        ENV=scgpt310;         MDIR=scGPT ;;
  *) die "unknown model '$MODEL' (scfoundation|geneformer|scgpt)" ;;
esac
conda env list | awk '{print $1}' | grep -qx "$ENV" || die "env '$ENV' not in this image (is this the right per-model image?)"
conda env list | awk '{print $1}' | grep -qx scdesign3_env || die "scdesign3_env not in this image; cannot run step0b augmentation"

# ---- 2. cohort -> workflow dir + raw data (URL + the name step0a expects) ------
case "$COHORT" in
  ild)
    case "$MODEL" in
      scfoundation) WFREL="scfoundation/augmentedv4/${DEMO}_scfoundation_workflow" ;;
      geneformer)   WFREL="Geneformer/augmented/${DEMO}_Geneformer_workflow" ;;
      scgpt)        WFREL="scGPT/${DEMO}_scGPT_workflow" ;;
    esac
    DATA_URL="https://datasets.cellxgene.cziscience.com/c3d9262e-0dc5-4eca-bf20-56e6d96d0306.h5ad"
    DATA_NAME="InterstitialLungDisease.h5ad" ;;
  crc)
    WFREL="${MDIR}/augmented_CRC/${DEMO}_${MODEL}_workflow"
    DATA_URL="https://datasets.cellxgene.cziscience.com/66cadf3b-4c71-4930-8add-fa748745704d.h5ad"
    DATA_NAME="ColorectalCancer_Epithelial.h5ad" ;;
  aida)
    WFREL="${MDIR}/augmented_AIDA/${DEMO}_${MODEL}_workflow"
    DATA_URL="https://datasets.cellxgene.cziscience.com/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"
    DATA_NAME="AIDA_phase1_v2.h5ad" ;;
  *) die "unknown cohort '$COHORT' (ild|crc|aida)" ;;
esac

SRC_WF="$SCFM_HOME/$WFREL"           # scripts (baked, read-only)
[ -d "$SRC_WF" ] || die "workflow not found for $MODEL/$COHORT/$DEMO: $WFREL"
WORK="$DATA_ROOT/$WFREL"             # data + outputs (writable, == /data/... the scripts hardcode)
mkdir -p "$WORK"

# The step0a (RawCounts) and step0c (seeded external-validation split) producers are
# model-agnostic dataset-prep and, for ILD/AIDA, live only in the Geneformer workflow.
# A model workflow that lacks its own step0*.py consumes those outputs — so runstep
# falls back to $PREP to find/run the canonical producer (writing into $WORK).
case "$COHORT" in
  ild)  PREPREL="Geneformer/augmented/${DEMO}_Geneformer_workflow" ;;
  aida) PREPREL="Geneformer/augmented_AIDA/${DEMO}_Geneformer_workflow" ;;
  crc)  PREPREL="" ;;   # CRC: the shared step0a extractor writes its own validation split
esac
PREP=""; [ -n "$PREPREL" ] && [ -d "$SCFM_HOME/$PREPREL" ] && PREP="$SCFM_HOME/$PREPREL"

say "REPRODUCE  $MODEL / $COHORT / $DEMO   ($(date +%T))"
echo "  workflow : $WFREL"
echo "  env      : $ENV (+ scdesign3_env for step0b)"
echo "  work dir : $WORK"

# ---- 3. wire baked checkpoints to the /data paths the scripts hardcode ----------
case "$MODEL" in
  geneformer)
    mkdir -p "$DATA_ROOT/Geneformer"
    [ -e "$DATA_ROOT/Geneformer/geneformer_repo" ] || ln -s "$MODELS_DIR/Geneformer" "$DATA_ROOT/Geneformer/geneformer_repo" ;;
  scgpt)
    mkdir -p "$DATA_ROOT/scGPT"
    [ -e "$DATA_ROOT/scGPT/scGPT_human" ] || ln -s "$MODELS_DIR/scGPT_human" "$DATA_ROOT/scGPT/scGPT_human" ;;
  scfoundation) : ;;  # loads via modelgenerator from the baked HF cache; no path needed
esac

# ---- 4. fetch the raw object (named as step0a expects) -------------------------
if [ ! -f "$WORK/$DATA_NAME" ]; then
  say "FETCH  $DATA_NAME"
  curl -fL --retry 3 -o "$WORK/$DATA_NAME" "$DATA_URL"
fi

# ---- 5. run the steps ----------------------------------------------------------
# runstep <label> <env> <interp: python|Rscript> <glob> [gpu]
runstep(){
  local label="$1" env="$2" interp="$3" glob="$4" gpu="${5:-0}"
  local script; script="$(ls "$SRC_WF"/$glob 2>/dev/null | sort | head -1 || true)"
  if [ -z "$script" ]; then echo ">>> $label: no script ($glob) — skipping"; return 0; fi
  say "$label  ($(basename "$script"), $(date +%T))"
  ( cd "$WORK" && conda run -n "$env" "$interp" "$script" ) \
    && echo ">>> $label OK" || die "$label FAILED"
}

if want prep; then
# ---- step0 prep: the two model-agnostic dataset artifacts step0b needs ----
# step0a -> RawCounts;  step0c -> seeded (RANDOM_STATE=42) external-validation split.
# step0c hardcodes an absolute BASE (== $DATA_ROOT/$PREPREL), and step0a is cwd-relative,
# so the canonical producers must run in $PREPWORK. We then stage both artifacts into
# $WORK, where step0b reads them by bare (cwd-relative) name. Cohorts whose model
# workflow carries its own step0a (no $PREP) run it in place as before.
if [ -n "$PREP" ]; then
  PREPWORK="$DATA_ROOT/$PREPREL"
  mkdir -p "$PREPWORK"
  ln -sf "$WORK/$DATA_NAME" "$PREPWORK/$DATA_NAME"      # share the fetched raw object
  p0a="$(ls "$PREP"/step0a*extract_raw_counts*.py 2>/dev/null | head -1 || true)"
  p0c="$(ls "$PREP"/step0c*external_validation*.py   2>/dev/null | head -1 || true)"
  if [ -n "$p0a" ]; then
    say "STEP 0a extract  ($(basename "$p0a"), $(date +%T))"
    ( cd "$PREPWORK" && conda run -n "$ENV" python "$p0a" ) && echo ">>> STEP 0a OK" || die "STEP 0a extract FAILED"
  fi
  if [ -n "$p0c" ]; then
    say "STEP 0c validation  ($(basename "$p0c"), $(date +%T))"
    ( cd "$PREPWORK" && conda run -n "$ENV" python "$p0c" ) && echo ">>> STEP 0c OK" || die "STEP 0c validation FAILED"
  fi
  cp -f "$PREPWORK"/*RawCounts*.h5ad          "$WORK"/ 2>/dev/null || true
  cp -f "$PREPWORK"/*External_Validation*.h5ad "$WORK"/ 2>/dev/null || die "step0c produced no external-validation file"
else
  runstep "STEP 0a extract"    "$ENV"           python  "step0a*extract_raw_counts*.py"
  runstep "STEP 0c validation" "$ENV"           python  "step0c*external_validation*.py"
fi

# step0b scDesign3 augmentation. scDesign3 is model-agnostic and runs once per
# (cohort, demographic); some model workflows (e.g. Geneformer/scGPT on AIDA) don't
# carry their own augmentation .R and instead embed the shared conditions. If this
# workflow has the .R, run it in place; otherwise run the canonical scFoundation
# augmentation and copy its (identically-named) condition files in.
if ls "$SRC_WF"/step0b_scdesign3_*augmentation*.R >/dev/null 2>&1; then
  runstep "STEP 0b scdesign3" scdesign3_env Rscript "step0b_scdesign3_*augmentation*.R"
else
  say "STEP 0b scdesign3 (shared — running canonical scFoundation augmentation)"
  case "$COHORT" in
    ild)  AUGREL="scfoundation/augmentedv4/${DEMO}_scfoundation_workflow" ;;
    crc)  AUGREL="scfoundation/augmented_CRC/${DEMO}_scfoundation_workflow" ;;
    aida) AUGREL="scfoundation/augmented_AIDA/${DEMO}_scfoundation_workflow" ;;
  esac
  AUGSRC="$SCFM_HOME/$AUGREL"; AUGWORK="$DATA_ROOT/$AUGREL"
  [ -d "$AUGSRC" ] || die "no canonical augmentation source: $AUGREL"
  mkdir -p "$AUGWORK"
  ln -sf "$WORK/$DATA_NAME" "$AUGWORK/$DATA_NAME"        # share the raw object
  # The canonical augmentation needs RawCounts + the external-validation split. The prep
  # stage already produced these dataset-level files (they're in $WORK); reuse them so the
  # shared branch doesn't have to re-extract or re-split (and so step0b doesn't die looking
  # for a validation file the shared workflow never generates itself).
  cp -f "$WORK"/*RawCounts*.h5ad           "$AUGWORK"/ 2>/dev/null || true
  cp -f "$WORK"/*External_Validation*.h5ad "$AUGWORK"/ 2>/dev/null || true
  a0a="$(ls "$AUGSRC"/step0a*extract_raw_counts*.py 2>/dev/null | head -1 || true)"
  a0b="$(ls "$AUGSRC"/step0b_scdesign3_*augmentation*.R 2>/dev/null | head -1 || true)"
  [ -n "$a0b" ] || die "canonical source has no augmentation .R: $AUGREL"
  # only re-run canonical step0a if prep didn't already supply RawCounts
  if [ -z "$(ls "$AUGWORK"/*RawCounts*.h5ad 2>/dev/null)" ] && [ -n "$a0a" ]; then
    echo ">>> (canonical) $(basename "$a0a")"; ( cd "$AUGWORK" && conda run -n "$ENV" python "$a0a" ) || die "canonical step0a FAILED"
  fi
  echo ">>> (canonical) $(basename "$a0b")"
  ( cd "$AUGWORK" && conda run -n scdesign3_env Rscript "$a0b" ) || die "canonical augmentation FAILED"
  echo ">>> copying shared *_Pilot_* conditions into $WORK"
  cp "$AUGWORK"/*_Pilot_*.h5ad "$WORK"/ 2>/dev/null || die "no *_Pilot_* conditions produced"
  echo ">>> STEP 0b scdesign3 (shared) OK"
fi
fi  # end: want prep

# step2a embed (GPU) — the only stage that needs a GPU
if want embed; then
  runstep "STEP 2a embed"    "$ENV"           python  "step2a*.py" 1
fi
# downstream (CPU)
if want down; then
  for s in step3a step3b step4 step4a step4b step5 step6 step7 step8 step9; do
    runstep "STEP ${s#step}" "$ENV"           python  "${s}[!0-9]*.py"
  done
fi

say "COMPLETE  $MODEL / $COHORT / $DEMO   ($(date +%T))"
echo "  outputs under: $WORK"
