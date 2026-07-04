#!/usr/bin/env bash
# Download the three cohorts from CZ CELLxGENE into ./data/.
# The files are NOT redistributed here; each keeps its own CELLxGENE license.
# These are the raw published objects. The pipeline filters them, so raw cell
# counts do not match the post-QC counts in the paper.
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"

# cohort label -> URL -> output filename
download() {
  local label="$1" url="$2" out="$DEST/$3"
  if [ -f "$out" ]; then
    echo "[$label] already present: $out (skipping)"
    return
  fi
  echo "[$label] downloading -> $out"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "ERROR: need curl or wget on PATH" >&2
    exit 1
  fi
}

# ILD  — Natri et al. 2024, Nature Genetics (GSE227136)
download ILD  "https://datasets.cellxgene.cziscience.com/c3d9262e-0dc5-4eca-bf20-56e6d96d0306.h5ad" "ILD_GSE227136.h5ad"

# CRC  — Moorman et al. 2024, Nature (epithelial compartment)
download CRC  "https://datasets.cellxgene.cziscience.com/66cadf3b-4c71-4930-8add-fa748745704d.h5ad" "CRC_Moorman2024.h5ad"

# AIDA — Kock et al. 2025, Cell (Phase 1 Data Freeze v2)
download AIDA "https://datasets.cellxgene.cziscience.com/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad" "AIDA_phase1_v2.h5ad"

echo
echo "Done. Files in $DEST/:"
ls -lh "$DEST"/*.h5ad 2>/dev/null || true
echo
echo "Note: each step script points BASE at its own data location; update BASE to match $DEST/ (or move the files) before running a workflow."
