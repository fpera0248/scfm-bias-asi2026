#!/bin/bash
#SBATCH --job-name=regen_prop
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/regen_prop_%j.out
#SBATCH --error=logs/regen_prop_%j.err

set -euo pipefail
cd /data/scfoundation/augmentedv4/ethnicity_scfoundation_workflow

module purge
module load miniforge3/25.3.0-3
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scdesign3_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export R_MAX_VSIZE=120Gb

echo "=== regen Proportional START $(date -Iseconds) ==="

Rscript --vanilla - <<'REOF'
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

INPUT_H5AD <- "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"
BIN_COL <- "self_reported_ethnicity"
SEX_COL <- "sex"
OUT_FILE <- "ILD_Ethnicity_Pilot_Proportional_2497_ETHNICITY.h5ad"

cat(sprintf("[%s] Loading shared state...\n", format(Sys.time(), "%H:%M:%S")))
ss <- readRDS("stage1_shared_state.rds")
prop_targets <- ss$prop_targets
all_bins <- ss$all_bins
VALIDATION_BARCODES <- ss$VALIDATION_BARCODES

cat(sprintf("[%s] Loading INPUT_H5AD fresh (use_hdf5=FALSE)...\n", format(Sys.time(), "%H:%M:%S")))
sce <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
if (!"counts" %in% assayNames(sce) && "X" %in% assayNames(sce)) {
  assay(sce, "counts") <- assay(sce, "X")
}
assays(sce) <- SimpleList(counts = as(as.matrix(assay(sce, "counts")), "dgCMatrix"))
cat(sprintf("[%s] Loaded %d cells x %d genes (mean lib %.0f)\n",
            format(Sys.time(), "%H:%M:%S"),
            ncol(sce), nrow(sce),
            mean(Matrix::colSums(assay(sce, "counts")))))

keep_val <- !colnames(sce) %in% VALIDATION_BARCODES
sce <- sce[, keep_val]

eth_raw <- tolower(trimws(as.character(colData(sce)[[BIN_COL]])))
sex_raw <- tolower(trimws(as.character(colData(sce)[[SEX_COL]])))
unknown_eth <- c("unknown", "na", "n/a", "not reported", "", "nan",
                 "multiethnic", "na na", "not applicable", "prefer not to say")
unknown_sex <- c("unknown", "na", "n/a", "not reported", "", "nan")
keep_demo <- !(eth_raw %in% unknown_eth) & !is.na(eth_raw) &
             !(sex_raw %in% unknown_sex) & !is.na(sex_raw)
sce <- sce[, keep_demo]
colData(sce)[[BIN_COL]] <- tolower(trimws(as.character(colData(sce)[[BIN_COL]])))
cat(sprintf("[%s] After validation+demo cleanup: %d cells\n",
            format(Sys.time(), "%H:%M:%S"), ncol(sce)))

bin_labels <- colData(sce)[[BIN_COL]]
set.seed(123)
prop_sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels == b)
  n_b <- prop_targets[[b]]
  if (n_b <= 0 || length(idx) == 0) return(integer(0))
  sample(idx, n_b)
}))
sce_prop <- sce[, prop_sample_idx]
cat(sprintf("[%s] Sampled %d cells (target sum %d, mean lib %.0f)\n",
            format(Sys.time(), "%H:%M:%S"),
            ncol(sce_prop), sum(prop_targets),
            mean(Matrix::colSums(assay(sce_prop, "counts")))))

n <- ncol(sce_prop); cd_raw <- colData(sce_prop)
col_list <- lapply(colnames(cd_raw), function(col) {
  v <- tryCatch({
    tmp <- cd_raw[[col]]
    vapply(seq_len(n), function(i) {
      xi <- tryCatch(tmp[[i]], error = function(e) NA)
      if (is.null(xi) || length(xi) == 0 || (length(xi) == 1 && is.na(xi))) return(NA_character_)
      paste(as.character(xi), collapse = ";")
    }, character(1))
  }, error = function(e) rep(NA_character_, n))
  num <- suppressWarnings(as.numeric(v))
  if (!any(is.na(num) & !is.na(v))) return(num)
  v
})
names(col_list) <- colnames(cd_raw)
rn_safe <- make.unique(as.character(colnames(sce_prop)), sep = "_dup")
cd_new <- structure(col_list, class = "data.frame", row.names = rn_safe, names = names(col_list))
cnt <- assay(sce_prop, "counts")
colnames(cnt) <- rn_safe
sce_out <- SingleCellExperiment(assays = list(counts = cnt), colData = DataFrame(cd_new))

cat(sprintf("[%s] Writing %s...\n", format(Sys.time(), "%H:%M:%S"), OUT_FILE))
zellkonverter::writeH5AD(sce_out, OUT_FILE, compression = "gzip")
cat(sprintf("[%s] DONE\n", format(Sys.time(), "%H:%M:%S")))
REOF

echo "=== regen Proportional END $(date -Iseconds) ==="
