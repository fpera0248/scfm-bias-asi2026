#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 3: AGGREGATE + WRITE FAIRNESS DATASETS (SEX)
# ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

INPUT_H5AD  <- "InterstitialLungDisease_RawCounts_SEX.h5ad"
OUTPUT_BASE <- "ILD_Sex_Pilot"
SUFFIX      <- "SEX"
ASSAY_USE   <- "counts"

BIN_COL  <- "sex"
VALID_SEX <- c("male", "female")

IN1 <- "stage1_outputs"
IN2 <- "stage2_outputs"


sanitize_for_h5ad <- function(sce_obj) {
  cd <- colData(sce_obj)
  n  <- ncol(sce_obj)
  rn <- make.unique(as.character(rownames(cd)), sep = "_dup")
  KEEP <- c("sex", "cell_type", "age_bin_10yr", "self_reported_ethnicity",
            "library", "source", "donor_id", "disease",
            "development_stage", "assay", "tissue")
  cols_present <- intersect(KEEP, colnames(cd))
  new_df <- data.frame(row.names = rn, stringsAsFactors = FALSE)
  for (col in cols_present) {
    v <- tryCatch(cd[[col]], error = function(e) NULL)
    if (is.null(v) || length(v) != n) {
      new_df[[col]] <- rep(NA_character_, n)
      next
    }
    if (col == "library") {
      new_df[[col]] <- as.numeric(unname(unclass(v)))
      next
    }
    if (is.factor(v)) {
      flat_chr <- as.character(v)
    } else {
      flat <- unclass(v)
      attributes(flat) <- NULL
      if (length(flat) != n) {
        new_df[[col]] <- rep(NA_character_, n)
        next
      }
      flat_chr <- as.character(flat)
    }
    if (all(is.na(flat_chr))) flat_chr <- rep("unknown", n)
    else flat_chr[is.na(flat_chr)] <- "unknown"
    new_df[[col]] <- flat_chr
  }
  colData(sce_obj) <- DataFrame(new_df, row.names = rn)
  cat("[sanitize] Final colData class summary:\n")
  for (cn in colnames(colData(sce_obj))) {
    v <- colData(sce_obj)[[cn]]
    cat(sprintf("  %s: class=%s, typeof=%s, len=%d, na=%d\n",
                cn, paste(class(v), collapse="/"), typeof(v), length(v), sum(is.na(v))))
  }
  sce_obj
}

log_msg <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
  flush.console()
}

log_msg("STAGE 3 START")

all_bins        <- readRDS(file.path(IN1, "all_bins.rds"))
prop_targets    <- readRDS(file.path(IN1, "prop_targets.rds"))
prop_total      <- readRDS(file.path(IN1, "prop_total.rds"))
target_per_bin  <- readRDS(file.path(IN1, "target_per_bin.rds"))
down_target     <- readRDS(file.path(IN1, "down_target.rds"))
val_barcodes    <- readRDS(file.path(IN1, "validation_barcodes.rds"))
original_genes  <- readRDS(file.path(IN1, "original_genes.rds"))

log_msg(sprintf("Config: prop_total=%d, target_per_bin=%d, down_target=%d",
                prop_total, target_per_bin, down_target))

out_prop    <- sprintf("%s_Proportional_%d_%s.h5ad", OUTPUT_BASE, prop_total, SUFFIX)
out_balaug  <- sprintf("%s_BalancedAugmented_%dEach_%s.h5ad", OUTPUT_BASE, target_per_bin, SUFFIX)
out_balup   <- sprintf("%s_BalancedUpsampled_%dEach_%s.h5ad", OUTPUT_BASE, target_per_bin, SUFFIX)
out_down    <- sprintf("%s_Downsampled_%dEach_%s.h5ad", OUTPUT_BASE, down_target, SUFFIX)
out_full    <- sprintf("%s_Full_BalancedAugmented_%s.h5ad", OUTPUT_BASE, SUFFIX)
out_real    <- sprintf("%s_RealOnly_%s.h5ad", OUTPUT_BASE, SUFFIX)
out_syn     <- sprintf("%s_SyntheticOnly_%s.h5ad", OUTPUT_BASE, SUFFIX)
out_sum     <- sprintf("%s_Summary_%s.csv", OUTPUT_BASE, SUFFIX)

log_msg("Reading per-bin SCEs from stage2...")
bin_objs <- list()
for (b in all_bins) {
  fn <- file.path(IN2, sprintf("%s.rds", b))
  if (!file.exists(fn)) stop(sprintf("Missing stage2 output: %s", fn))
  bin_objs[[b]] <- readRDS(fn)
  log_msg(sprintf("  %s: %d cells", b, ncol(bin_objs[[b]])))
}

log_msg("Combining bins on common HVGs...")
common_genes <- Reduce(intersect, lapply(bin_objs, rownames))
bin_objs <- lapply(bin_objs, function(s) { s <- s[common_genes, ]; rowData(s) <- NULL; s })
sce_combined <- do.call(cbind, bin_objs)
log_msg(sprintf("Combined: %d cells x %d genes", ncol(sce_combined), nrow(sce_combined)))

colData(sce_combined)$source <- ifelse(grepl("^synthetic_", colnames(sce_combined)),
                                       "synthetic", "real")

log_msg("Restoring full gene set via triplet construction...")
log_msg("  Loading source h5ad for real-cell gene restoration...")
src_sce <- readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
if (!"counts" %in% assayNames(src_sce)) {
  if ("X" %in% assayNames(src_sce)) assay(src_sce, "counts") <- assay(src_sce, "X")
  else stop("No counts/X assay")
}
src_counts <- as(assay(src_sce, "counts"), "dgCMatrix")
log_msg(sprintf("  Source: %d cells x %d genes", ncol(src_counts), nrow(src_counts)))

real_mask <- colData(sce_combined)$source == "real"
real_names <- colnames(sce_combined)[real_mask]
syn_names  <- colnames(sce_combined)[!real_mask]
n_real <- length(real_names); n_syn <- length(syn_names)
log_msg(sprintf("  Combined cells: %d real + %d synthetic", n_real, n_syn))

real_in_src <- match(real_names, colnames(src_counts))
if (any(is.na(real_in_src))) {
  stop(sprintf("FATAL: %d real barcodes not in source", sum(is.na(real_in_src))))
}

log_msg("  Subsetting real cells from source...")
real_full <- src_counts[, real_in_src, drop=FALSE]
colnames(real_full) <- real_names

log_msg("  Building synthetic full-gene matrix via triplet construction...")
syn_hvg <- assay(sce_combined, "counts")[, !real_mask, drop=FALSE]
syn_hvg <- as(syn_hvg, "dgCMatrix")

hvg_to_full <- match(rownames(syn_hvg), original_genes)
if (any(is.na(hvg_to_full))) {
  log_msg(sprintf("  WARN: %d HVGs not in original_genes, dropping", sum(is.na(hvg_to_full))))
  keep <- !is.na(hvg_to_full)
  syn_hvg <- syn_hvg[keep, ]
  hvg_to_full <- hvg_to_full[keep]
}

syn_t <- as(syn_hvg, "TsparseMatrix")
syn_full <- sparseMatrix(
  i = hvg_to_full[syn_t@i + 1L],
  j = syn_t@j + 1L,
  x = syn_t@x,
  dims = c(length(original_genes), n_syn),
  dimnames = list(original_genes, syn_names)
)

log_msg("  Combining real + synthetic on full gene space...")
combined_full <- cbind(syn_full, real_full)
combined_names <- c(syn_names, real_names)
order_map <- match(colnames(sce_combined), combined_names)
combined_full <- combined_full[, order_map, drop=FALSE]
colnames(combined_full) <- colnames(sce_combined)

log_msg(sprintf("  combined_full: %d genes x %d cells", nrow(combined_full), ncol(combined_full)))

log_msg("Building final SCE objects...")
cd_combined <- colData(sce_combined)
sce_full <- SingleCellExperiment(
  assays = list(counts = combined_full),
  colData = cd_combined
)

log_msg("\n=== BUILDING FAIRNESS DATASETS ===")

log_msg(sprintf("Writing %s...", out_balaug))
writeH5AD(sanitize_for_h5ad(sce_full), out_balaug, compression = "gzip")

real_mask_full <- colData(sce_full)$source == "real"
sce_real_only <- sce_full[, real_mask_full]
log_msg(sprintf("Writing %s (n=%d)...", out_real, ncol(sce_real_only)))
writeH5AD(sanitize_for_h5ad(sce_real_only), out_real, compression = "gzip")

sce_syn_only <- sce_full[, !real_mask_full]
log_msg(sprintf("Writing %s (n=%d)...", out_syn, ncol(sce_syn_only)))
writeH5AD(sanitize_for_h5ad(sce_syn_only), out_syn, compression = "gzip")

log_msg(sprintf("Writing %s...", out_full))
writeH5AD(sanitize_for_h5ad(sce_full), out_full, compression = "gzip")

# Proportional from source
log_msg(sprintf("Building Proportional (target sum=%d)...", prop_total))
src_obs_sex <- tolower(trimws(as.character(colData(src_sce)[[BIN_COL]])))
src_obs_names <- colnames(src_sce)
not_val_mask <- !src_obs_names %in% val_barcodes
valid_sex_mask <- src_obs_sex %in% VALID_SEX

set.seed(123)
prop_idx <- unlist(lapply(all_bins, function(b) {
  candidate <- which(not_val_mask & valid_sex_mask & src_obs_sex == b)
  n_b <- prop_targets[b]
  if (n_b <= 0 || length(candidate) == 0) integer(0)
  else sample(candidate, min(n_b, length(candidate)))
}))
sce_prop <- src_sce[, prop_idx]
sce_prop <- sanitize_for_h5ad(sce_prop)

prop_check <- sum(assay(sce_prop, "counts"))
log_msg(sprintf("  Proportional count sum: %.0f", prop_check))
if (prop_check == 0) stop("FATAL: Proportional count matrix all zeros")
log_msg(sprintf("Writing %s (n=%d)...", out_prop, ncol(sce_prop)))
writeH5AD(sce_prop, out_prop, compression = "gzip")

# BalancedUpsampled
log_msg(sprintf("Building BalancedUpsampled (TARGET_PER_BIN=%d)...", target_per_bin))
real_bins_full <- as.character(colData(sce_real_only)[[BIN_COL]])
set.seed(456)
up_idx <- unlist(lapply(all_bins, function(b) {
  ids <- which(real_bins_full == b)
  if (length(ids) == 0) integer(0)
  else sample(ids, target_per_bin, replace = TRUE)
}))
sce_up <- sce_real_only[, up_idx]
sce_up <- sanitize_for_h5ad(sce_up)
log_msg(sprintf("Writing %s (n=%d)...", out_balup, ncol(sce_up)))
writeH5AD(sce_up, out_balup, compression = "gzip")

# Downsampled
log_msg(sprintf("Building Downsampled (DOWN_TARGET=%d per bin)...", down_target))
set.seed(789)
down_idx <- unlist(lapply(all_bins, function(b) {
  ids <- which(real_bins_full == b)
  n <- min(down_target, length(ids))
  if (n == 0) integer(0) else sample(ids, n, replace = FALSE)
}))
sce_down <- sce_real_only[, down_idx]
sce_down <- sanitize_for_h5ad(sce_down)
log_msg(sprintf("Writing %s (n=%d)...", out_down, ncol(sce_down)))
writeH5AD(sce_down, out_down, compression = "gzip")

log_msg(sprintf("Writing %s...", out_sum))
summary_df <- as.data.frame(table(
  Sex = colData(sce_full)[[BIN_COL]],
  Source = colData(sce_full)$source
))
colnames(summary_df) <- c("Sex", "Source", "Count")
write.csv(summary_df, out_sum, row.names = FALSE)

log_msg("\n=== STAGE 3 COMPLETE ===")
log_msg("Output files:")
for (f in c(out_prop, out_balaug, out_balup, out_down, out_full, out_real, out_syn, out_sum)) {
  log_msg(sprintf("  %s", f))
}
