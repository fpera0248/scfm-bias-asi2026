#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 1: SHARED SETUP
# - Loads input h5ad once
# - Computes proportional targets
# - Selects HVGs via scran (with Geneformer vocab filter)
# - Pre-caches anchor cells per group
# - Saves: shared_state.rds  (everything Stage 2 needs)
#          ld_<group>.rds    (per-group loaded SCE for Stage 2)
# ============================================================
# After this runs once, Stage 2 (per-group synthesis) can run in
# parallel for each minority group reading shared_state.rds and
# its own ld_<group>.rds.
# ============================================================

cat("\n=== STAGE 1: SETUP ===\n")

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDesign3)
  library(zellkonverter)
  library(Matrix)
  library(BiocParallel)
  library(RhpcBLASctl)
  library(scran)
  library(scuttle)
})

blas_set_num_threads(1L)
register(SerialParam(progressbar = FALSE))

# ============================================================
# CONFIG (must match stage 2)
# ============================================================

INPUT_H5AD      <- "ColorectalCancer_RawCounts_ETH.h5ad"
VALIDATION_H5AD <- "CRC_Eth_External_Validation_8572.h5ad"
GF_VOCAB_PATH   <- "geneformer_vocab_genes.txt"
SHARED_STATE    <- "stage1_shared_state.rds"

BIN_COL      <- "self_reported_ethnicity"
CELLTYPE_COL <- "cell_type"
SEX_COL      <- "sex"
AGE_COL      <- "development_stage"
AGE_BIN_COL  <- "age_bin_10yr"

UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                    "multiethnic", "na na", "not applicable", "prefer not to say")
CT_UNKNOWN     <- c("unknown", "na", "n/a", "not reported", "", "nan")
SEX_UNKNOWN    <- c("unknown", "na", "n/a", "not reported", "", "nan")
AGE_UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                        "not applicable", "adult", "child", "infant",
                        "embryonic", "fetal", "newborn")
AGE_BREAKS <- seq(10, 90, by = 10)
AGE_LABELS <- paste0(AGE_BREAKS[-length(AGE_BREAKS)], "_", AGE_BREAKS[-1] - 1)

PROPORTIONAL_SIZE <- 2500L
N_HVG             <- 1000L
MIN_DETECTION_RATE <- 0.005
MIN_CELLS         <- 5L
MIN_COUNTS        <- 10L
HVG_SAMPLE_SIZE   <- 5000L
MIN_CT_CELLS      <- 20L
ANCHOR_PER_CT_PER_BIN <- 2L
ASSAY_USE         <- "counts"

# ============================================================
# LOGGING
# ============================================================

log_con <- file("stage1_log.txt", open = "at")
on.exit(close(log_con), add = TRUE)
heartbeat <- function(msg, newline = FALSE) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg, if (newline) "\n" else "")
  cat(line, file = log_con); cat(line, file = stderr()); flush(log_con)
}

heartbeat("\n=== STAGE 1 START ===\n", TRUE)

# ============================================================
# HELPERS
# ============================================================

ensure_counts_assay <- function(sce_obj) {
  a <- assayNames(sce_obj)
  if ("counts" %in% a) {
    # ok
  } else if ("X" %in% a) {
    assay(sce_obj, "counts") <- assay(sce_obj, "X")
  } else stop("Neither 'counts' nor 'X' assay present.")
  assays(sce_obj) <- SimpleList(counts = assay(sce_obj, "counts"))
  mat <- assay(sce_obj, "counts")
  if (!inherits(mat, "dgCMatrix")) assay(sce_obj, "counts") <- as(mat, "dgCMatrix")
  sce_obj
}

clean_demographics <- function(sce_obj) {
  eth <- tolower(trimws(as.character(colData(sce_obj)[[BIN_COL]])))
  sex <- tolower(trimws(as.character(colData(sce_obj)[[SEX_COL]])))
  keep <- (!(eth %in% UNKNOWN_VALUES)) & (!is.na(eth)) &
          (!(sex %in% SEX_UNKNOWN)) & (!is.na(sex))
  sce_obj <- sce_obj[, keep]
  colData(sce_obj)[[BIN_COL]] <- as.character(tolower(trimws(as.character(colData(sce_obj)[[BIN_COL]]))))
  colData(sce_obj)[[SEX_COL]] <- as.character(tolower(trimws(as.character(colData(sce_obj)[[SEX_COL]]))))
  sce_obj
}

add_age_bins <- function(sce_obj) {
  if (!AGE_COL %in% colnames(colData(sce_obj))) {
    colData(sce_obj)[[AGE_BIN_COL]] <- NA_character_
    return(sce_obj)
  }
  age_raw <- tolower(trimws(as.character(colData(sce_obj)[[AGE_COL]])))
  unknown_mask <- age_raw %in% AGE_UNKNOWN_VALUES | is.na(age_raw)
  age_digits <- gsub("[^0-9]", "", age_raw)
  age_num <- suppressWarnings(as.numeric(age_digits))
  age_num[unknown_mask | age_digits == ""] <- NA_real_
  in_range <- !is.na(age_num) & age_num >= AGE_BREAKS[1] & age_num < AGE_BREAKS[length(AGE_BREAKS)]
  age_bin <- rep(NA_character_, ncol(sce_obj))
  if (any(in_range)) {
    cut_result <- cut(age_num[in_range], breaks = AGE_BREAKS, labels = AGE_LABELS,
                      include.lowest = TRUE, right = FALSE)
    age_bin[in_range] <- as.character(droplevels(cut_result))
  }
  colData(sce_obj)[[AGE_BIN_COL]] <- age_bin
  sce_obj
}

# ============================================================
# 1. Validation barcodes
# ============================================================

heartbeat("Loading validation barcodes...\n", TRUE)
sce_val  <- zellkonverter::readH5AD(VALIDATION_H5AD, use_hdf5 = TRUE)
VALIDATION_BARCODES <- colnames(sce_val)
rm(sce_val); gc()
heartbeat(sprintf("Loaded %d validation barcodes\n", length(VALIDATION_BARCODES)), TRUE)

exclude_validation_cells <- function(sce_obj) {
  keep <- !colnames(sce_obj) %in% VALIDATION_BARCODES
  sce_obj[, keep]
}

# ============================================================
# 2. Pre-compute full-gene library sizes
# ============================================================

heartbeat("Pre-computing full-gene library sizes...\n", TRUE)
sce_libsize_tmp <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_libsize_tmp <- ensure_counts_assay(sce_libsize_tmp)
LIBRARY_SIZES <- Matrix::colSums(assay(sce_libsize_tmp, "counts"))
heartbeat(sprintf("Lib size mean=%.0f sd=%.0f\n", mean(LIBRARY_SIZES), sd(LIBRARY_SIZES)), TRUE)
rm(sce_libsize_tmp); gc()

# ============================================================
# 3. Auto-detect groups
# ============================================================

heartbeat("Auto-detecting groups...\n", TRUE)
sce_detect <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_detect <- ensure_counts_assay(sce_detect)
sce_detect <- exclude_validation_cells(sce_detect)
sce_detect <- clean_demographics(sce_detect)
sce_detect <- add_age_bins(sce_detect)

bin_table <- table(colData(sce_detect)[[BIN_COL]])
all_bins  <- sort(names(bin_table))
MAJ_BIN   <- all_bins[which.max(bin_table)]
total_real <- sum(as.numeric(bin_table))

prop_targets <- sapply(as.numeric(bin_table),
                       function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins
TARGET_PER_BIN <- prop_targets[[MAJ_BIN]]

heartbeat(sprintf("Groups: %s\n", paste(sprintf("%s=%d", all_bins, bin_table), collapse=", ")), TRUE)
heartbeat(sprintf("MAJ=%s  TARGET_PER_BIN=%d  total_pool=%d\n",
                  MAJ_BIN, TARGET_PER_BIN, total_real), TRUE)

# Save sce_detect for the proportional+downsample dataset generation in stage 3
saveRDS(sce_detect, "stage1_sce_detect.rds")
rm(sce_detect); gc()

# ============================================================
# 4. HVG selection with Geneformer vocab filter
# ============================================================

heartbeat("HVG selection (scran::modelGeneVar + Geneformer vocab)...\n", TRUE)
sce_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_full <- ensure_counts_assay(sce_full)
sce_full <- exclude_validation_cells(sce_full)
sce_full <- clean_demographics(sce_full)
sce_full <- add_age_bins(sce_full)
assay(sce_full, ASSAY_USE) <- as(as.matrix(assay(sce_full, ASSAY_USE)), "dgCMatrix")
original_genes <- rownames(sce_full)
heartbeat(sprintf("Full cleaned: %d cells x %d genes\n", ncol(sce_full), nrow(sce_full)), TRUE)

bin_labels_full <- colData(sce_full)[[BIN_COL]]
bin_counts_full <- table(bin_labels_full)
target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
n_per_bin_hvg <- pmin(as.numeric(bin_counts_full), target_per_bin_hvg)
names(n_per_bin_hvg) <- names(bin_counts_full)

set.seed(123)
sample_idx_hvg <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_full == b)
  if (length(idx) > n_per_bin_hvg[b]) sample(idx, n_per_bin_hvg[b]) else idx
}))

sce_sample <- sce_full[, sample_idx_hvg]
count_mat <- assay(sce_sample, ASSAY_USE)
gene_detect_rate <- Matrix::rowMeans(count_mat > 0)
gene_sums <- Matrix::rowSums(count_mat)
gene_nonzero <- Matrix::rowSums(count_mat > 0)
keep_genes <- (gene_sums >= MIN_COUNTS) & (gene_nonzero >= MIN_CELLS) & (gene_detect_rate >= MIN_DETECTION_RATE)
heartbeat(sprintf("Detection filter: %d -> %d genes\n", nrow(sce_sample), sum(keep_genes)), TRUE)

sce_filt <- sce_sample[keep_genes, ]
sce_filt <- scuttle::logNormCounts(sce_filt)
var_model <- scran::modelGeneVar(sce_filt, BPPARAM = SerialParam())
hvg_names <- scran::getTopHVGs(var_model, n = N_HVG)
hvg_names <- intersect(hvg_names, rownames(sce_full))
heartbeat(sprintf("Selected %d HVGs via scran\n", length(hvg_names)), TRUE)

# Geneformer vocab filter
if (file.exists(GF_VOCAB_PATH)) {
  gf_vocab <- readLines(GF_VOCAB_PATH)
  hvg_stripped <- sub("\\.[0-9]+$", "", hvg_names)
  in_vocab <- hvg_stripped %in% gf_vocab
  hvg_names <- hvg_names[in_vocab]
  heartbeat(sprintf("Geneformer vocab filter: %d / %d HVGs retained\n",
                    sum(in_vocab), length(in_vocab)), TRUE)
  if (length(hvg_names) < 100L) stop(sprintf("FATAL: only %d HVGs in vocab", length(hvg_names)))
} else {
  heartbeat("WARNING: vocab file not found, no filter applied\n", TRUE)
}

rm(sce_sample, sce_filt, count_mat, var_model); gc()

# ============================================================
# 5. Per-group SCE preparation + anchor caching
# ============================================================

heartbeat("Preparing per-group SCEs and anchor cache...\n", TRUE)

prepare_group_sce <- function(bin_label) {
  heartbeat(sprintf("  Preparing '%s'...\n", bin_label), TRUE)
  sce_raw <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
  common_early <- intersect(hvg_names, rownames(sce_raw))
  sce_raw <- sce_raw[common_early, ]
  sce_raw <- ensure_counts_assay(sce_raw)
  sce_raw <- exclude_validation_cells(sce_raw)
  sce_raw <- clean_demographics(sce_raw)
  sce_raw <- add_age_bins(sce_raw)

  ct_raw <- as.character(colData(sce_raw)[[CELLTYPE_COL]])
  valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
  sce_raw <- sce_raw[, valid_ct]

  eth_vec <- as.character(colData(sce_raw)[[BIN_COL]])
  sce_bin <- sce_raw[, eth_vec == bin_label]

  ct_vec <- as.character(colData(sce_bin)[[CELLTYPE_COL]])
  tab_ct <- table(ct_vec)
  keep_ct <- names(tab_ct)[tab_ct >= MIN_CT_CELLS]
  if (length(keep_ct) == 0) return(NULL)
  sce_bin <- sce_bin[, ct_vec %in% keep_ct]
  ct_vec <- ct_vec[ct_vec %in% keep_ct]
  ct_vec[is.na(ct_vec)] <- "unknown_cleaned"
  colData(sce_bin)[[CELLTYPE_COL]] <- droplevels(factor(ct_vec))
  colData(sce_bin)[[BIN_COL]] <- as.character(colData(sce_bin)[[BIN_COL]])
  colData(sce_bin)[[SEX_COL]] <- as.character(colData(sce_bin)[[SEX_COL]])
  colData(sce_bin)[[AGE_BIN_COL]] <- as.character(colData(sce_bin)[[AGE_BIN_COL]])

  lib_vec <- LIBRARY_SIZES[colnames(sce_bin)]
  lib_vec[is.na(lib_vec)] <- median(LIBRARY_SIZES, na.rm = TRUE)
  colData(sce_bin)$library <- as.numeric(lib_vec)
  rm(sce_raw); gc()
  sce_bin
}

anchor_cache <- list()
for (b in all_bins) {
  sce_b <- tryCatch(prepare_group_sce(b), error = function(e) {
    heartbeat(sprintf("  ERROR preparing '%s': %s\n", b, conditionMessage(e)), TRUE)
    NULL
  })
  if (is.null(sce_b)) next

  # Save full per-group SCE for Stage 2
  safe_b <- gsub("[^a-zA-Z0-9_]", "_", b)
  saveRDS(sce_b, sprintf("stage1_sce_%s.rds", safe_b))

  # Anchor sampling
  ct_b <- as.character(colData(sce_b)[[CELLTYPE_COL]])
  tab_b <- table(ct_b)
  set.seed(456)
  anch_idx <- unlist(lapply(names(tab_b), function(ct) {
    ids <- which(ct_b == ct)
    sample(ids, min(ANCHOR_PER_CT_PER_BIN, length(ids)))
  }))
  anchor_cache[[b]] <- sce_b[, anch_idx]
  heartbeat(sprintf("  Saved %s SCE (%d cells) + %d anchors\n",
                    b, ncol(sce_b), length(anch_idx)), TRUE)
  rm(sce_b); gc()
}

# ============================================================
# 6. Save shared state
# ============================================================

heartbeat("Saving shared state...\n", TRUE)
saveRDS(list(
  hvg_names      = hvg_names,
  original_genes = original_genes,
  prop_targets   = prop_targets,
  TARGET_PER_BIN = TARGET_PER_BIN,
  MAJ_BIN        = MAJ_BIN,
  all_bins       = all_bins,
  anchor_cache   = anchor_cache,
  bin_table      = bin_table,
  total_real     = total_real,
  LIBRARY_SIZES  = LIBRARY_SIZES,
  VALIDATION_BARCODES = VALIDATION_BARCODES,
  rd_full        = rowData(sce_full)
), SHARED_STATE)

heartbeat(sprintf("Wrote %s and per-group SCEs\n", SHARED_STATE), TRUE)

# Save full sce for stage 3 (final restoration step)
heartbeat("Saving full SCE for stage 3 gene restoration...\n", TRUE)
saveRDS(sce_full, "stage1_sce_full.rds")

heartbeat("\n=== STAGE 1 COMPLETE ===\n", TRUE)
