#!/usr/bin/env Rscript
# ============================================================
# STEP 0B (v6): ETHNICITY AUGMENTATION
# Base: Geneformer v5 proportional-allocation logic
# Patches: v6 fixes for scFoundation workflow
# ============================================================
# Produces 4 fairness datasets at PROPORTIONAL_SIZE=2500 cells total:
#   - Proportional     : natural ethnicity proportions, real cells only
#   - BalancedAugmented: every group padded with scDesign3 synthetic
#                        up to TARGET_PER_BIN (EA share = ~2143)
#                        EA contributes real only (no synthesis)
#   - BalancedUpsampled: every group duplicated with replacement to TARGET_PER_BIN
#   - Downsampled      : every group reduced to smallest real group count
#
# ============================================================
# V6 changes vs v5 (Geneformer script as base):
#   [V6 1] HVG: scran::modelGeneVar instead of CV^2
#          (CV^2 = var/mean^2 systematically picks low-mean sporadic genes;
#          Zhao et al. 2024 mixHVG benchmark shows scran tied for best)
#   [V6 2] family_use: "nb" instead of "zinb"
#          (Svensson 2020, Cao 2021, Kim 2024: droplet UMI data are not
#          zero-inflated; ZINB's forced-zero step compounds sparsity)
#   [V6 3] important_feature: explicit logical vector of length = nrow(sce)
#          named by gene. scDesign3 1.5.0's "auto" produces a wrong-length
#          vector for simu_new.
#   [V6 4] readH5AD(use_hdf5=FALSE) + as.matrix realization in HVG section.
#          zellkonverter's DelayedMatrix silently returns zeros after
#          transposition failure, causing rowSums=0.
#   [V6 5] mgcv::bam acceleration: usebam=TRUE, edf_flexible=TRUE
#   [V6 6] Parallelization: bpmapply + MulticoreParam(N_CORES)
#   [V6 7] ANCHOR_PER_CT_PER_BIN 2 -> 20
#   [V6 8] N_CORES from SLURM_CPUS_PER_TASK (was hardcoded 1)
#   [V6 9] Post-sim QC gate: reject chunk if synth < 50% of real on
#          active genes or library size, compared to sub_real.
#   [V6 10] Aggressive gc() between chunks + mem_used logging
#   [V6 11] Real-cell full-gene restoration before writing final h5ad
#   [V6 12] Fallback additive mu_formula if interaction fails
#   [V6 13] mu_formula fallback retry in run_sd3_lowlevel()
# All FIX 1-22 patches from older scripts preserved verbatim.
# ============================================================

cat("\n=== SLURM environment ===\n")
cat("SLURM_CPUS_PER_TASK =", Sys.getenv("SLURM_CPUS_PER_TASK"), "\n")
cat("SLURM_JOB_ID =", Sys.getenv("SLURM_JOB_ID"), "\n")

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

# Serial execution (per Geneformer v5 pattern).
# Parallel BPPARAM duplicates SCE + marginal state per worker, causing 8x
# memory blowup and OOM kills on mgcv::bam with many cell types. mcmapply
# with n_cores=1 is slower per chunk but uses ~1/8 the memory.
N_CORES <- 1L
blas_set_num_threads(1L)
BP <- SerialParam(progressbar = FALSE)
register(BP)
cat(sprintf("Using N_CORES=%d (serial) for scDesign3 to avoid OOM\n", N_CORES))

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_H5AD      <- "AIDA_RawCounts_ETHNICITY_900k.h5ad"  # [FIX] match step0a OUTPUT_FILE + rest of chain (step0c/step2a); was "AIDA_RawCounts_ETHNICITY.h5ad" -> FileNotFoundError
VALIDATION_H5AD <- "AIDA_Ethnicity_External_Validation_12500.h5ad"
OUTPUT_BASE     <- "AIDA_Ethnicity_Pilot"
CKPTDIR         <- "checkpoints_ethnicity_v6"
LOGFILE         <- "augmentation_ethnicity_v6_log.txt"

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

PROPORTIONAL_SIZE  <- 2500L
CHUNK_SIZE_DEFAULT <- 200L
CHUNK_SIZE_MIN     <- 100L
MAX_RETRIES        <- 3L

# [V6 1] HVG via scran
N_HVG              <- 1000L
MIN_DETECTION_RATE <- 0.005
MIN_CELLS          <- 5L
MIN_COUNTS         <- 10L
HVG_SAMPLE_SIZE    <- 5000L

MIN_CT_CELLS          <- 20L
MAX_CELLS_PER_BIN     <- 500L
MIN_CELLS_INTERACTION <- 50L

# [V6 7] More anchors per CT per group
ANCHOR_PER_CT_PER_BIN <- 2L

TARGET_PER_BIN <- NULL
TARGET_DOWN    <- NULL
MAJ_BIN        <- NULL
MIN_BIN        <- NULL

ASSAY_USE <- "counts"

# [V6 2, 3, 5] scDesign3 config
FAMILY_USE  <- "nb"
COPULA_TYPE <- "gaussian"
USE_BAM     <- FALSE
EDF_FLEX    <- TRUE

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

log_con <- file(LOGFILE, open = "at")
on.exit(close(log_con), add = TRUE)

heartbeat <- function(msg, newline = FALSE) {
  ts   <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg)
  if (newline) line <- paste0(line, "\n")
  cat(line, file = log_con)
  cat(line, file = stderr())
  flush(log_con)
}

heartbeat("\n=== STARTING ETHNICITY AUGMENTATION v6 ===\n", TRUE)
heartbeat(sprintf("PROPORTIONAL_SIZE=%d  N_HVG=%d  family=%s  copula=%s  usebam=%s\n",
                  PROPORTIONAL_SIZE, N_HVG, FAMILY_USE, COPULA_TYPE, USE_BAM), TRUE)
heartbeat(sprintf("ANCHOR_PER_CT_PER_BIN=%d  N_CORES=%d  MAX_CELLS_PER_BIN=%d\n",
                  ANCHOR_PER_CT_PER_BIN, N_CORES, MAX_CELLS_PER_BIN), TRUE)

# ============================================================
# VALIDATION BARCODE EXCLUSION
# ============================================================

load_validation_barcodes <- function() {
  if (!file.exists(VALIDATION_H5AD))
    stop(sprintf("Validation file not found: %s", VALIDATION_H5AD))
  sce_val  <- zellkonverter::readH5AD(VALIDATION_H5AD, use_hdf5 = TRUE)
  barcodes <- colnames(sce_val)
  rm(sce_val); gc()
  heartbeat(sprintf("Loaded %d validation barcodes to exclude.\n", length(barcodes)), TRUE)
  barcodes
}

exclude_validation_cells <- function(sce_obj) {
  if (length(VALIDATION_BARCODES) == 0L) return(sce_obj)
  n_before  <- ncol(sce_obj)
  keep      <- !colnames(sce_obj) %in% VALIDATION_BARCODES
  sce_obj   <- sce_obj[, keep]
  n_removed <- n_before - ncol(sce_obj)
  if (n_removed > 0) {
    heartbeat(sprintf(" Excluded %d validation cells (kept %d).\n",
                      n_removed, ncol(sce_obj)), TRUE)
  }
  sce_obj
}

VALIDATION_BARCODES <- load_validation_barcodes()

# ============================================================
# ADD_AGE_BINS
# ============================================================

add_age_bins <- function(sce_obj) {
  if (!AGE_COL %in% colnames(colData(sce_obj))) {
    colData(sce_obj)[[AGE_BIN_COL]] <- NA_character_
    return(sce_obj)
  }
  age_raw      <- tolower(trimws(as.character(colData(sce_obj)[[AGE_COL]])))
  unknown_mask <- age_raw %in% AGE_UNKNOWN_VALUES | is.na(age_raw)
  age_digits   <- gsub("[^0-9]", "", age_raw)
  age_num      <- suppressWarnings(as.numeric(age_digits))
  age_num[unknown_mask | age_digits == ""] <- NA_real_
  in_range <- !is.na(age_num) & age_num >= AGE_BREAKS[1] &
              age_num < AGE_BREAKS[length(AGE_BREAKS)]
  age_bin <- rep(NA_character_, ncol(sce_obj))
  if (any(in_range)) {
    cut_result        <- cut(age_num[in_range], breaks = AGE_BREAKS,
                             labels = AGE_LABELS, include.lowest = TRUE, right = FALSE)
    age_bin[in_range] <- as.character(droplevels(cut_result))
  }
  colData(sce_obj)[[AGE_BIN_COL]] <- age_bin
  sce_obj
}

# ============================================================
# ASSAY HELPERS
# ============================================================

assert_raw_counts <- function(sce_obj, assay_name = "counts") {
  if (!assay_name %in% assayNames(sce_obj))
    stop(sprintf("Assay '%s' not found.", assay_name))
  mat <- assay(sce_obj, assay_name)
  vals_all <- if (inherits(mat, "dgCMatrix")) mat@x else {
    nr <- nrow(mat); nc <- ncol(mat)
    if (nr == 0 || nc == 0) return(invisible(TRUE))
    samp_n <- min(10000L, nr * nc)
    mat[cbind(sample.int(nr, samp_n, replace = TRUE),
              sample.int(nc, samp_n, replace = TRUE))]
  }
  if (length(vals_all) == 0L) return(invisible(TRUE))
  vals <- vals_all[seq_len(min(10000L, length(vals_all)))]
  if (any(abs(vals - round(vals)) > .Machine$double.eps^0.5))
    stop(sprintf("Assay '%s' contains non-integer values.", assay_name))
  invisible(TRUE)
}

ensure_counts_assay <- function(sce_obj) {
  a <- assayNames(sce_obj)
  if ("counts" %in% a) {
    assert_raw_counts(sce_obj, "counts")
  } else if ("X" %in% a) {
    assert_raw_counts(sce_obj, "X")
    assay(sce_obj, "counts") <- assay(sce_obj, "X")
  } else {
    stop("Neither 'counts' nor 'X' assay present.")
  }
  assays(sce_obj) <- SimpleList(counts = assay(sce_obj, "counts"))
  mat <- assay(sce_obj, "counts")
  if (!inherits(mat, "dgCMatrix"))
    assay(sce_obj, "counts") <- as(mat, "dgCMatrix")
  sce_obj
}

# ============================================================
# DEMOGRAPHIC CLEANING
# ============================================================

clean_demographics <- function(sce_obj) {
  eth <- tolower(trimws(as.character(colData(sce_obj)[[BIN_COL]])))
  sex <- tolower(trimws(as.character(colData(sce_obj)[[SEX_COL]])))
  keep <- (!(eth %in% UNKNOWN_VALUES)) & (!is.na(eth)) &
          (!(sex %in% SEX_UNKNOWN))    & (!is.na(sex))
  sce_obj <- sce_obj[, keep]
  colData(sce_obj)[[BIN_COL]] <- as.character(
    tolower(trimws(as.character(colData(sce_obj)[[BIN_COL]]))))
  colData(sce_obj)[[SEX_COL]] <- as.character(
    tolower(trimws(as.character(colData(sce_obj)[[SEX_COL]]))))
  sce_obj
}

# ============================================================
# PRE-COMPUTE FULL-GENE LIBRARY SIZES
# ============================================================

heartbeat("Pre-computing full-gene library sizes...\n", TRUE)
sce_libsize_tmp <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_libsize_tmp <- ensure_counts_assay(sce_libsize_tmp)
LIBRARY_SIZES   <- Matrix::colSums(assay(sce_libsize_tmp, "counts"))
heartbeat(sprintf("Full-gene lib size: mean=%.0f sd=%.0f (%d cells)\n",
                  mean(LIBRARY_SIZES), sd(LIBRARY_SIZES), length(LIBRARY_SIZES)), TRUE)
rm(sce_libsize_tmp); gc()

# ============================================================
# COLDATA HELPERS
# ============================================================

sanitize_coldata_for_h5ad <- function(sce_obj) {
  n <- ncol(sce_obj); cd <- colData(sce_obj); rn <- rownames(cd)
  new_cols <- lapply(colnames(cd), function(col) {
    raw <- tryCatch(cd[[col]], error = function(e) rep(NA_character_, n))
    if (is.double(raw) && is.null(dim(raw)) && length(raw) == n) return(raw)
    if (is.integer(raw) && is.null(dim(raw)) && length(raw) == n) return(raw)
    result <- character(n)
    for (i in seq_len(n)) {
      xi <- tryCatch(raw[[i]], error = function(e) NA_character_)
      result[[i]] <- if (is.null(xi) || length(xi) == 0 ||
                         (length(xi) == 1 && is.na(xi))) NA_character_
                     else paste(as.character(xi), collapse = ";")
    }
    result
  })
  names(new_cols) <- colnames(cd)
  colData(sce_obj) <- DataFrame(new_cols, row.names = rn)
  sce_obj
}

rebuild_for_write <- function(sce_obj, drop_all_na = FALSE) {
  assay_nm <- if (ASSAY_USE %in% assayNames(sce_obj)) ASSAY_USE else assayNames(sce_obj)[1]
  cnt <- assay(sce_obj, assay_nm)
  if (!inherits(cnt, "dgCMatrix")) cnt <- as(cnt, "dgCMatrix")
  n <- ncol(sce_obj); cd_raw <- colData(sce_obj)
  col_list <- lapply(colnames(cd_raw), function(col) {
    v <- tryCatch({
      tmp <- cd_raw[[col]]
      vapply(seq_len(n), function(i) {
        xi <- tryCatch(tmp[[i]], error = function(e) NA)
        if (is.null(xi) || length(xi) == 0 || (length(xi) == 1 && is.na(xi)))
          return(NA_character_)
        paste(as.character(xi), collapse = ";")
      }, character(1))
    }, error = function(e) rep(NA_character_, n))
    num <- suppressWarnings(as.numeric(v))
    if (!any(is.na(num) & !is.na(v))) return(num)
    v
  })
  names(col_list) <- colnames(cd_raw)
  if (drop_all_na) {
    not_all_na <- vapply(col_list, function(v) !all(is.na(v)), logical(1))
    col_list   <- col_list[not_all_na]
  }
  rn_safe <- make.unique(as.character(colnames(sce_obj)), sep = "_dup")
  col_list_clean <- lapply(col_list, function(v) {
    v <- unclass(v); attributes(v) <- NULL; length(v) <- n; v
  })
  names(col_list_clean) <- names(col_list)
  cd_new <- structure(col_list_clean, class = "data.frame",
                      row.names = rn_safe, names = names(col_list_clean))
  colnames(cnt) <- rn_safe
  SingleCellExperiment(assays = list(counts = cnt), colData = DataFrame(cd_new))
}

# ============================================================
# 1. AUTO-DETECT GROUP STRUCTURE
# ============================================================

heartbeat("\n=== AUTO-DETECTING ETHNICITY GROUP STRUCTURE ===\n", TRUE)

sce_detect <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_detect <- ensure_counts_assay(sce_detect)
sce_detect <- exclude_validation_cells(sce_detect)
sce_detect <- clean_demographics(sce_detect)
sce_detect <- add_age_bins(sce_detect)

bin_table <- table(colData(sce_detect)[[BIN_COL]])
all_bins  <- sort(names(bin_table))

heartbeat("Real dataset ethnicity group sizes (validation-excluded):\n", TRUE)
for (b in all_bins) heartbeat(sprintf(" %s: %d cells", b, bin_table[b]), TRUE)

MAJ_BIN <- all_bins[which.max(bin_table)]
MIN_BIN <- all_bins[which.min(bin_table)]
heartbeat(sprintf(">>> MAJORITY: %s (%d cells)\n", MAJ_BIN, bin_table[MAJ_BIN]), TRUE)
heartbeat(sprintf(">>> MINORITY: %s (%d cells)\n", MIN_BIN, bin_table[MIN_BIN]), TRUE)

# ============================================================
# 2. COMPUTE PROPORTIONAL TARGETS
# ============================================================

heartbeat(sprintf("\n=== COMPUTING PROPORTIONAL TARGETS (%d cells total) ===\n",
                  PROPORTIONAL_SIZE), TRUE)

total_real <- sum(as.numeric(bin_table))
if (total_real < PROPORTIONAL_SIZE) {
  stop(sprintf("Not enough cells: have %d, need %d", total_real, PROPORTIONAL_SIZE))
}

prop_targets <- sapply(as.numeric(bin_table),
                       function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins

heartbeat("Proportional allocation:\n", TRUE)
for (b in all_bins) heartbeat(sprintf(" %s: %d cells", b, prop_targets[b]), TRUE)

TARGET_PER_BIN <- prop_targets[[MAJ_BIN]]
TARGET_DOWN    <- prop_targets[[MIN_BIN]]
if (TARGET_PER_BIN < TARGET_DOWN) {
  tmp <- TARGET_PER_BIN; TARGET_PER_BIN <- TARGET_DOWN; TARGET_DOWN <- tmp
}

heartbeat(sprintf(">>> TARGET_PER_BIN: %d (%s's share = ceiling for all groups)\n",
                  TARGET_PER_BIN, MAJ_BIN), TRUE)
heartbeat(sprintf(">>> TARGET_DOWN: %d (min group's real count)\n", TARGET_DOWN), TRUE)

heartbeat("Per-group synthesis plan:\n", TRUE)
for (b in all_bins) {
  n_real_b  <- prop_targets[[b]]
  n_synth_b <- if (b == MAJ_BIN) 0 else max(0, TARGET_PER_BIN - n_real_b)
  heartbeat(sprintf(" %s: %d real + %d synthetic = %d total",
                    b, n_real_b, n_synth_b, n_real_b + n_synth_b), TRUE)
}

if (!dir.exists(CKPTDIR)) dir.create(CKPTDIR, recursive = TRUE)

# ============================================================
# 3. HVG SELECTION  [V6 1, 4]
# ============================================================

heartbeat("\n=== HVG SELECTION (scran::modelGeneVar) ===\n", TRUE)

# [V6 4] use_hdf5=FALSE + realize to dgCMatrix to avoid DelayedMatrix transposition bug
sce_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_full <- ensure_counts_assay(sce_full)
sce_full <- exclude_validation_cells(sce_full)
sce_full <- clean_demographics(sce_full)
sce_full <- add_age_bins(sce_full)
assay(sce_full, ASSAY_USE) <- as(as.matrix(assay(sce_full, ASSAY_USE)), "dgCMatrix")
original_genes <- rownames(sce_full)
heartbeat(sprintf("Full cleaned dataset: %d cells x %d genes\n",
                  ncol(sce_full), nrow(sce_full)), TRUE)

bin_labels_full  <- colData(sce_full)[[BIN_COL]]
bin_counts_full  <- table(bin_labels_full)
target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
n_per_bin_hvg    <- pmin(as.numeric(bin_counts_full), target_per_bin_hvg)
names(n_per_bin_hvg) <- names(bin_counts_full)

set.seed(123)
sample_idx_hvg <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_full == b)
  if (length(idx) > n_per_bin_hvg[b]) sample(idx, n_per_bin_hvg[b]) else idx
}))

sce_sample <- sce_full[, sample_idx_hvg]
count_mat  <- assay(sce_sample, ASSAY_USE)
gene_detect_rate <- Matrix::rowMeans(count_mat > 0)
gene_sums        <- Matrix::rowSums(count_mat)
gene_nonzero     <- Matrix::rowSums(count_mat > 0)
keep_genes       <- (gene_sums >= MIN_COUNTS) &
                    (gene_nonzero >= MIN_CELLS) &
                    (gene_detect_rate >= MIN_DETECTION_RATE)
heartbeat(sprintf("Detection-rate + min-count filter: %d -> %d genes\n",
                  nrow(sce_sample), sum(keep_genes)), TRUE)

sce_filt <- sce_sample[keep_genes, ]
sce_filt <- scuttle::logNormCounts(sce_filt)
var_model <- scran::modelGeneVar(sce_filt, BPPARAM = BP)
hvg_names <- scran::getTopHVGs(var_model, n = N_HVG)
hvg_names <- intersect(hvg_names, rownames(sce_full))
heartbeat(sprintf("Selected %d HVGs via scran::modelGeneVar\n", length(hvg_names)), TRUE)

# Restrict HVGs to Geneformer vocabulary (same step0b output usable across
# scFoundation, Geneformer, scGPT). Source IDs are Ensembl without version.
gf_vocab_path <- "geneformer_vocab_genes.txt"
if (file.exists(gf_vocab_path)) {
  gf_vocab     <- readLines(gf_vocab_path)
  hvg_stripped <- sub("\\.[0-9]+$", "", hvg_names)
  in_vocab     <- hvg_stripped %in% gf_vocab
  hvg_names    <- hvg_names[in_vocab]
  heartbeat(sprintf("Geneformer vocab filter: %d / %d HVGs retained\n",
                    sum(in_vocab), length(in_vocab)), TRUE)
  if (length(hvg_names) < 100L) {
    stop(sprintf("FATAL: only %d HVGs in Geneformer vocab, need >=100",
                 length(hvg_names)))
  }
} else {
  heartbeat(sprintf("WARNING: geneformer_vocab_genes.txt not found, no vocab filter\n"), TRUE)
}

# Restrict HVGs to Geneformer vocabulary (same step0b output usable across
# scFoundation, Geneformer, scGPT). Source IDs are Ensembl without version.
gf_vocab_path <- "geneformer_vocab_genes.txt"
if (file.exists(gf_vocab_path)) {
  gf_vocab     <- readLines(gf_vocab_path)
  hvg_stripped <- sub("\\.[0-9]+$", "", hvg_names)
  in_vocab     <- hvg_stripped %in% gf_vocab
  hvg_names    <- hvg_names[in_vocab]
  heartbeat(sprintf("Geneformer vocab filter: %d / %d HVGs retained\n",
                    sum(in_vocab), length(in_vocab)), TRUE)
  if (length(hvg_names) < 100L) {
    stop(sprintf("FATAL: only %d HVGs in Geneformer vocab, need >=100",
                 length(hvg_names)))
  }
} else {
  heartbeat(sprintf("WARNING: geneformer_vocab_genes.txt not found, no vocab filter\n"), TRUE)
}

hvg_mean <- Matrix::rowMeans(assay(sce_filt, ASSAY_USE)[hvg_names, ])
heartbeat(sprintf("HVG mean expression: min=%.3f  median=%.3f  max=%.3f\n",
                  min(hvg_mean), median(hvg_mean), max(hvg_mean)), TRUE)

rm(sce_sample, sce_filt, count_mat, var_model); gc()

# ============================================================
# 4. LOAD BIN FRESH
# ============================================================

load_bin_fresh <- function(bin_label) {
  heartbeat(sprintf("Loading group '%s' (HVG subset)...", bin_label), TRUE)

  sce_raw <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
  common_early <- intersect(hvg_names, rownames(sce_raw))
  if (length(common_early) == 0)
    stop(sprintf("No HVGs found for %s", bin_label))
  sce_raw <- sce_raw[common_early, ]
  sce_raw <- ensure_counts_assay(sce_raw)
  sce_raw <- exclude_validation_cells(sce_raw)
  sce_raw <- clean_demographics(sce_raw)
  sce_raw <- add_age_bins(sce_raw)

  ct_raw   <- as.character(colData(sce_raw)[[CELLTYPE_COL]])
  valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
  sce_raw  <- sce_raw[, valid_ct]

  eth_vec  <- as.character(colData(sce_raw)[[BIN_COL]])
  sce_bin  <- sce_raw[, eth_vec == bin_label]
  if (ncol(sce_bin) == 0) stop(sprintf("No cells for '%s'", bin_label))

  common  <- intersect(hvg_names, rownames(sce_bin))
  sce_bin <- sce_bin[common, ]

  ct_vec  <- as.character(colData(sce_bin)[[CELLTYPE_COL]])
  tab_ct  <- table(ct_vec)
  keep_ct <- names(tab_ct)[tab_ct >= MIN_CT_CELLS]
  if (length(keep_ct) == 0)
    stop(sprintf("No cell types in '%s' with >= %d cells", bin_label, MIN_CT_CELLS))
  sce_bin <- sce_bin[, ct_vec %in% keep_ct]
  ct_vec  <- ct_vec[ct_vec %in% keep_ct]
  ct_vec[is.na(ct_vec)] <- "unknown_cleaned"
  colData(sce_bin)[[CELLTYPE_COL]] <- droplevels(factor(ct_vec))
  colData(sce_bin)[[BIN_COL]]      <- as.character(colData(sce_bin)[[BIN_COL]])
  colData(sce_bin)[[SEX_COL]]      <- as.character(colData(sce_bin)[[SEX_COL]])
  colData(sce_bin)[[AGE_BIN_COL]]  <- as.character(colData(sce_bin)[[AGE_BIN_COL]])

  # Full-gene library size by barcode lookup from pre-computed LIBRARY_SIZES
  lib_vec <- LIBRARY_SIZES[colnames(sce_bin)]
  lib_vec[is.na(lib_vec)] <- median(LIBRARY_SIZES, na.rm = TRUE)
  colData(sce_bin)$library <- as.numeric(lib_vec)

  heartbeat(sprintf(
    "  -> %d cells, %d CTs, lib size (full genes) mean=%.0f median=%.0f",
    ncol(sce_bin), length(unique(ct_vec)),
    mean(colData(sce_bin)$library), median(colData(sce_bin)$library)), TRUE)
  sce_bin
}

# ============================================================
# 5. BUILD MU FORMULA  [V6 12]
# ============================================================

build_mu_formula <- function(train_sce, use_fallback = FALSE) {
  # Default = additive (no interaction). Interaction term creates
  # n_ct * n_groups coefficient blocks per gene which makes mgcv::bam
  # fits prohibitively slow on this data. Set use_fallback=TRUE to
  # restore the original interaction formula.
  if (!use_fallback) {
    base <- "cell_type + self_reported_ethnicity"
  } else {
    base <- "cell_type + self_reported_ethnicity + cell_type:self_reported_ethnicity"
    heartbeat(" Using interaction formula (slower, may rank-deficient)\n", TRUE)
  }
  formula_parts <- c(base)
  other_extra   <- character(0)

  for (covar in c(SEX_COL, AGE_BIN_COL)) {
    if (!covar %in% colnames(colData(train_sce))) next
    n_levels <- length(unique(na.omit(as.character(colData(train_sce)[[covar]]))))
    if (n_levels < 2L) {
      heartbeat(sprintf(" '%s' has %d level(s) -> soft-degraded.\n", covar, n_levels), TRUE)
      next
    }
    formula_parts <- c(formula_parts, covar)
    other_extra   <- c(other_extra, covar)
  }

  formula_str  <- paste(paste(formula_parts, collapse = " + "),
                        "+ offset(log(library))")
  other_covars <- c(BIN_COL, other_extra, "library")
  heartbeat(sprintf(" mu_formula = %s\n", formula_str), TRUE)
  list(formula = formula_str, other_covariates = other_covars)
}

# ============================================================
# 6. VALIDATE NEW COUNTS
# ============================================================

validate_new_counts <- function(new_count, ref_genes, chunk_id, bin_label) {
  if (is.null(new_count)) stop("scDesign3 returned NULL")
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(new_count, nrow = length(new_count), ncol = 1,
                        dimnames = list(names(new_count), NULL))
  }
  if (!is.null(rownames(new_count)) && !is.null(ref_genes)) {
    if (ncol(new_count) == length(ref_genes) && nrow(new_count) != length(ref_genes))
      new_count <- t(new_count)
  }
  if (ncol(new_count) == 0) stop("Empty count matrix")
  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(0, nrow = length(missing), ncol = ncol(new_count),
                  dimnames = list(missing, colnames(new_count)))
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop = FALSE]
  colnames(new_count) <- paste0("synthetic_", bin_label, "_chunk", chunk_id,
                                "_", seq_len(ncol(new_count)))
  new_count
}

# ============================================================
# 7. ANCHOR CACHE  [V6 7]
# ============================================================

heartbeat(sprintf("\n=== PRE-CACHING ANCHOR CELLS (%d per CT per group) ===\n",
                  ANCHOR_PER_CT_PER_BIN), TRUE)
anchor_cache <- list()
for (b in all_bins) {
  tryCatch({
    sce_b    <- load_bin_fresh(b)
    ct_b     <- as.character(colData(sce_b)[[CELLTYPE_COL]])
    tab_b    <- table(ct_b)
    anch_idx <- unlist(lapply(names(tab_b), function(ct) {
      ids <- which(ct_b == ct)
      sample(ids, min(ANCHOR_PER_CT_PER_BIN, length(ids)))
    }))
    anchor_cache[[b]] <- sce_b[, anch_idx]
    heartbeat(sprintf(" Cached %d anchor cells for '%s' (%d CTs)\n",
                      length(anch_idx), b, length(tab_b)), TRUE)
    rm(sce_b); gc()
  }, error = function(e) {
    heartbeat(sprintf(" WARNING: anchor cache FAILED for '%s': %s\n",
                      b, conditionMessage(e)), TRUE)
  })
}

build_joint_train <- function(sub_train_target, target_bin) {
  other_anchors <- anchor_cache[setdiff(names(anchor_cache), target_bin)]
  if (length(other_anchors) == 0) return(sub_train_target)

  target_cts <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
  other_anchors <- lapply(other_anchors, function(a) {
    ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
    a[, ct_a %in% target_cts, drop = FALSE]
  })
  other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]
  if (length(other_anchors) == 0) return(sub_train_target)

  common_genes <- rownames(sub_train_target)
  for (sce_a in other_anchors) common_genes <- intersect(common_genes, rownames(sce_a))
  if (length(common_genes) == 0) return(sub_train_target)

  pieces <- c(list(sub_train_target[common_genes, ]),
              lapply(other_anchors, function(a) a[common_genes, ]))
  joint  <- do.call(cbind, pieces)

  if (!"library" %in% colnames(colData(joint)) ||
      any(is.na(colData(joint)$library))) {
    colData(joint)$library <- Matrix::colSums(assay(joint, ASSAY_USE))
  }

  ct_joint  <- as.character(colData(joint)[[CELLTYPE_COL]])
  tab_joint <- table(ct_joint)
  keep_ct   <- names(tab_joint)[tab_joint >= 2L]
  if (length(keep_ct) < length(tab_joint)) {
    joint    <- joint[, ct_joint %in% keep_ct]
    ct_clean <- as.character(colData(joint)[[CELLTYPE_COL]])
    colData(joint)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
  }

  colData(joint)[[BIN_COL]]     <- as.character(colData(joint)[[BIN_COL]])
  colData(joint)[[SEX_COL]]     <- as.character(colData(joint)[[SEX_COL]])
  colData(joint)[[AGE_BIN_COL]] <- as.character(colData(joint)[[AGE_BIN_COL]])

  n_target <- ncol(sub_train_target)
  n_anchor <- ncol(joint) - n_target
  heartbeat(sprintf(" Joint: %d target + %d anchor = %d total, %d groups, %d CTs\n",
                    n_target, n_anchor, ncol(joint),
                    length(unique(as.character(colData(joint)[[BIN_COL]]))),
                    length(unique(as.character(colData(joint)[[CELLTYPE_COL]])))), TRUE)
  joint
}

# ============================================================
# 8. RUN scDESIGN3 (low-level)  [V6 2, 3, 5, 6, 13]
# ============================================================

run_sd3_lowlevel <- function(sub_train, sub_train_target, bin_label, chunk,
                             use_fallback_formula = FALSE) {
  tryCatch({
    mu_spec <- build_mu_formula(sub_train, use_fallback = use_fallback_formula)

    joint_data <- construct_data(
      sce              = sub_train,
      assay_use        = ASSAY_USE,
      celltype         = CELLTYPE_COL,
      pseudotime       = NULL,
      spatial          = NULL,
      other_covariates = mu_spec$other_covariates,
      ncell            = ncol(sub_train),
      corr_by          = "cell_type",
      parallelization  = "mcmapply",
      BPPARAM          = NULL
    )

    marginal_res <- fit_marginal(
      data            = joint_data,
      predictor       = "gene",
      mu_formula      = mu_spec$formula,
      sigma_formula   = "cell_type",
      family_use      = FAMILY_USE,
      n_cores         = N_CORES,
      usebam          = USE_BAM,
      edf_flexible    = EDF_FLEX,
      parallelization = "mcmapply",
      BPPARAM         = NULL
    )

    # [V6 3] Explicit logical vector named by gene
    n_target_cells     <- ncol(sub_train_target)
    mat_for_imp        <- assay(sub_train, ASSAY_USE)
    zero_frac          <- Matrix::rowMeans(mat_for_imp == 0)
    sparsity_threshold <- if (n_target_cells < MIN_CELLS_INTERACTION) 0.95 else 0.80
    imp_feat_vec       <- zero_frac <= sparsity_threshold
    if (sum(imp_feat_vec) < 2L) imp_feat_vec <- rep(TRUE, length(imp_feat_vec))
    names(imp_feat_vec) <- rownames(sub_train)
    heartbeat(sprintf(" important_feature: %d / %d genes kept for copula (threshold=%.2f)\n",
                      sum(imp_feat_vec), length(imp_feat_vec), sparsity_threshold), TRUE)

    copula_res <- fit_copula(
      sce               = sub_train,
      assay_use         = ASSAY_USE,
      marginal_list     = marginal_res,
      family_use        = FAMILY_USE,
      copula            = COPULA_TYPE,
      DT                = TRUE,
      pseudo_obs        = FALSE,
      n_cores           = N_CORES,
      input_data        = joint_data$dat,
      important_feature = imp_feat_vec,
      if_sparse         = FALSE,
      parallelization   = "mcmapply",
      BPPARAM           = NULL
    )

    target_dat_rows <- which(
      as.character(joint_data$dat[[BIN_COL]]) == bin_label)
    if (length(target_dat_rows) == 0)
      stop(sprintf("No target-group rows in joint_data$dat for '%s'", bin_label))

    new_cov_idx <- sample(target_dat_rows, chunk,
                          replace = (length(target_dat_rows) < chunk))
    new_cov_df  <- joint_data$dat[new_cov_idx, , drop = FALSE]

    if (!"library" %in% colnames(new_cov_df))
      new_cov_df$library <- median(colData(sub_train)$library)
    if (!SEX_COL %in% colnames(new_cov_df))
      new_cov_df[[SEX_COL]] <- names(sort(table(
        as.character(colData(sub_train_target)[[SEX_COL]])), decreasing = TRUE))[1]
    if (!AGE_BIN_COL %in% colnames(new_cov_df))
      new_cov_df[[AGE_BIN_COL]] <- names(sort(table(
        as.character(colData(sub_train_target)[[AGE_BIN_COL]])), decreasing = TRUE))[1]

    cov_cts    <- as.character(new_cov_df[[CELLTYPE_COL]])
    ct_tab_cov <- table(cov_cts)
    thin_cts   <- names(ct_tab_cov)[ct_tab_cov < 2L]
    if (length(thin_cts) > 0) {
      extra_rows <- lapply(thin_cts, function(ct) {
        ct_rows <- target_dat_rows[
          as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct]
        if (length(ct_rows) == 0L) return(NULL)
        joint_data$dat[sample(ct_rows, 2L - ct_tab_cov[[ct]], replace = TRUE), ]
      })
      extra_rows <- do.call(rbind, Filter(Negate(is.null), extra_rows))
      if (!is.null(extra_rows) && nrow(extra_rows) > 0)
        new_cov_df <- rbind(new_cov_df, extra_rows)
    }

    new_cov_df$corr_group <- as.character(new_cov_df[[CELLTYPE_COL]])
    rownames(new_cov_df)  <- paste0("Cell", seq_len(nrow(new_cov_df)))

    para_list <- extract_para(
      sce             = sub_train,
      assay_use       = ASSAY_USE,
      marginal_list   = marginal_res,
      n_cores         = N_CORES,
      family_use      = FAMILY_USE,
      new_covariate   = new_cov_df,
      parallelization = "mcmapply",
      BPPARAM         = NULL,
      data            = joint_data$dat
    )

    new_count <- simu_new(
      sce               = sub_train,
      assay_use         = ASSAY_USE,
      mean_mat          = para_list$mean_mat,
      sigma_mat         = para_list$sigma_mat,
      zero_mat          = para_list$zero_mat,
      quantile_mat      = NULL,
      copula_list       = copula_res$copula_list,
      n_cores           = N_CORES,
      family_use        = FAMILY_USE,
      nonnegative       = TRUE,
      nonzerovar        = TRUE,
      input_data        = joint_data$dat,
      new_covariate     = new_cov_df,
      important_feature = imp_feat_vec,
      parallelization   = "mcmapply",
      BPPARAM           = NULL,
      filtered_gene     = joint_data$filtered_gene
    )

    list(new_count = new_count, new_covariate = new_cov_df)
  }, error = function(e) {
    heartbeat(sprintf(" ERROR: %s\n", conditionMessage(e)), TRUE); NULL
  })
}

# ============================================================
# 9. AUGMENT ONE GROUP
# ============================================================

augment_one_bin <- function(bin_label) {
  ckpt_file <- file.path(CKPTDIR,
    paste0("ethnicity_", gsub("[^a-zA-Z0-9_]", "_", bin_label), ".rds"))

  if (file.exists(ckpt_file)) {
    heartbeat(sprintf("Checkpoint found for '%s' -- loading.\n", bin_label), TRUE)
    return(sanitize_coldata_for_h5ad(readRDS(ckpt_file)))
  }

  sub_real_full <- load_bin_fresh(bin_label)
  n_full        <- ncol(sub_real_full)

  if (bin_label == MAJ_BIN) {
    if (n_full > TARGET_PER_BIN) {
      heartbeat(sprintf("Majority '%s': downsampling %d -> %d.\n",
                        bin_label, n_full, TARGET_PER_BIN), TRUE)
      set.seed(123)
      sub_real <- sub_real_full[, sample(n_full, TARGET_PER_BIN)]
    } else {
      sub_real <- sub_real_full
    }
    assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  # Minority: downsample to natural proportional share, then augment up to TARGET_PER_BIN
  target_real_min <- prop_targets[[bin_label]]
  if (is.na(target_real_min) || target_real_min <= 0)
    target_real_min <- min(n_full, TARGET_PER_BIN)

  sub_real <- if (n_full > target_real_min) {
    set.seed(123)
    sub_real_full[, sample(n_full, target_real_min)]
  } else {
    sub_real_full
  }
  assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")

  n_real <- ncol(sub_real)
  need   <- TARGET_PER_BIN - n_real
  if (need <= 0) {
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  heartbeat(sprintf("Augmenting '%s': %d -> %d (need %d synthetic)\n",
                    bin_label, n_real, TARGET_PER_BIN, need), TRUE)

  sims      <- list()
  remaining <- need
  chunk_id  <- 1L

  while (remaining > 0) {
    chunk   <- min(CHUNK_SIZE_DEFAULT, remaining)
    attempt <- 1L
    success <- FALSE

    while (attempt <= MAX_RETRIES && !success) {
      # [V6 10] Aggressive gc at top of each retry
      gc(reset = TRUE, full = TRUE)
      heartbeat(sprintf("Chunk %d | attempt %d | n=%d | remaining=%d  mem_used=%.1fGB\n",
                        chunk_id, attempt, chunk, remaining,
                        sum(gc()[, 2]) / 1024), TRUE)

      sub_train      <- load_bin_fresh(bin_label)
      n_train_loaded <- ncol(sub_train)

      n_ct_train <- length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
      if (chunk < 2L * n_ct_train) chunk <- 2L * n_ct_train

      if (n_train_loaded > MAX_CELLS_PER_BIN) {
        ct_vec <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        tab_ct <- table(ct_vec[ct_vec != ""])
        alloc  <- pmax(2L, as.integer(floor(MAX_CELLS_PER_BIN *
                        as.numeric(tab_ct) / sum(tab_ct))))
        names(alloc) <- names(tab_ct)
        alloc <- pmin(alloc, as.integer(tab_ct))
        keep_idx  <- unlist(lapply(names(alloc), function(ct) {
          ids <- which(ct_vec == ct)
          sample(ids, alloc[[ct]], replace = FALSE)
        }))
        sub_train <- sub_train[, keep_idx, drop = FALSE]
        ct_new    <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        ct_new[is.na(ct_new)] <- "unknown_cleaned"
        colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_new))
      }

      ct_final <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      ct_final[is.na(ct_final)] <- "unknown_cleaned"
      colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_final))
      if (any(table(as.character(colData(sub_train)[[CELLTYPE_COL]])) < 2L))
        stop("Training set has CT with <2 cells")

      sub_train_target <- sub_train
      sub_train        <- build_joint_train(sub_train_target, bin_label)

      eth_vals <- unique(as.character(colData(sub_train)[[BIN_COL]]))
      if (length(eth_vals) < 2L)
        stop(sprintf("[Tier 1 gate] No ethnicity variance for '%s'.", bin_label))

      ct_eth_tab <- table(
        CT  = as.character(colData(sub_train)[[CELLTYPE_COL]]),
        Grp = as.character(colData(sub_train)[[BIN_COL]])
      )
      ct_n_grps <- rowSums(ct_eth_tab > 0)
      single_ct <- names(ct_n_grps)[ct_n_grps < 2L]
      if (length(single_ct) > 0)
        stop(sprintf("[Tier 1 gate] CT(s) in only 1 ethnicity group: %s",
                     paste(single_ct, collapse = ", ")))

      heartbeat(" [Tier 1] running low-level scDesign3 with v6 config\n", TRUE)
      res <- run_sd3_lowlevel(sub_train, sub_train_target, bin_label, chunk,
                              use_fallback_formula = FALSE)

      # [V6 13] fallback on interaction failure
      if (is.null(res)) {
        heartbeat(" Interaction formula failed; retrying with additive fallback\n", TRUE)
        res <- run_sd3_lowlevel(sub_train, sub_train_target, bin_label, chunk,
                                use_fallback_formula = TRUE)
      }

      if (is.null(res)) {
        heartbeat(sprintf(" [Tier 1 FAILED] '%s' chunk %d attempt %d\n",
                          bin_label, chunk_id, attempt), TRUE)
        attempt <- attempt + 1L
        if (attempt > MAX_RETRIES) break
        next
      }

      new_counts <- validate_new_counts(res$new_count, rownames(sub_real),
                                        chunk_id, bin_label)
      new_cov    <- res$new_covariate
      if (is.null(new_cov)) new_cov <- DataFrame(row.names = colnames(new_counts))
      else rownames(new_cov) <- colnames(new_counts)

      sim <- SingleCellExperiment(assays = list(counts = new_counts),
                                  colData = new_cov)

      # [V6 9] QC gate: compare synthetic to real on active-genes and library size
      syn_active <- Matrix::colSums(as(assay(sim, ASSAY_USE), "dgCMatrix") > 0)
      syn_lib    <- Matrix::colSums(as(assay(sim, ASSAY_USE), "dgCMatrix"))
      med_active_syn <- median(syn_active)
      med_lib_syn    <- median(syn_lib)
      real_active_vec <- Matrix::colSums(as(assay(sub_real, ASSAY_USE), "dgCMatrix") > 0)
      real_lib_vec    <- Matrix::colSums(as(assay(sub_real, ASSAY_USE), "dgCMatrix"))
      med_active_real <- median(real_active_vec)
      med_lib_real    <- median(real_lib_vec)
      heartbeat(sprintf(" QC synthetic HVG-space: active med=%.0f (real %.0f)  lib med=%.0f (real %.0f)\n",
                        med_active_syn, med_active_real, med_lib_syn, med_lib_real), TRUE)
      if (med_active_syn < 0.5 * med_active_real || med_lib_syn < 0.5 * med_lib_real) {
        heartbeat(sprintf(" QC FAIL: synthetic < 50%% of real. Retrying attempt %d\n",
                          attempt + 1L), TRUE)
        attempt <- attempt + 1L
        if (attempt > MAX_RETRIES) break
        next
      }
      heartbeat(" QC PASS\n", TRUE)

      ref_cols     <- colnames(colData(sub_real))
      missing_cols <- setdiff(ref_cols, colnames(colData(sim)))
      for (m in missing_cols) {
        ref_val <- colData(sub_real)[[m]]
        colData(sim)[[m]] <- if (is.list(ref_val))    rep(NA_character_, ncol(sim))
                             else if (is.double(ref_val))  rep(NA_real_, ncol(sim))
                             else if (is.integer(ref_val)) rep(NA_integer_, ncol(sim))
                             else                          rep(NA_character_, ncol(sim))
      }
      colData(sim) <- colData(sim)[, ref_cols, drop = FALSE]
      colData(sim)[[BIN_COL]] <- rep(bin_label, ncol(sim))

      sims[[length(sims) + 1L]] <- sim
      success <- TRUE
      heartbeat(sprintf(" Chunk %d done: %d cells generated\n", chunk_id, ncol(sim)), TRUE)

      rm(res, sim, new_counts, new_cov, sub_train, sub_train_target)
      gc(reset = TRUE, full = TRUE)
      heartbeat(sprintf(" post-chunk gc: mem_used=%.1fGB\n",
                        sum(gc()[, 2]) / 1024), TRUE)
    }

    if (!success) {
      heartbeat(sprintf("FAILED after %d attempts -- naive resampling fallback\n",
                        MAX_RETRIES), TRUE)
      set.seed(123)
      resample_idx <- sample.int(ncol(sub_real), remaining, replace = TRUE)
      res_counts   <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")[, resample_idx]
      colnames(res_counts) <- paste0("synthetic_resample_", bin_label,
                                     "_", seq_len(ncol(res_counts)))
      res_cov <- as.data.frame(colData(sub_real))[resample_idx, ]
      rownames(res_cov) <- colnames(res_counts)
      sim_fallback <- SingleCellExperiment(
        assays  = list(counts = res_counts),
        colData = DataFrame(res_cov))
      colData(sim_fallback)[[BIN_COL]] <- rep(bin_label, ncol(sim_fallback))
      sims[[length(sims) + 1L]] <- sim_fallback
      remaining <- 0L
      break
    }

    remaining <- remaining - chunk
    chunk_id  <- chunk_id + 1L
  }

  out_bin <- do.call(cbind, c(list(sub_real), sims))
  heartbeat(sprintf("Group '%s' final size: %d cells\n", bin_label, ncol(out_bin)), TRUE)
  saveRDS(out_bin, ckpt_file)
  out_bin
}

# ============================================================
# 10. PROCESS ALL GROUPS
# ============================================================

heartbeat("\n=== PROCESSING ALL ETHNICITY GROUPS ===\n", TRUE)
bin_objects <- list()
for (b in all_bins) {
  heartbeat(sprintf("\n=== PROCESSING: %s ===\n", b), TRUE)
  bin_objects[[b]] <- augment_one_bin(b)
}

# ============================================================
# 11. COMBINE AND RESTORE FULL GENE SET
# ============================================================

heartbeat("\n=== COMBINING ALL GROUPS ===\n", TRUE)
sce_combined <- do.call(cbind, bin_objects)
heartbeat(sprintf("Combined: %d cells x %d genes\n",
                  ncol(sce_combined), nrow(sce_combined)), TRUE)

present_genes <- rownames(sce_combined)
missing_genes <- setdiff(original_genes, present_genes)
current_counts <- as(assay(sce_combined, ASSAY_USE), "dgCMatrix")

if (length(missing_genes) > 0) {
  pad_mat <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(missing_genes), ncol(sce_combined)),
    dimnames = list(missing_genes, colnames(sce_combined)))
  full_counts <- rbind(current_counts, pad_mat)
} else {
  full_counts <- current_counts
}
full_counts <- as(full_counts, "dgCMatrix")
gene_order  <- match(original_genes, rownames(full_counts))
valid_order <- gene_order[!is.na(gene_order)]
full_counts <- full_counts[valid_order, , drop = FALSE]
rownames(full_counts) <- original_genes[!is.na(gene_order)]

# [V6 11] Restore real cells' full-gene counts from source
heartbeat("\n=== RESTORING FULL-GENE COUNTS FOR REAL CELLS ===\n", TRUE)
source_names <- colnames(sce_full)
src_counts   <- as(as.matrix(assay(sce_full, ASSAY_USE)), "dgCMatrix")
combined_names <- colnames(sce_combined)
is_synthetic   <- grepl("^synthetic_", combined_names)
real_positions <- which(!is_synthetic)
real_names     <- combined_names[real_positions]
src_lookup     <- match(real_names, source_names)
found_mask     <- !is.na(src_lookup)
heartbeat(sprintf("Real cells found in source: %d/%d\n",
                  sum(found_mask), length(real_positions)), TRUE)
if (sum(found_mask) > 0) {
  src_cols_sub <- src_counts[, src_lookup[found_mask], drop = FALSE]
  src_gene_idx <- match(original_genes, rownames(src_counts))
  src_cols_reordered <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(original_genes), ncol(src_cols_sub)),
    dimnames = list(original_genes, colnames(src_cols_sub)))
  valid_idx <- !is.na(src_gene_idx)
  src_cols_reordered[valid_idx, ] <- src_cols_sub[src_gene_idx[valid_idx], ]
  target_cols <- real_positions[found_mask]
  full_counts[, target_cols] <- src_cols_reordered
  heartbeat(" Restored full-gene counts for real cells\n", TRUE)
}

cd      <- colData(sce_combined)
rd_full <- rowData(sce_full)
rd_genes_present <- intersect(original_genes, rownames(rd_full))
rd_subset <- if (length(rd_genes_present) > 0) rd_full[rd_genes_present, , drop = FALSE] else rd_full[integer(0), , drop = FALSE]
rd_genes_missing <- setdiff(original_genes, rownames(rd_full))
if (length(rd_genes_missing) > 0) {
  rd_pad <- as.data.frame(matrix(NA, nrow = length(rd_genes_missing),
                                 ncol = ncol(rd_subset),
                                 dimnames = list(rd_genes_missing, colnames(rd_subset))))
  rd_combined <- rbind(as.data.frame(rd_subset), rd_pad)
} else {
  rd_combined <- as.data.frame(rd_subset)
}
rd_combined <- rd_combined[original_genes, , drop = FALSE]

sce_combined <- SingleCellExperiment(
  assays  = list(counts = full_counts),
  colData = cd,
  rowData = DataFrame(rd_combined))
rownames(sce_combined) <- original_genes
rm(sce_full, src_counts); gc()

# ============================================================
# 12. SOURCE COLUMN + SANITIZATION
# ============================================================

colData(sce_combined)$source <- ifelse(
  grepl("^synthetic_", colnames(sce_combined)), "synthetic", "real")
sce_combined <- sanitize_coldata_for_h5ad(sce_combined)

# ============================================================
# 13. BASE OUTPUT FILES
# ============================================================

heartbeat("\n=== GENERATING BASE OUTPUT FILES ===\n", TRUE)

out_full <- paste0(OUTPUT_BASE, "_Full_BalancedAugmented_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined, drop_all_na = FALSE),
                         out_full, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_full), TRUE)

sce_real <- sce_combined[, colData(sce_combined)$source == "real"]
out_real <- paste0(OUTPUT_BASE, "_RealOnly_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_real, drop_all_na = FALSE),
                         out_real, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_real), TRUE)

sce_syn <- sce_combined[, colData(sce_combined)$source == "synthetic"]
out_syn <- paste0(OUTPUT_BASE, "_SyntheticOnly_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_syn, drop_all_na = TRUE),
                         out_syn, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_syn), TRUE)

out_sum <- paste0(OUTPUT_BASE, "_Summary_ETHNICITY.csv")
summary_df <- as.data.frame(table(
  Ethnicity = colData(sce_combined)[[BIN_COL]],
  Source    = colData(sce_combined)$source))
colnames(summary_df) <- c("Ethnicity", "Source", "Count")
write.csv(summary_df, out_sum, row.names = FALSE)
heartbeat(sprintf("Wrote: %s\n", out_sum), TRUE)

# ============================================================
# 14. FAIRNESS DATASETS
# ============================================================

heartbeat("\n=== GENERATING FAIRNESS DATASETS ===\n", TRUE)

sample_exact <- function(idx, n, replace = FALSE) {
  n <- as.integer(round(as.numeric(n)))
  if (is.na(n) || n < 0) stop("invalid n")
  if (n == 0) return(integer(0))
  if (!replace && length(idx) < n) stop("not enough cells")
  sample(idx, n, replace = replace)
}

# 1) Proportional (real only, natural proportions)
bin_labels_det  <- colData(sce_detect)[[BIN_COL]]
set.seed(123)
prop_sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_det == b)
  n_b <- prop_targets[[b]]
  if (n_b <= 0 || length(idx) == 0) integer(0) else sample(idx, n_b)
}))

sce_for_prop <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_for_prop <- ensure_counts_assay(sce_for_prop)
sce_for_prop <- exclude_validation_cells(sce_for_prop)
sce_for_prop <- clean_demographics(sce_for_prop)
sce_for_prop <- add_age_bins(sce_for_prop)
sce_prop     <- sce_for_prop[, prop_sample_idx]
if (sum(assay(sce_prop, ASSAY_USE)) == 0)
  stop("FATAL: Proportional count matrix is all zeros.")

out_prop <- paste0(OUTPUT_BASE, "_Proportional_", sum(prop_targets), "_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_prop), out_prop, compression = "gzip")
heartbeat(sprintf("Wrote Proportional: %s\n", out_prop), TRUE)
rm(sce_for_prop); gc()

# 2) BalancedAugmented (alias of full combined)
out_bal_aug <- paste0(OUTPUT_BASE, "_BalancedAugmented_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined), out_bal_aug, compression = "gzip")
heartbeat(sprintf("Wrote BalancedAugmented: %s\n", out_bal_aug), TRUE)

# 3) BalancedUpsampled (real cells only, duplicated with replacement)
bin_real_vec <- tolower(trimws(as.character(colData(sce_real)[[BIN_COL]])))
up_idx <- unlist(lapply(all_bins, function(b) {
  sample_exact(which(bin_real_vec == b), TARGET_PER_BIN, replace = TRUE)
}))
sce_up <- sce_real[, up_idx]
out_up <- paste0(OUTPUT_BASE, "_BalancedUpsampled_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_up), out_up, compression = "gzip")
heartbeat(sprintf("Wrote BalancedUpsampled: %s\n", out_up), TRUE)

# 4) Downsampled (real cells only, every group to smallest real group count)
bin_real_tbl <- table(bin_real_vec)
DOWN_TARGET  <- as.integer(min(bin_real_tbl))
heartbeat(sprintf("Downsampled target = %d (smallest real group: %s)\n",
                  DOWN_TARGET, names(bin_real_tbl)[which.min(bin_real_tbl)]), TRUE)
down_idx <- unlist(lapply(all_bins, function(b) {
  sample_exact(which(bin_real_vec == b), DOWN_TARGET, replace = FALSE)
}))
sce_down <- sce_real[, down_idx]
out_down <- paste0(OUTPUT_BASE, "_Downsampled_", DOWN_TARGET, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_down), out_down, compression = "gzip")
heartbeat(sprintf("Wrote Downsampled: %s\n", out_down), TRUE)

# ============================================================
# 15. FINAL REPORT
# ============================================================

heartbeat("\n=== ETHNICITY AUGMENTATION V6 COMPLETE ===\n", TRUE)
heartbeat("Output files:\n", TRUE)
heartbeat(sprintf(" 1. %s  (Full BalancedAugmented with real+synthetic)\n", out_full), TRUE)
heartbeat(sprintf(" 2. %s  (RealOnly - all real cells from pilot)\n", out_real), TRUE)
heartbeat(sprintf(" 3. %s  (SyntheticOnly - all synthetic cells)\n", out_syn), TRUE)
heartbeat(sprintf(" 4. %s  (group-by-source count summary)\n", out_sum), TRUE)
heartbeat(sprintf(" 5. %s  (Proportional, real only, natural proportions)\n", out_prop), TRUE)
heartbeat(sprintf(" 6. %s  (BalancedAugmented, real+synthetic to %d each)\n",
                  out_bal_aug, TARGET_PER_BIN), TRUE)
heartbeat(sprintf(" 7. %s  (BalancedUpsampled, real with replacement)\n", out_up), TRUE)
heartbeat(sprintf(" 8. %s  (Downsampled to %d each, real only)\n", out_down, DOWN_TARGET), TRUE)
heartbeat(sprintf("\nConfig: PROPORTIONAL_SIZE=%d TARGET_PER_BIN=%d\n",
                  PROPORTIONAL_SIZE, TARGET_PER_BIN), TRUE)
heartbeat(sprintf("family_use=%s copula=%s usebam=%s N_CORES=%d\n",
                  FAMILY_USE, COPULA_TYPE, USE_BAM, N_CORES), TRUE)
