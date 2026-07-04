#!/usr/bin/env Rscript
# ============================================================
# STEP 0B: AGE AUGMENTATION WITH scDesign3 (PILOT)
# ============================================================
# * Group by 10-year age bins derived from "development_stage"
#   (e.g., "45-year-old human stage" → "40_49")
# * Majority bin : real-only, down-sampled to TARGET_PER_BIN
# * Minority bins: real down-sampled proportionally, then
#   scDesign3-augmented up to TARGET_PER_BIN
# * 1,000 HVGs, stratified training sets
# * Preferred model:
#   - family_use = "zinb"          (was "nb", [V5 5])
#   - assay = "counts" (raw integer counts ONLY)
#   - mu_formula via build_mu_formula() [V5 6/7/8]:
#       "cell_type + age_bin_10yr + cell_type:age_bin_10yr
#        + sex + self_reported_ethnicity + offset(log(library))"
#       sex / self_reported_ethnicity dropped if <2 levels (soft-degrade)
#   - sigma_formula = "cell_type"
#   - corr_formula  = "cell_type"
#   - copula        = "gaussian"
#
# * ANCHOR-CELL ARCHITECTURE [FIX 17]:
#   scDesign3 trains per age bin, so age_bin_10yr would be constant
#   (zero variance) within a single bin -> copula crash.
#   Fix: pre-cache CT-stratified anchor cells from every other age bin
#   once before the augmentation loop, then inject them into each
#   per-bin training set via build_joint_train(). This gives scDesign3
#   real variance in age_bin_10yr so the age covariate can be fitted.
#   NOTE: new_covariate is NOT passed to scdesign3() (unsupported in
#   this version). Age identity on synthetic cells is stamped post-hoc
#   via FIX 15 (colData[[BIN_COL]] <- bin_label).
#
# * FORMULA TIER LADDER (applied per bin, most to least complex):
#   Tier 1: mu  = "cell_type + age_bin_10yr + cell_type:age_bin_10yr
#                  + sex + self_reported_ethnicity + offset(log(library))"
#           corr= "cell_type"  (requires has_age_var AND n_target >= 50)
#   Soft-degrade: sex/self_reported_ethnicity dropped if <2 levels [V5 8]
#
# * Fallback: if scDesign3 still fails after MAX_RETRIES,
#   fill remaining cells by naive resampling of real cells.
# * All SCEs normalized to a single assay: "counts"
#
# ============================================================
# CHANGES vs. ETHNICITY PILOT
# ============================================================
#   [AGE 1]  AGE_COL / BIN_COL:
#              BIN_COL="self_reported_ethnicity"
#              -> AGE_COL="development_stage" (source text col)
#                BIN_COL="age_bin_10yr"      (derived, 10yr bins)
#   [AGE 2]  add_age_bins() helper added:
#              Parses numeric age from development_stage text,
#              cuts into bins 10_19, 20_29, ..., 80_89 (10yr wide)
#              using underscore labels to avoid special chars.
#              Called after every readH5AD() call.
#   [AGE 3]  INPUT_H5AD   : ...ETHNICITY.h5ad -> ...AGE.h5ad
#   [AGE 4]  OUTPUT_BASE  : ILD_Ethnicity_Pilot -> AIDA_Age_Pilot
#   [AGE 5]  CKPTDIR      : checkpoints_ethnicity_... -> checkpoints_age_...
#   [AGE 6]  LOGFILE      : augmentation_ethnicity_... -> augmentation_age_...
#   [AGE 7]  UNKNOWN_VALUES: age-specific noise ("unknown", "adult", etc.)
#   [AGE 8]  All _ETHNICITY file suffixes -> _AGE
#   [AGE 9]  All "ethnicity" log strings -> "age bin"
#   [AGE 10] PROPORTIONAL_SIZE = 2500 (pilot; ~250 per bin for 8-bin case)
#            MAX_CELLS_PER_BIN = 500  (pilot)
#
# All FIX patches (1-22) from prior scripts preserved verbatim.
# V5 patches added in this version:
#   [V5 1]  VALIDATION_H5AD constant: AIDA_Age_External_Validation_10000.h5ad
#   [V5 2]  load_validation_barcodes() + exclude_validation_cells() helpers;
#           VALIDATION_BARCODES global populated at startup.
#   [V5 3]  exclude_validation_cells() called after every readH5AD(INPUT_H5AD)
#           (5 sites: sce_detect, sce_full, sce_sample, load_bin_fresh, sce_for_prop).
#   [V5 4]  LIBRARY_SIZES global: full 31,432-gene colSums pre-computed once.
#           load_bin_fresh() attaches colData$library by barcode lookup.
#   [V5 5]  family_use = "zinb" at all 4 sites in run_sd3_lowlevel()
#           (fit_marginal, fit_copula, extract_para, simu_new).
#   [V5 6]  mu_formula extended: + sex + self_reported_ethnicity + offset(log(library))
#   [V5 7]  construct_data other_covariates extended: c(BIN_COL, "sex",
#           "self_reported_ethnicity", "library") — soft-degraded by build_mu_formula.
#   [V5 8]  build_mu_formula() soft-degrade helper: drops sex and/or
#           self_reported_ethnicity from formula + other_covariates if <2 levels
#           in the joint training set. Always includes library.
# ============================================================

cat("\n=== SLURM environment ===\n")
cat("SLURM_CPUS_PER_TASK =", Sys.getenv("SLURM_CPUS_PER_TASK"), "\n")
cat("SLURM_JOB_ID =", Sys.getenv("SLURM_JOB_ID"), "\n")
cat("SLURM_NTASKS =", Sys.getenv("SLURM_NTASKS"), "\n")
cat("SLURM_CPUS_PER_TASK (numeric) =",
    as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")), "\n")

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDesign3)
  library(zellkonverter)
  library(Matrix)
  library(BiocParallel)
  library(RhpcBLASctl)
})

# ------------------------------------------------------------
# PARALLELIZATION (serial - 1 core)
# ------------------------------------------------------------

N_CORES <- 1L
cat(sprintf("Parallelization disabled: using %d core (serial)\n", N_CORES),
    file = stderr())
blas_set_num_threads(N_CORES)
message("[INFO] OpenBLAS/MKL now using ", N_CORES, " threads")
register(SerialParam(progressbar = FALSE))

# ============================================================
# CONFIGURATION  [AGE 1-10]
# ============================================================

INPUT_H5AD      <- "AIDA_RawCounts_AGE.h5ad"   # [AGE 3]
VALIDATION_H5AD <- "AIDA_Age_External_Validation_10000.h5ad"       # [V5 1]
OUTPUT_BASE     <- "AIDA_Age_Pilot"                                 # [AGE 4]
CKPTDIR         <- "checkpoints_age_augmentation_pilot_anchors"    # [FIX 17]
LOGFILE         <- "augmentation_age_pilot_anchors_log.txt"        # [FIX 17]

AGE_COL      <- "development_stage"          # [AGE 1] source text column in h5ad
BIN_COL      <- "age_bin_10yr"               # [AGE 1] derived 10yr bin column
CELLTYPE_COL <- "cell_type"

# [AGE 7] Values to treat as unknown/missing for the age text column
UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                    "not applicable", "adult", "child", "infant",
                    "embryonic", "fetal", "newborn")
CT_UNKNOWN     <- c("unknown", "na", "n/a", "not reported", "", "nan")

# [AGE 2] 10-year bin breaks and underscore labels (avoid "-" special char)
AGE_BREAKS <- seq(10, 90, by = 10)          # 10, 20, ..., 90
AGE_LABELS <- paste0(                        # "10_19", "20_29", ..., "80_89"
  AGE_BREAKS[-length(AGE_BREAKS)], "_",
  AGE_BREAKS[-1] - 1
)

# [AGE 10] Pilot sizes (scaled for up to 8 age bins)
PROPORTIONAL_SIZE  <- 2500L
CHUNK_SIZE_DEFAULT <- 200L
CHUNK_SIZE_MIN     <- 100L
MAX_RETRIES        <- 3L
BACKOFF_FACTOR     <- 0.7

N_HVG            <- 1000L
MIN_CELLS        <- 5L
MIN_COUNTS       <- 10L
HVG_SAMPLE_SIZE  <- 2000L

MIN_CT_CELLS      <- 20L
MAX_CELLS_PER_BIN <- 500L

# [V2] Minimum target-bin training cells required to attempt Tier 1.
MIN_CELLS_INTERACTION <- 50L

# [FIX 17] CT-STRATIFIED anchor cells per CT per other bin.
ANCHOR_PER_CT_PER_BIN <- 2L

TARGET_PER_BIN <- NULL
TARGET_DOWN    <- NULL
MAJ_BIN        <- NULL
MIN_BIN        <- NULL

ASSAY_USE <- "counts"

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

heartbeat("\n=== STARTING AGE AUGMENTATION v5 (anchor-cell approach, PILOT) ===\n", TRUE)
heartbeat(sprintf("Parallelization: %d cores available\n", N_CORES), TRUE)
heartbeat(sprintf("Memory protection: MAX_CELLS_PER_BIN = %d\n", MAX_CELLS_PER_BIN), TRUE)
heartbeat(sprintf("Anchor cells per CT per bin (stratified): %d\n", ANCHOR_PER_CT_PER_BIN), TRUE)
heartbeat(sprintf("Min cells for interaction term: %d\n", MIN_CELLS_INTERACTION), TRUE)

# ============================================================
# [V5 2] VALIDATION BARCODE EXCLUSION
# ============================================================

load_validation_barcodes <- function() {
  if (!file.exists(VALIDATION_H5AD)) {
    stop(sprintf("[V5 2] Validation file not found: %s", VALIDATION_H5AD))
  }
  sce_val  <- zellkonverter::readH5AD(VALIDATION_H5AD, use_hdf5 = TRUE)
  barcodes <- colnames(sce_val)
  rm(sce_val); gc()
  heartbeat(sprintf("[V5 2] Loaded %d validation barcodes to exclude.\n", length(barcodes)), TRUE)
  barcodes
}

exclude_validation_cells <- function(sce_obj) {
  if (length(VALIDATION_BARCODES) == 0L) return(sce_obj)
  n_before  <- ncol(sce_obj)
  keep      <- !colnames(sce_obj) %in% VALIDATION_BARCODES
  sce_obj   <- sce_obj[, keep]
  n_removed <- n_before - ncol(sce_obj)
  if (n_removed > 0) {
    heartbeat(sprintf("[V5 2] Excluded %d validation cells (kept %d).\n",
                      n_removed, ncol(sce_obj)), TRUE)
  }
  sce_obj
}

VALIDATION_BARCODES <- load_validation_barcodes()   # [V5 2]

# ============================================================
# [AGE 2] ADD_AGE_BINS HELPER
# ============================================================

add_age_bins <- function(sce_obj) {
  if (!AGE_COL %in% colnames(colData(sce_obj))) {
    stop(sprintf(
      "[AGE 2] Column '%s' not found in colData. Available: %s",
      AGE_COL, paste(colnames(colData(sce_obj)), collapse = ", ")
    ))
  }

  age_raw <- tolower(trimws(as.character(colData(sce_obj)[[AGE_COL]])))

  unknown_mask <- age_raw %in% UNKNOWN_VALUES | is.na(age_raw)

  age_digits <- gsub("[^0-9]", "", age_raw)
  age_num    <- suppressWarnings(as.numeric(age_digits))
  age_num[unknown_mask | age_digits == ""] <- NA_real_

  in_range <- !is.na(age_num) & age_num >= AGE_BREAKS[1] &
              age_num < AGE_BREAKS[length(AGE_BREAKS)]
  n_before <- ncol(sce_obj)
  sce_obj  <- sce_obj[, in_range]
  age_num  <- age_num[in_range]
  n_after  <- ncol(sce_obj)

  if (n_before > n_after) {
    message(sprintf(
      "[AGE 2] Removed %d cells with unparseable/out-of-range age (kept %d).",
      n_before - n_after, n_after
    ))
  }

  if (n_after == 0) {
    stop("[AGE 2] No cells remain after age filtering. Check AGE_COL values.")
  }

  age_bin <- cut(
    age_num,
    breaks         = AGE_BREAKS,
    labels         = AGE_LABELS,
    include.lowest = TRUE,
    right          = FALSE
  )

  colData(sce_obj)[[BIN_COL]] <- as.character(droplevels(age_bin))  # [FIX 17] plain character

  message(sprintf(
    "[AGE 2] Age bin distribution:\n%s",
    paste(capture.output(print(table(colData(sce_obj)[[BIN_COL]]))),
          collapse = "\n")
  ))

  sce_obj
}

# ------------------------------------------------------------
# INTEGER-COUNT SANITY CHECK + ASSAY-GUARANTEE HELPERS
# ------------------------------------------------------------

assert_raw_counts <- function(sce_obj, assay_name = "counts") {
  if (!assay_name %in% assayNames(sce_obj)) {
    stop(sprintf("Assay '%s' not found in the SCE object.", assay_name))
  }
  mat <- assay(sce_obj, assay_name)

  if (inherits(mat, "dgCMatrix")) {
    vals_all <- mat@x
  } else {
    nr <- nrow(mat); nc <- ncol(mat)
    if (nr == 0 || nc == 0) {
      message(sprintf("Assay '%s' has zero dimension; treating as integer counts.", assay_name))
      return(invisible(TRUE))
    }
    samp_n <- min(10000L, nr * nc)
    r_idx  <- sample.int(nr, samp_n, replace = TRUE)
    c_idx  <- sample.int(nc, samp_n, replace = TRUE)
    vals_all <- mat[cbind(r_idx, c_idx)]
  }

  if (length(vals_all) == 0L) {
    message(sprintf("Assay '%s' is all zeros or extremely sparse; treating as integer counts.", assay_name))
    return(invisible(TRUE))
  }

  n    <- min(10000L, length(vals_all))
  vals <- vals_all[seq_len(n)]

  if (any(abs(vals - round(vals)) > .Machine$double.eps^0.5)) {
    bad <- vals[which.max(abs(vals - round(vals)))]
    stop(sprintf(
      "Assay '%s' contains non-integer values (e.g., %0.6f). Must be raw integer counts for NB modelling.",
      assay_name, bad
    ))
  }

  message(sprintf("Assay '%s' passed integer-count sanity check (%d entries sampled).", assay_name, n))
  invisible(TRUE)
}

ensure_counts_assay <- function(sce_obj) {
  a <- assayNames(sce_obj)

  if ("counts" %in% a) {
    assert_raw_counts(sce_obj, "counts")
  } else if ("X" %in% a) {
    assert_raw_counts(sce_obj, "X")
    assay(sce_obj, "counts") <- assay(sce_obj, "X")
    message("No 'counts' assay found; using 'X' as 'counts' after integer check.")
    assert_raw_counts(sce_obj, "counts")
  } else {
    stop("Neither 'counts' nor 'X' assay present. Provide raw integer counts in 'counts'.")
  }

  assays(sce_obj) <- SimpleList(counts = assay(sce_obj, "counts"))

  # [FIX 1] Force materialization to dgCMatrix
  mat <- assay(sce_obj, "counts")
  if (!inherits(mat, "dgCMatrix")) {
    message(sprintf(
      "Coercing assay 'counts' from class '%s' to dgCMatrix for scDesign3 compatibility.",
      class(mat)[1]
    ))
    assay(sce_obj, "counts") <- as(mat, "dgCMatrix")
  }

  sce_obj
}

# ============================================================
# [V5 4] PRE-COMPUTE FULL-GENE LIBRARY SIZES
# Library size must be from all 31,432 genes, not the HVG subset.
# Computed once here; load_bin_fresh() attaches by barcode lookup.
# Must be after ensure_counts_assay() is defined.
# ============================================================

heartbeat("[V5 4] Pre-computing full-gene library sizes...\n", TRUE)
sce_libsize_tmp <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_libsize_tmp <- ensure_counts_assay(sce_libsize_tmp)
LIBRARY_SIZES   <- Matrix::colSums(assay(sce_libsize_tmp, "counts"))
heartbeat(sprintf("[V5 4] lib size (full genes) mean=%.0f sd=%.0f (%d cells)\n",
                  mean(LIBRARY_SIZES), sd(LIBRARY_SIZES), length(LIBRARY_SIZES)), TRUE)
rm(sce_libsize_tmp); gc()

# ============================================================
# 1. AUTO-DETECT AGE BIN STRUCTURE  [AGE 9]
# ============================================================

heartbeat("\n=== AUTO-DETECTING AGE BIN STRUCTURE ===\n", TRUE)

sce_detect <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_detect <- ensure_counts_assay(sce_detect)
sce_detect <- add_age_bins(sce_detect)           # [AGE 2]
sce_detect <- exclude_validation_cells(sce_detect)  # [V5 3]

bin_table <- table(colData(sce_detect)[[BIN_COL]])
all_bins  <- sort(names(bin_table))

heartbeat("Real dataset age bin sizes (cleaned, validation-excluded):\n", TRUE)
for (b in all_bins) {
  heartbeat(sprintf("%s: %d cells", b, bin_table[b]), TRUE)
}

MAJ_BIN <- all_bins[which.max(bin_table)]
MIN_BIN <- all_bins[which.min(bin_table)]

heartbeat(sprintf("\n>>> DETECTED MAJORITY AGE BIN: %s (%d cells)\n", MAJ_BIN, bin_table[MAJ_BIN]), TRUE)
heartbeat(sprintf(">>> DETECTED MINORITY AGE BIN: %s (%d cells)\n", MIN_BIN, bin_table[MIN_BIN]), TRUE)

# ============================================================
# 2. COMPUTE PROPORTIONAL TARGETS
# ============================================================

heartbeat(sprintf("\n=== COMPUTING PROPORTIONAL TARGETS (PILOT, %d CELLS) ===\n", PROPORTIONAL_SIZE), TRUE)

total_real <- sum(as.numeric(bin_table))
heartbeat(sprintf("Total cells in cleaned real data (validation-excluded): %d\n", total_real), TRUE)

if (total_real < PROPORTIONAL_SIZE) {
  stop(sprintf(
    "Not enough cells after cleaning: have %d, need %d",
    total_real, PROPORTIONAL_SIZE
  ))
}

prop_targets <- sapply(as.numeric(bin_table), function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins

heartbeat(sprintf("Proportional targets (%d sample):\n", PROPORTIONAL_SIZE), TRUE)
for (b in all_bins) {
  heartbeat(sprintf("%s: %d cells", b, prop_targets[b]), TRUE)
}

TARGET_PER_BIN <- prop_targets[MAJ_BIN]
TARGET_DOWN    <- prop_targets[MIN_BIN]

if (TARGET_PER_BIN < TARGET_DOWN) {
  tmp            <- TARGET_PER_BIN
  TARGET_PER_BIN <- TARGET_DOWN
  TARGET_DOWN    <- tmp
}

heartbeat(sprintf("\n>>> AUGMENTATION TARGET (per age bin): %d cells\n", TARGET_PER_BIN), TRUE)
heartbeat(sprintf(">>> DOWN-SAMPLING TARGET (per age bin): %d cells\n", TARGET_DOWN), TRUE)

# ============================================================
# 3. DIRECTORIES
# ============================================================

if (!dir.exists(CKPTDIR)) dir.create(CKPTDIR, recursive = TRUE)

# ============================================================
# 4. HVG SELECTION (stratified sample)
# ============================================================

heartbeat(sprintf("\n=== LOADING STRATIFIED SAMPLE FOR HVG SELECTION ===\n"), TRUE)
heartbeat(sprintf("Target sample size: %d cells\n", HVG_SAMPLE_SIZE), TRUE)

sce_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_full <- ensure_counts_assay(sce_full)
sce_full <- add_age_bins(sce_full)               # [AGE 2]
sce_full <- exclude_validation_cells(sce_full)   # [V5 3]

available_assays <- assayNames(sce_full)
heartbeat(sprintf("Available assays: %s\n", paste(available_assays, collapse = ", ")), TRUE)
heartbeat(sprintf("--> Using assay '%s'\n", ASSAY_USE), TRUE)

original_genes <- rownames(sce_full)
heartbeat(sprintf("Full cleaned dataset: %d cells x %d genes\n", ncol(sce_full), nrow(sce_full)), TRUE)

bin_labels_full <- colData(sce_full)[[BIN_COL]]
bin_counts_full <- table(bin_labels_full)

target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
n_per_bin_hvg      <- pmin(as.numeric(bin_counts_full), target_per_bin_hvg)
names(n_per_bin_hvg) <- names(bin_counts_full)

heartbeat("Sampling cells per age bin for HVG selection:\n", TRUE)
for (b in all_bins) {
  heartbeat(sprintf("%s: sampling %d of %d cells", b, n_per_bin_hvg[b], bin_counts_full[b]), TRUE)
}

sample_idx_hvg <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_full == b)
  if (length(idx) > n_per_bin_hvg[b]) sample(idx, n_per_bin_hvg[b]) else idx
}))

heartbeat(sprintf("Sampled %d cells across %d bins\n", length(sample_idx_hvg), length(all_bins)), TRUE)
rm(bin_counts_full, bin_labels_full); gc()

heartbeat("Loading sampled cells into memory...\n", TRUE)
sce_sample <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_sample <- ensure_counts_assay(sce_sample)
sce_sample <- add_age_bins(sce_sample)              # [AGE 2]
sce_sample <- exclude_validation_cells(sce_sample)  # [V5 3]
sce_sample <- sce_sample[, sample_idx_hvg]

heartbeat(sprintf("Sample loaded: %d cells x %d genes\n", ncol(sce_sample), nrow(sce_sample)), TRUE)

# HVG selection (sparse-safe)
heartbeat("Selecting top HVGs from sample...\n", TRUE)

count_mat    <- assay(sce_sample, ASSAY_USE)
gene_sums    <- Matrix::rowSums(count_mat)
gene_nonzero <- Matrix::rowSums(count_mat > 0)

keep_genes <- (gene_sums >= MIN_COUNTS) & (gene_nonzero >= MIN_CELLS)
n_kept     <- sum(keep_genes)
heartbeat(sprintf("Genes after filtering: %d / %d\n", n_kept, length(gene_sums)), TRUE)
if (n_kept == 0) stop("ERROR: No genes passed filtering!")

sce_filt <- sce_sample[keep_genes, ]
filt_mat <- assay(sce_filt, ASSAY_USE)

gene_means    <- Matrix::rowMeans(filt_mat)
filt_mat_sq   <- filt_mat
filt_mat_sq@x <- filt_mat_sq@x^2
gene_means_sq <- Matrix::rowMeans(filt_mat_sq)
gene_vars     <- pmax(gene_means_sq - gene_means^2, 0)

cv2 <- gene_vars / (gene_means^2 + 1e-8)
cv2[is.na(cv2) | is.infinite(cv2)] <- 0

n_select  <- min(N_HVG, length(cv2))
top_idx   <- order(cv2, decreasing = TRUE)[seq_len(n_select)]
hvg_names <- rownames(sce_filt)[top_idx]

heartbeat(sprintf("Selected %d HVGs (requested %d)\n", length(hvg_names), N_HVG), TRUE)
if (length(hvg_names) == 0) stop("ERROR: HVG selection failed!")

# [GF VOCAB] Restrict HVGs to Geneformer vocabulary genes only
gf_vocab_path <- file.path(dirname(INPUT_H5AD), "geneformer_vocab_genes.txt")
if (file.exists(gf_vocab_path)) {
  gf_vocab <- readLines(gf_vocab_path)
  # Strip version suffixes from hvg_names to match vocab
  hvg_stripped <- sub("\\.[0-9]+$", "", hvg_names)
  in_vocab     <- hvg_stripped %in% gf_vocab
  hvg_names    <- hvg_names[in_vocab]
  heartbeat(sprintf("[GF VOCAB] HVGs restricted to Geneformer vocab: %d / %d retained\n",
                    length(hvg_names), sum(in_vocab | !in_vocab)), TRUE)
  if (length(hvg_names) == 0) stop("ERROR: No HVGs remain after Geneformer vocab filtering!")
} else {
  heartbeat(sprintf("[GF VOCAB] WARNING: vocab file not found at %s -- skipping filter\n", gf_vocab_path), TRUE)
}

rm(sce_sample, sce_filt, count_mat, filt_mat, gene_sums, gene_nonzero,
   gene_means, gene_means_sq, gene_vars, cv2, top_idx, keep_genes, filt_mat_sq)
gc()

# ============================================================
# 5. COLDATA SANITIZER
# ============================================================

sanitize_coldata_for_h5ad <- function(sce_obj) {
  # [FIX 12]
  n   <- ncol(sce_obj)
  cd  <- colData(sce_obj)
  rn  <- rownames(cd)

  new_cols <- lapply(colnames(cd), function(col) {
    raw <- tryCatch(cd[[col]], error = function(e) rep(NA_character_, n))

    if (is.double(raw) && is.null(dim(raw)) && length(raw) == n) return(raw)
    if (is.integer(raw) && is.null(dim(raw)) && length(raw) == n) return(raw)

    result <- character(n)
    for (i in seq_len(n)) {
      xi <- tryCatch(raw[[i]], error = function(e) NA_character_)
      if (is.null(xi) || length(xi) == 0) {
        result[[i]] <- NA_character_
      } else if (length(xi) == 1 && is.na(xi)) {
        result[[i]] <- NA_character_
      } else {
        result[[i]] <- tryCatch(
          paste(as.character(xi), collapse = ";"),
          error = function(e) NA_character_
        )
      }
    }
    result
  })
  names(new_cols) <- colnames(cd)

  colData(sce_obj) <- DataFrame(new_cols, row.names = rn)
  sce_obj
}

# ============================================================
# 6. LOAD BIN FRESH (HVG subset, memory-safe)  [AGE 2]
# ============================================================

load_bin_fresh <- function(bin_label) {
  heartbeat(sprintf("Loading age bin '%s' (HVG subset)...", bin_label), TRUE)

  # [FIX 8] use_hdf5=FALSE to avoid HDF5Array transposition crash
  sce_raw <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)

  common_early <- intersect(hvg_names, rownames(sce_raw))
  if (length(common_early) == 0) {
    stop(sprintf("No HVGs found in raw H5AD for %s", bin_label))
  }
  sce_raw <- sce_raw[common_early, ]
  sce_raw <- ensure_counts_assay(sce_raw)

  # [AGE 2] Add age bins AFTER subsetting to HVGs (saves memory)
  sce_raw <- add_age_bins(sce_raw)

  # [V5 3] Exclude validation cells
  sce_raw <- exclude_validation_cells(sce_raw)

  # Filter unknown cell types
  ct_raw   <- as.character(colData(sce_raw)[[CELLTYPE_COL]])
  valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
  sce_raw  <- sce_raw[, valid_ct]

  # Subset to requested bin
  bin_vec  <- as.character(colData(sce_raw)[[BIN_COL]])
  bin_mask <- bin_vec == bin_label
  sce_bin  <- sce_raw[, bin_mask]

  if (ncol(sce_bin) == 0) {
    stop(sprintf("No cells for age bin '%s' after basic filtering", bin_label))
  }

  common <- intersect(hvg_names, rownames(sce_bin))
  if (length(common) == 0) {
    stop(sprintf("No HVGs present in age bin '%s' after intersection", bin_label))
  }
  sce_bin <- sce_bin[common, ]

  # Filter rare cell types
  ct_vec  <- as.character(colData(sce_bin)[[CELLTYPE_COL]])
  tab_ct  <- table(ct_vec)
  keep_ct <- names(tab_ct)[tab_ct >= MIN_CT_CELLS]
  if (length(keep_ct) == 0) {
    stop(sprintf("No cell types in age bin '%s' have >= %d cells", bin_label, MIN_CT_CELLS))
  }

  cells_keep <- ct_vec %in% keep_ct
  sce_bin    <- sce_bin[, cells_keep]
  ct_vec     <- ct_vec[cells_keep]

  # [FIX 3] Guard against NA levels
  ct_vec[is.na(ct_vec)] <- "unknown_cleaned"
  colData(sce_bin)[[CELLTYPE_COL]] <- droplevels(factor(ct_vec))
  colData(sce_bin)[[BIN_COL]]      <- as.character(colData(sce_bin)[[BIN_COL]])  # [FIX 17]

  # [V5 4] Attach full-gene library sizes by barcode lookup
  lib_vec <- LIBRARY_SIZES[colnames(sce_bin)]
  lib_vec[is.na(lib_vec)] <- median(LIBRARY_SIZES, na.rm = TRUE)
  colData(sce_bin)$library <- as.numeric(lib_vec)
  heartbeat(sprintf("  -> lib size (full genes) mean=%.0f sd=%.0f [V5 4]",
                    mean(colData(sce_bin)$library), sd(colData(sce_bin)$library)), TRUE)

  heartbeat(sprintf("  -> %d cells x %d genes, %d cell types retained",
                    ncol(sce_bin), nrow(sce_bin), length(unique(ct_vec))), TRUE)

  sce_bin
}

# ============================================================
# [V5 8] BUILD_MU_FORMULA: soft-degrade helper
# Checks sex and self_reported_ethnicity in the joint training set.
# Drops each from mu_formula and other_covariates if <2 levels.
# Always includes age_bin_10yr interaction and offset(log(library)).
# Returns list(formula = character, other_covariates = character).
# ============================================================

build_mu_formula <- function(train_sce) {
  formula_parts <- c("cell_type", "age_bin_10yr", "cell_type:age_bin_10yr")
  other_extra   <- character(0)

  for (covar in c("sex", "self_reported_ethnicity")) {
    if (!covar %in% colnames(colData(train_sce))) {
      heartbeat(sprintf(" [V5 8] Column '%s' absent from colData -> skipped.\n", covar), TRUE)
      next
    }
    n_levels <- length(unique(na.omit(as.character(colData(train_sce)[[covar]]))))
    if (n_levels < 2L) {
      heartbeat(sprintf(" [V5 8] '%s' has %d level(s) -> soft-degraded out of mu_formula.\n",
                        covar, n_levels), TRUE)
      next
    }
    formula_parts <- c(formula_parts, covar)
    other_extra   <- c(other_extra, covar)
    heartbeat(sprintf(" [V5 8] '%s' has %d levels -> included in mu_formula.\n",
                      covar, n_levels), TRUE)
  }

  formula_str  <- paste(paste(formula_parts, collapse = " + "),
                        "+ offset(log(library))")
  other_covars <- c(BIN_COL, other_extra, "library")

  heartbeat(sprintf(" [V5 8] mu_formula = \"%s\"\n", formula_str), TRUE)
  heartbeat(sprintf(" [V5 8] other_covariates = c(%s)\n",
                    paste(sprintf("'%s'", other_covars), collapse = ", ")), TRUE)

  list(formula = formula_str, other_covariates = other_covars)
}

# ============================================================
# 7. VALIDATE NEW COUNTS
# ============================================================

validate_new_counts <- function(new_count, ref_genes, chunk_id, bin_label) {
  if (is.null(new_count)) {
    stop("scDesign3 returned NULL count matrix")
  }

  # [FIX 11] Coerce 1D vector to matrix if needed
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(
      new_count,
      nrow = length(new_count),
      ncol = 1,
      dimnames = list(names(new_count), NULL)
    )
  }
  if (!is.null(rownames(new_count)) && !is.null(ref_genes)) {
    if (ncol(new_count) == length(ref_genes) && nrow(new_count) != length(ref_genes)) {
      new_count <- t(new_count)
    }
  }

  if (ncol(new_count) == 0) {
    stop("scDesign3 returned empty count matrix (0 columns)")
  }

  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(
      0, nrow = length(missing), ncol = ncol(new_count),
      dimnames = list(missing, colnames(new_count))
    )
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop = FALSE]
  colnames(new_count) <- paste0(
    "synthetic_", bin_label, "_chunk", chunk_id, "_", seq_len(ncol(new_count))
  )
  new_count
}

# ============================================================
# 8. H5AD WRITE HELPERS
# ============================================================

sanitize_for_h5ad <- function(sce_obj) {
  cd <- as.data.frame(colData(sce_obj))
  for (col in colnames(cd)) {
    v <- cd[[col]]
    if (is.factor(v)) {
      cd[[col]] <- as.character(v)
    } else if (is.integer(v) || is.numeric(v)) {
      if (grepl("joinid|id|index|barcode|name", col, ignore.case = TRUE)) {
        cd[[col]] <- as.character(v)
      }
      if (all(is.na(v))) cd[[col]] <- as.character(v)
    }
  }
  colData(sce_obj) <- DataFrame(cd)
  sce_obj
}

rebuild_for_write <- function(sce_obj, drop_all_na = FALSE) {
  assay_nm <- if (ASSAY_USE %in% assayNames(sce_obj)) ASSAY_USE else assayNames(sce_obj)[1]
  cnt <- assay(sce_obj, assay_nm)
  if (!inherits(cnt, "dgCMatrix")) cnt <- as(cnt, "dgCMatrix")

  n      <- ncol(sce_obj)
  cd_raw <- colData(sce_obj)

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
    heartbeat(sprintf(" Dropped %d all-NA colData columns before writing\n",
                      sum(!not_all_na)), TRUE)
  }

  rn_safe <- make.unique(as.character(colnames(sce_obj)), sep = "_dup")

  col_list_clean <- lapply(col_list, function(v) {
    v <- unclass(v); attributes(v) <- NULL; length(v) <- n; v
  })
  names(col_list_clean) <- names(col_list)

  cd_new <- structure(
    col_list_clean,
    class     = "data.frame",
    row.names = rn_safe,
    names     = names(col_list_clean)
  )

  colnames(cnt) <- rn_safe

  SingleCellExperiment(
    assays  = list(counts = cnt),
    colData = DataFrame(cd_new)
  )
}

# ============================================================
# 8b. PRE-CACHE ANCHOR CELLS  [FIX 17]
# ============================================================

heartbeat(sprintf("\n=== PRE-CACHING ANCHOR CELLS (CT-stratified, %d per CT per bin) ===\n",
                  ANCHOR_PER_CT_PER_BIN), TRUE)
anchor_cache <- list()
for (b in all_bins) {
  tryCatch({
    sce_b  <- load_bin_fresh(b)
    ct_b   <- as.character(colData(sce_b)[[CELLTYPE_COL]])
    tab_b  <- table(ct_b)
    anch_idx <- unlist(lapply(names(tab_b), function(ct) {
      ids <- which(ct_b == ct)
      n   <- min(ANCHOR_PER_CT_PER_BIN, length(ids))
      sample(ids, n)
    }))
    anchor_cache[[b]] <- sce_b[, anch_idx]
    heartbeat(sprintf(" Cached %d anchor cells for bin '%s' (%d CTs, %d per CT)\n",
                      length(anch_idx), b, length(tab_b), ANCHOR_PER_CT_PER_BIN), TRUE)
    rm(sce_b); gc()
  }, error = function(e) {
    heartbeat(sprintf(" WARNING: anchor cache FAILED for bin '%s': %s\n", b, conditionMessage(e)), TRUE)
  })
}

# build_joint_train(): combine target-bin cells with anchor cells from
# all other bins. Applies FIX 19 (singleton CT guard) on the joint set.
build_joint_train <- function(sub_train_target, target_bin) {
  other_anchors <- anchor_cache[setdiff(names(anchor_cache), target_bin)]
  if (length(other_anchors) == 0) {
    heartbeat(" WARNING: no anchor bins available; using target-bin cells only.\n", TRUE)
    return(sub_train_target)
  }

  # [FIX 20] Subset each anchor bin to only CTs present in the TARGET bin.
  target_cts    <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
  other_anchors <- lapply(other_anchors, function(a) {
    ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
    a[, ct_a %in% target_cts, drop = FALSE]
  })
  other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]
  if (length(other_anchors) == 0) {
    heartbeat(" WARNING: no anchor cells share CTs with target bin; using target only.\n", TRUE)
    return(sub_train_target)
  }
  heartbeat(sprintf(" [FIX 20] Anchors subsetted to %d target CTs; %d anchor bins remain.\n",
                    length(target_cts), length(other_anchors)), TRUE)

  common_genes <- rownames(sub_train_target)
  for (sce_a in other_anchors) common_genes <- intersect(common_genes, rownames(sce_a))
  if (length(common_genes) == 0) {
    heartbeat(" WARNING: no common genes with anchors; skipping anchor injection.\n", TRUE)
    return(sub_train_target)
  }

  pieces <- c(
    list(sub_train_target[common_genes, ]),
    lapply(other_anchors, function(a) a[common_genes, ])
  )
  joint <- do.call(cbind, pieces)

  # [FIX 19] Drop cell types with <2 cells in the JOINT set.
  ct_joint  <- as.character(colData(joint)[[CELLTYPE_COL]])
  tab_joint <- table(ct_joint)
  keep_ct   <- names(tab_joint)[tab_joint >= 2L]
  if (length(keep_ct) < length(tab_joint)) {
    n_dropped <- sum(tab_joint[tab_joint < 2L])
    heartbeat(sprintf(
      " [FIX 19] Dropping %d cells from %d singleton CT(s) in joint training set.\n",
      n_dropped, sum(tab_joint < 2L)
    ), TRUE)
    joint    <- joint[, ct_joint %in% keep_ct]
    ct_clean <- as.character(colData(joint)[[CELLTYPE_COL]])
    colData(joint)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
  }

  colData(joint)[[BIN_COL]] <- as.character(colData(joint)[[BIN_COL]])

  n_target <- ncol(sub_train_target)
  n_anchor <- ncol(joint) - n_target
  n_bins   <- length(unique(as.character(colData(joint)[[BIN_COL]])))
  n_cts    <- length(unique(as.character(colData(joint)[[CELLTYPE_COL]])))
  heartbeat(sprintf(
    " Joint training: %d target + %d anchor = %d total, %d bins, %d CTs\n",
    n_target, n_anchor, ncol(joint), n_bins, n_cts
  ), TRUE)
  joint
}

# ============================================================
# 9. AUGMENTATION FOR ONE AGE BIN  [AGE 9]
# ============================================================

augment_one_bin <- function(bin_label) {
  ckpt_file <- file.path(
    CKPTDIR,
    paste0("age_bin_", gsub("[^a-zA-Z0-9_]", "_", bin_label), ".rds")
  )

  if (file.exists(ckpt_file)) {
    heartbeat(sprintf(
      "Checkpoint found for '%s' -- loading from disk, skipping recomputation.",
      bin_label
    ), TRUE)
    sce_ckpt <- readRDS(ckpt_file)
    sce_ckpt <- sanitize_coldata_for_h5ad(sce_ckpt)
    return(sce_ckpt)
  }

  sub_real_full <- load_bin_fresh(bin_label)
  n_full        <- ncol(sub_real_full)

  # Majority bin: real-only, possibly downsample
  if (bin_label == MAJ_BIN) {
    if (n_full > TARGET_PER_BIN) {
      heartbeat(sprintf(
        "Majority age bin '%s': downsampling %d -> %d real cells (no synthetic).",
        bin_label, n_full, TARGET_PER_BIN
      ), TRUE)
      set.seed(123)
      keep_idx <- sample(n_full, TARGET_PER_BIN)
      sub_real <- sub_real_full[, keep_idx]
    } else {
      heartbeat(sprintf(
        "Majority age bin '%s': at/below target (n=%d), no synthetic.",
        bin_label, n_full
      ), TRUE)
      sub_real <- sub_real_full
    }
    assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  # Minority bins: proportional downsample, then augment
  target_real_min <- prop_targets[bin_label]
  if (is.na(target_real_min) || target_real_min <= 0) {
    target_real_min <- min(n_full, TARGET_PER_BIN)
    heartbeat(sprintf(
      "WARNING: Invalid proportional target for '%s'; using %d instead.\n",
      bin_label, target_real_min
    ), TRUE)
  }

  if (n_full > target_real_min) {
    heartbeat(sprintf(
      "Minority age bin '%s': downsampling %d -> %d real cells (proportional).",
      bin_label, n_full, target_real_min
    ), TRUE)
    set.seed(123)
    keep_idx <- sample(n_full, target_real_min)
    sub_real <- sub_real_full[, keep_idx]
  } else {
    heartbeat(sprintf(
      "Minority age bin '%s': using all %d real cells (<= proportional target %d).",
      bin_label, n_full, target_real_min
    ), TRUE)
    sub_real <- sub_real_full
  }

  # [FIX 4] Materialize sub_real counts
  assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")

  n_real <- ncol(sub_real)
  need   <- TARGET_PER_BIN - n_real

  if (need <= 0) {
    heartbeat(sprintf(
      "Age bin '%s' already at/above TARGET_PER_BIN (n=%d). No augmentation needed.",
      bin_label, n_real
    ), TRUE)
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  heartbeat(sprintf(
    "Age bin '%s': augmenting %d -> %d cells (need %d synthetic).",
    bin_label, n_real, TARGET_PER_BIN, need
  ), TRUE)

  sims      <- list()
  remaining <- need
  chunk_id  <- 1L

  while (remaining > 0) {
    chunk   <- min(CHUNK_SIZE_DEFAULT, remaining)
    attempt <- 1L
    success <- FALSE

    while (attempt <= MAX_RETRIES && !success) {
      heartbeat(sprintf(
        "Augment '%s' | chunk %d | attempt %d | n=%d | remaining=%d",
        bin_label, chunk_id, attempt, chunk, remaining
      ), TRUE)

      sub_train      <- load_bin_fresh(bin_label)
      n_train_loaded <- ncol(sub_train)

      # [FIX 5] chunk >= 2 * n_cell_types
      n_ct_train <- length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
      min_chunk  <- 2L * n_ct_train
      if (chunk < min_chunk) {
        chunk <- min_chunk
        heartbeat(sprintf(
          " Bumping chunk size to %d (must be >= 2 x n_cell_types=%d)",
          chunk, n_ct_train
        ), TRUE)
      }
      heartbeat(sprintf(" Loaded age bin '%s' for training: %d cells", bin_label, n_train_loaded), TRUE)

      # Stratified sampling if loaded size > MAX_CELLS_PER_BIN
      if (n_train_loaded > MAX_CELLS_PER_BIN) {
        ct_vec <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        tab_ct <- table(ct_vec)
        tab_ct <- tab_ct[tab_ct > 0]
        if (length(tab_ct) == 0) stop("No cell types found in training data")

        min_per_ct <- 2L
        cap        <- as.integer(MAX_CELLS_PER_BIN)
        base_alloc <- floor(cap * as.numeric(tab_ct) / sum(tab_ct))
        alloc      <- pmax(min_per_ct, as.integer(base_alloc))
        names(alloc) <- names(tab_ct)
        alloc        <- pmin(alloc, as.integer(tab_ct))

        while (sum(alloc) > cap) {
          reducible      <- names(alloc)[alloc > min_per_ct]
          if (length(reducible) == 0) break
          largest        <- reducible[which.max(alloc[reducible])]
          alloc[largest] <- alloc[largest] - 1L
        }

        if (sum(alloc) < cap) {
          room <- as.integer(tab_ct) - alloc
          names(room) <- names(tab_ct)
          while (sum(alloc) < cap && any(room > 0)) {
            addable     <- names(room)[room > 0]
            pick        <- addable[which.max(room[addable])]
            alloc[pick] <- alloc[pick] + 1L
            room[pick]  <- room[pick] - 1L
          }
        }

        keep_idx  <- unlist(lapply(names(alloc), function(ct) {
          ids <- which(ct_vec == ct)
          sample(ids, alloc[[ct]], replace = FALSE)
        }))
        sub_train <- sub_train[, keep_idx, drop = FALSE]

        # [FIX 3] Force clean factor
        ct_new <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        ct_new[is.na(ct_new)] <- "unknown_cleaned"
        colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_new))

        heartbeat(sprintf(
          " -> Stratified training sample: %d cells (cap %d), cell types: %d",
          ncol(sub_train), cap,
          length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
        ), TRUE)

        ct_tab2 <- table(colData(sub_train)[[CELLTYPE_COL]])
        bad     <- names(ct_tab2)[ct_tab2 < 2]
        if (length(bad) > 0) {
          stop(sprintf(
            "Training subset has cell type(s) with <2 cells: %s",
            paste(bad, collapse = ", ")
          ))
        }
      } else {
        heartbeat(sprintf(
          " -> Training on FULL bin (%d cells, below MAX_CELLS_PER_BIN=%d)",
          n_train_loaded, MAX_CELLS_PER_BIN
        ), TRUE)
      }

      heartbeat(sprintf(" Final training set: %d cells x %d genes",
                        ncol(sub_train), nrow(sub_train)), TRUE)

      # [FIX 3] Final NA guard and droplevels
      ct_final <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      ct_final[is.na(ct_final)] <- "unknown_cleaned"
      colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_final))

      ct_vec_train    <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      ct_counts_train <- table(ct_vec_train)

      if (any(ct_counts_train < 2L)) {
        stop(sprintf(
          "Training subset for '%s' has cell type(s) with <2 cells: %s",
          bin_label,
          paste(names(ct_counts_train)[ct_counts_train < 2L], collapse = ", ")
        ))
      }

      # [FIX 17] Build joint training set: target-bin cells + anchor cells
      sub_train_target <- sub_train
      sub_train        <- build_joint_train(sub_train_target, bin_label)
      ct_counts_train  <- table(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))

      # [V5 6/7/8] Build mu_formula and other_covariates with soft-degrade
      mu_spec         <- build_mu_formula(sub_train)
      mu_formula_str  <- mu_spec$formula
      other_covars_v5 <- mu_spec$other_covariates

      # [FIX 18] Adaptive sparsity threshold for important_feature.
      n_target_cells     <- ncol(sub_train_target)
      mat_for_imp        <- assay(sub_train, ASSAY_USE)
      zero_frac          <- Matrix::rowMeans(mat_for_imp == 0)
      sparsity_threshold <- if (n_target_cells < MIN_CELLS_INTERACTION) 0.95 else 0.80
      imp_feat_vec       <- zero_frac <= sparsity_threshold
      n_imp              <- sum(imp_feat_vec)
      heartbeat(sprintf(
        " important_feature: %d / %d genes (threshold=%.2f) [FIX 18]\n",
        n_imp, length(imp_feat_vec), sparsity_threshold
      ), TRUE)
      if (n_imp < 2L) {
        heartbeat(" WARNING: <2 genes pass sparsity filter; using all genes.", TRUE)
        imp_feat_vec <- rep(TRUE, length(imp_feat_vec))
      }

      # Determine which tiers are available based on age variance in joint set
      age_vals        <- unique(as.character(colData(sub_train)[[BIN_COL]]))
      has_age_var     <- length(age_vals) > 1L
      use_interaction <- has_age_var && (n_target_cells >= MIN_CELLS_INTERACTION)
      if (!has_age_var) {
        heartbeat(" WARNING: anchor injection failed; age has no variance -> Tier 1 unavailable.\n", TRUE)
      } else {
        heartbeat(sprintf(" Age bins in joint training: %s\n", paste(age_vals, collapse = ", ")), TRUE)
      }

      # ----------------------------------------------------------------
      # [FIX 21] LOW-LEVEL API FOR TIER 1 (age + sex + ethnicity covariates)
      # ----------------------------------------------------------------
      run_sd3_lowlevel <- function(mu_f, corr_f) {
        heartbeat(sprintf(" [low-level] mu=%-65s corr=%-12s\n", mu_f, corr_f), TRUE)
        tryCatch({
          # Step 1: Build training data from JOINT set
          joint_data <- construct_data(
            sce              = sub_train,
            assay_use        = ASSAY_USE,
            celltype         = CELLTYPE_COL,
            pseudotime       = NULL,
            spatial          = NULL,
            other_covariates = other_covars_v5,   # [V5 7]
            ncell            = ncol(sub_train),
            corr_by          = corr_f,
            parallelization  = "mcmapply",
            BPPARAM          = NULL
          )

          # Step 2: Fit marginals on joint set
          marginal_res <- fit_marginal(
            data            = joint_data,
            predictor       = "gene",
            mu_formula      = mu_f,
            sigma_formula   = "cell_type",
            family_use      = "zinb",             # [V5 5]
            n_cores         = N_CORES,
            usebam          = FALSE,
            parallelization = "mcmapply",
            BPPARAM         = NULL
          )

          # Step 3: Fit copula on joint set
          copula_res <- fit_copula(
            sce               = sub_train,
            assay_use         = ASSAY_USE,
            marginal_list     = marginal_res,
            family_use        = "zinb",           # [V5 5]
            copula            = "gaussian",
            n_cores           = N_CORES,
            input_data        = joint_data$dat,
            important_feature = imp_feat_vec,
            if_sparse         = FALSE,
            parallelization   = "mcmapply",
            BPPARAM           = NULL
          )

          # Step 4: Build new_covariate from TARGET-BIN ROWS ONLY [FIX 21 KEY]
          target_dat_rows <- which(
            as.character(joint_data$dat[[BIN_COL]]) == bin_label
          )
          if (length(target_dat_rows) == 0) {
            stop(sprintf("[FIX 21] No target-bin rows in joint_data$dat for '%s'", bin_label))
          }
          new_cov_idx <- sample(target_dat_rows, chunk,
                                replace = (length(target_dat_rows) < chunk))
          new_cov_df  <- joint_data$dat[new_cov_idx, , drop = FALSE]

          # [FIX 22] Ensure every CT in new_cov_df has >= 2 rows.
          cov_cts    <- as.character(new_cov_df[[CELLTYPE_COL]])
          ct_tab_cov <- table(cov_cts)
          thin_cts   <- names(ct_tab_cov)[ct_tab_cov < 2L]
          if (length(thin_cts) > 0) {
            extra_rows <- lapply(thin_cts, function(ct) {
              ct_target_rows <- target_dat_rows[
                as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct
              ]
              if (length(ct_target_rows) == 0L) return(NULL)
              extra_idx <- sample(ct_target_rows,
                                  2L - ct_tab_cov[[ct]],
                                  replace = TRUE)
              joint_data$dat[extra_idx, , drop = FALSE]
            })
            extra_rows <- do.call(rbind, Filter(Negate(is.null), extra_rows))
            if (!is.null(extra_rows) && nrow(extra_rows) > 0) {
              new_cov_df <- rbind(new_cov_df, extra_rows)
              heartbeat(sprintf(
                " [FIX 22] Added %d row(s) to new_cov_df to ensure >=2 per CT.\n",
                nrow(extra_rows)
              ), TRUE)
            }
          }

          # Add corr_group column
          if (corr_f == "1") {
            new_cov_df$corr_group <- 1L
          } else {
            new_cov_df$corr_group <- as.character(new_cov_df[[CELLTYPE_COL]])
          }
          rownames(new_cov_df) <- paste0("Cell", seq_len(nrow(new_cov_df)))

          # Step 5: Extract parameters for target-bin new cells
          para_list <- extract_para(
            sce             = sub_train,
            assay_use       = ASSAY_USE,
            marginal_list   = marginal_res,
            n_cores         = N_CORES,
            family_use      = "zinb",             # [V5 5]
            new_covariate   = new_cov_df,
            parallelization = "mcmapply",
            BPPARAM         = NULL,
            data            = joint_data$dat
          )

          # Step 6: Simulate synthetic count matrix
          new_count <- simu_new(
            sce               = sub_train,
            assay_use         = ASSAY_USE,
            mean_mat          = para_list$mean_mat,
            sigma_mat         = para_list$sigma_mat,
            zero_mat          = para_list$zero_mat,
            quantile_mat      = NULL,
            copula_list       = copula_res$copula_list,
            n_cores           = N_CORES,
            family_use        = "zinb",           # [V5 5]
            nonnegative       = TRUE,
            nonzerovar        = TRUE,
            input_data        = joint_data$dat,
            new_covariate     = new_cov_df,
            important_feature = copula_res$important_feature,
            parallelization   = "mcmapply",
            BPPARAM           = NULL,
            filtered_gene     = joint_data$filtered_gene
          )

          list(new_count = new_count, new_covariate = new_cov_df)
        }, error = function(e) {
          heartbeat(sprintf(" ERROR: %s\n", conditionMessage(e)), TRUE); NULL
        })
      }

      # --- TIER 1 ONLY ---
      # Gate: every CT must appear in >= 2 age bins.
      ct_bin_tab <- table(
        CT  = as.character(colData(sub_train)[[CELLTYPE_COL]]),
        Bin = as.character(colData(sub_train)[[BIN_COL]])
      )
      ct_n_bins    <- rowSums(ct_bin_tab > 0)
      single_ct    <- names(ct_n_bins)[ct_n_bins < 2L]
      if (length(single_ct) > 0) {
        stop(sprintf(
          "[Tier 1 gate] %d CT(s) appear in only 1 bin and cannot support interaction term: %s",
          length(single_ct), paste(single_ct, collapse = ", ")
        ))
      }

      heartbeat(" [Tier 1] mu=cell_type+age+interaction+sex+ethnicity+offset(lib), corr=cell_type [low-level V5]\n", TRUE)
      res <- run_sd3_lowlevel(mu_formula_str, "cell_type")

      if (is.null(res)) {
        stop(sprintf(
          "[Tier 1 FAILED] Bin '%s' chunk %d attempt %d: Tier 1 returned NULL. Fix the underlying error above.",
          bin_label, chunk_id, attempt
        ))
      }

      new_counts <- validate_new_counts(
        res$new_count, rownames(sub_real), chunk_id, bin_label
      )
      new_cov <- res$new_covariate
      if (is.null(new_cov)) {
        new_cov <- DataFrame(row.names = colnames(new_counts))
      } else {
        rownames(new_cov) <- colnames(new_counts)
      }

      sim <- SingleCellExperiment(
        assays  = list(counts = new_counts),
        colData = new_cov
      )

      # [FIX 13] Align metadata with typed NAs matching real data
      ref_cols     <- colnames(colData(sub_real))
      missing_cols <- setdiff(ref_cols, colnames(colData(sim)))
      for (m in missing_cols) {
        ref_val <- colData(sub_real)[[m]]
        if (is.list(ref_val)) {
          colData(sim)[[m]] <- rep(NA_character_, ncol(sim))
        } else if (is.double(ref_val)) {
          colData(sim)[[m]] <- rep(NA_real_, ncol(sim))
        } else if (is.integer(ref_val)) {
          colData(sim)[[m]] <- rep(NA_integer_, ncol(sim))
        } else {
          colData(sim)[[m]] <- rep(NA_character_, ncol(sim))
        }
      }
      colData(sim) <- colData(sim)[, ref_cols, drop = FALSE]

      # [FIX 15 / AGE 2] Explicitly set age bin label on synthetic cells
      colData(sim)[[BIN_COL]] <- rep(bin_label, ncol(sim))
      heartbeat(sprintf(" Set %s='%s' on all %d synthetic cells.",
                        BIN_COL, bin_label, ncol(sim)), TRUE)

      if (any(assay(sim, ASSAY_USE) < 0)) {
        warning("Negative counts detected in synthetic matrix!")
      }

      lib_train <- colSums(assay(sub_train, ASSAY_USE))
      lib_syn   <- colSums(assay(sim, ASSAY_USE))
      heartbeat(sprintf(
        " Library size - Training: mean=%.0f, sd=%.0f", mean(lib_train), sd(lib_train)
      ), TRUE)
      heartbeat(sprintf(
        " Library size - Synthetic: mean=%.0f, sd=%.0f", mean(lib_syn), sd(lib_syn)
      ), TRUE)

      ct_syn <- table(colData(sim)[[CELLTYPE_COL]])
      heartbeat(" Cell type proportions (Training vs Synthetic):\n", TRUE)
      for (ct in names(ct_counts_train)) {
        pct_train <- 100 * ct_counts_train[ct] / sum(ct_counts_train)
        pct_syn   <- ifelse(is.na(ct_syn[ct]), 0, 100 * ct_syn[ct] / sum(ct_syn))
        heartbeat(sprintf("  %s: Train=%.2f%%, Syn=%.2f%%", ct, pct_train, pct_syn), TRUE)
      }

      sims[[length(sims) + 1L]] <- sim
      success <- TRUE
      heartbeat(sprintf(" Chunk %d completed: generated %d cells", chunk_id, ncol(sim)), TRUE)

      rm(res, sim, new_counts, new_cov, sub_train, sub_train_target, lib_train, lib_syn,
         ct_counts_train, ct_syn, mu_spec, mu_formula_str, other_covars_v5)
      gc()
    } # end attempt loop

    if (!success) {
      # FINAL SAFETY FALLBACK: naive resampling
      heartbeat(sprintf(
        "scDesign3 failed for age bin '%s' after %d attempts; FALLING BACK to naive resampling (%d cells).",
        bin_label, MAX_RETRIES, remaining
      ), TRUE)

      set.seed(123)
      n_real_cells    <- ncol(sub_real)
      resample_idx    <- sample.int(n_real_cells, remaining, replace = TRUE)
      real_counts_mat <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
      res_counts      <- real_counts_mat[, resample_idx, drop = FALSE]
      colnames(res_counts) <- paste0(
        "synthetic_resample_", bin_label, "_", seq_len(ncol(res_counts))
      )

      res_cov <- as.data.frame(colData(sub_real))[resample_idx, , drop = FALSE]
      rownames(res_cov) <- colnames(res_counts)

      sim_fallback <- SingleCellExperiment(
        assays  = list(counts = res_counts),
        colData = DataFrame(res_cov)
      )

      # [FIX 15b] Ensure bin label on fallback cells
      colData(sim_fallback)[[BIN_COL]] <- rep(bin_label, ncol(sim_fallback))
      sims[[length(sims) + 1L]] <- sim_fallback
      heartbeat(sprintf(
        " Fallback produced %d pseudo-synthetic cells for age bin '%s'.",
        ncol(sim_fallback), bin_label
      ), TRUE)

      remaining <- 0L
      break
    }

    remaining <- remaining - chunk
    chunk_id  <- chunk_id + 1L
  } # end while(remaining)

  heartbeat(sprintf(
    "Combining %d synthetic chunks for age bin '%s'...",
    length(sims), bin_label
  ), TRUE)

  out_bin <- do.call(cbind, c(list(sub_real), sims))
  heartbeat(sprintf("Age bin '%s' final size: %d cells", bin_label, ncol(out_bin)), TRUE)

  saveRDS(out_bin, ckpt_file)
  out_bin
}

# ============================================================
# 10. PROCESS ALL AGE BINS  [AGE 9]
# ============================================================

heartbeat("\n=== PROCESSING ALL AGE BINS ===\n", TRUE)
bin_objects <- list()
for (b in all_bins) {
  heartbeat(sprintf("\n=== PROCESSING AGE BIN: %s ===\n", b), TRUE)
  bin_objects[[b]] <- augment_one_bin(b)
}

# ============================================================
# 11. COMBINE ALL BINS AND RESTORE FULL GENE SET
# ============================================================

heartbeat("\n=== COMBINING ALL AGE BINS ===\n", TRUE)

common_genes <- Reduce(intersect, lapply(bin_objects, rownames))
bin_objects <- lapply(bin_objects, function(sce) { sce <- sce[common_genes, ]; rowData(sce) <- NULL; sce })
sce_combined <- do.call(cbind, bin_objects)
heartbeat(sprintf(
  "Combined (Balanced Augmented): %d cells x %d genes",
  ncol(sce_combined), nrow(sce_combined)
), TRUE)

heartbeat("Zero-padding to restore full gene set...\n", TRUE)

present_genes  <- rownames(sce_combined)
missing_genes  <- setdiff(original_genes, present_genes)

# [FIX 7] Force dgCMatrix before rbind
current_counts <- as(assay(sce_combined, ASSAY_USE), "dgCMatrix")

if (length(missing_genes) > 0) {
  heartbeat(sprintf("Adding %d missing genes...\n", length(missing_genes)), TRUE)
  # [FIX 9b] sparseMatrix for explicit dgCMatrix
  pad_mat <- Matrix::sparseMatrix(
    i = integer(0), j = integer(0), x = numeric(0),
    dims     = c(length(missing_genes), ncol(sce_combined)),
    dimnames = list(missing_genes, colnames(sce_combined))
  )
  full_counts <- rbind(current_counts, pad_mat)
} else {
  full_counts <- current_counts
}

full_counts <- as(full_counts, "dgCMatrix")
gene_order  <- match(original_genes, rownames(full_counts))
valid_order <- gene_order[!is.na(gene_order)]
full_counts <- full_counts[valid_order, , drop = FALSE]
rownames(full_counts) <- original_genes[!is.na(gene_order)]

truly_missing <- original_genes[is.na(gene_order)]
if (length(truly_missing) > 0) {
  heartbeat(sprintf("Safety-padding %d genes still absent after rbind\n", length(truly_missing)), TRUE)
  extra_pad <- Matrix::sparseMatrix(
    i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(truly_missing), ncol(full_counts)),
    dimnames = list(truly_missing, colnames(full_counts))
  )
  full_counts <- rbind(full_counts, extra_pad)
  full_counts <- full_counts[original_genes, , drop = FALSE]
}

heartbeat("Rebuilding SCE with full gene set...\n", TRUE)

cd      <- colData(sce_combined)
rd_full <- rowData(sce_full)

rd_genes_present <- intersect(original_genes, rownames(rd_full))
rd_genes_missing <- setdiff(original_genes, rownames(rd_full))

if (length(rd_genes_present) > 0) {
  rd_subset <- rd_full[rd_genes_present, , drop = FALSE]
} else {
  rd_subset <- rd_full[integer(0), , drop = FALSE]
}

if (length(rd_genes_missing) > 0) {
  heartbeat(sprintf("WARNING: %d genes missing from rowData; filling with NA.\n",
                    length(rd_genes_missing)), TRUE)
  rd_pad <- as.data.frame(matrix(
    NA, nrow = length(rd_genes_missing), ncol = ncol(rd_subset),
    dimnames = list(rd_genes_missing, colnames(rd_subset))
  ))
  rd_combined <- rbind(as.data.frame(rd_subset), rd_pad)
} else {
  rd_combined <- as.data.frame(rd_subset)
}

rd_combined <- rd_combined[original_genes, , drop = FALSE]

sce_full_restored <- SingleCellExperiment(
  assays  = list(counts = full_counts),
  colData = cd,
  rowData = DataFrame(rd_combined)
)
rownames(sce_full_restored) <- original_genes
sce_combined <- sce_full_restored
rm(sce_full_restored); gc()

# ============================================================
# 12. SOURCE COLUMN + COLDATA SANITIZATION
# ============================================================

heartbeat("Adding source column (real vs synthetic)...\n", TRUE)
colData(sce_combined)$source <- ifelse(
  grepl("^synthetic_", colnames(sce_combined)),
  "synthetic", "real"
)

heartbeat("Sanitizing colData for H5AD writing...\n", TRUE)
sce_combined <- sanitize_coldata_for_h5ad(sce_combined)   # [FIX 12]

# ============================================================
# 13. BASE OUTPUT FILES  [AGE 8]
# ============================================================

heartbeat("\n=== GENERATING BASE OUTPUT FILES ===\n", TRUE)

out_full <- paste0(OUTPUT_BASE, "_Full_BalancedAugmented_AGE.h5ad")
heartbeat(sprintf("Writing: %s\n", out_full), TRUE)
zellkonverter::writeH5AD(rebuild_for_write(sce_combined, drop_all_na = FALSE),
                         out_full, compression = "gzip")

out_real <- paste0(OUTPUT_BASE, "_RealOnly_AGE.h5ad")
heartbeat(sprintf("Writing: %s\n", out_real), TRUE)
sce_real <- sce_combined[, colData(sce_combined)$source == "real"]
zellkonverter::writeH5AD(rebuild_for_write(sce_real, drop_all_na = FALSE),
                         out_real, compression = "gzip")

out_syn <- paste0(OUTPUT_BASE, "_SyntheticOnly_AGE.h5ad")
heartbeat(sprintf("Writing: %s\n", out_syn), TRUE)
sce_syn <- sce_combined[, colData(sce_combined)$source == "synthetic"]
zellkonverter::writeH5AD(rebuild_for_write(sce_syn, drop_all_na = TRUE),
                         out_syn, compression = "gzip")

out_sum <- paste0(OUTPUT_BASE, "_Summary_AGE.csv")
heartbeat(sprintf("Writing: %s\n", out_sum), TRUE)
summary_df <- as.data.frame(table(
  AgeBin = colData(sce_combined)[[BIN_COL]],
  Source = colData(sce_combined)$source
))
colnames(summary_df) <- c("AgeBin", "Source", "Count")
write.csv(summary_df, out_sum, row.names = FALSE)

# ============================================================
# 14. FAIRNESS DATASETS  [AGE 8]
# ============================================================

heartbeat("\n=== GENERATING FAIRNESS DATASETS ===\n", TRUE)

grp_counts <- function(sce_obj) {
  bv <- as.character(colData(sce_obj)[[BIN_COL]])
  table(bv)
}

sample_exact <- function(idx, n, replace = FALSE) {
  n <- as.integer(round(as.numeric(n)))
  if (is.na(n) || n < 0) stop("sample_exact(): invalid n")
  if (n == 0) return(integer(0))
  if (!replace && length(idx) < n) stop("sample_exact(): not enough cells")
  sample(idx, n, replace = replace)
}

# 1) Proportional (~2500, real-only)  [FIX 16]
heartbeat(sprintf("Building Proportional (~%d, real-only) dataset\n", PROPORTIONAL_SIZE), TRUE)

bin_labels_det  <- colData(sce_detect)[[BIN_COL]]
prop_sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_det == b)
  n_b <- prop_targets[b]
  if (n_b <= 0 || length(idx) == 0) integer(0) else sample(idx, n_b)
}))

# [FIX 16] Load with use_hdf5=FALSE and materialize counts
sce_for_prop <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_for_prop <- ensure_counts_assay(sce_for_prop)
sce_for_prop <- add_age_bins(sce_for_prop)               # [AGE 2]
sce_for_prop <- exclude_validation_cells(sce_for_prop)   # [V5 3]
sce_prop     <- sce_for_prop[, prop_sample_idx]

prop_sum <- sum(assay(sce_prop, ASSAY_USE))
heartbeat(sprintf("[FIX 16] Proportional count matrix sum = %.0f (should be >> 0)\n", prop_sum), TRUE)
if (prop_sum == 0) {
  stop("FATAL [FIX 16]: Proportional count matrix is all zeros after materialization.")
}

out_prop <- paste0(OUTPUT_BASE, "_Proportional_", sum(prop_targets), "_AGE.h5ad")
heartbeat(sprintf("Writing Proportional dataset: %s\n", out_prop), TRUE)
zellkonverter::writeH5AD(rebuild_for_write(sce_prop), out_prop, compression = "gzip")

rm(sce_for_prop); gc()

# 2) Balanced Augmented (copy of full combined)
out_bal_aug <- paste0(OUTPUT_BASE, "_BalancedAugmented_", TARGET_PER_BIN, "Each_AGE.h5ad")
heartbeat(sprintf("Copying Full augmented to BalancedAugmented: %s\n", out_bal_aug), TRUE)
zellkonverter::writeH5AD(rebuild_for_write(sce_combined), out_bal_aug, compression = "gzip")

# 3) Balanced Upsampled (real-only, all bins to TARGET_PER_BIN)
heartbeat("Building Balanced Upsampled (real-only) dataset\n", TRUE)

bin_real_vec <- as.character(colData(sce_real)[[BIN_COL]])

up_idx <- unlist(lapply(all_bins, function(b) {
  sample_exact(which(bin_real_vec == b), TARGET_PER_BIN, replace = TRUE)
}))
sce_up <- sce_real[, up_idx]

out_up <- paste0(OUTPUT_BASE, "_BalancedUpsampled_", TARGET_PER_BIN, "Each_AGE.h5ad")
heartbeat(sprintf("Writing Balanced Upsampled: %s\n", out_up), TRUE)
zellkonverter::writeH5AD(rebuild_for_write(sce_up), out_up, compression = "gzip")

# 4) Downsampled (real-only, all bins to minority bin size)
heartbeat("Building Downsampled (real-only) dataset\n", TRUE)

bin_real_tbl <- table(bin_real_vec)
DOWN_TARGET  <- min(as.numeric(bin_real_tbl))

down_idx <- unlist(lapply(all_bins, function(b) {
  sample_exact(which(bin_real_vec == b), DOWN_TARGET, replace = FALSE)
}))
sce_down <- sce_real[, down_idx]

out_down <- paste0(OUTPUT_BASE, "_Downsampled_", DOWN_TARGET, "Each_AGE.h5ad")
heartbeat(sprintf("Writing Downsampled: %s\n", out_down), TRUE)
zellkonverter::writeH5AD(rebuild_for_write(sce_down), out_down, compression = "gzip")

# ============================================================
# 15. FINAL VALIDATION
# ============================================================

heartbeat("\n=== FINAL VALIDATION ===\n", TRUE)

final_counts <- table(colData(sce_combined)[[BIN_COL]])
heartbeat("Final age bin sizes (Balanced Augmented full):\n", TRUE)
for (b in names(final_counts)) {
  heartbeat(sprintf("%s: %d cells (target: %d)", b, final_counts[b], TARGET_PER_BIN), TRUE)
}

heartbeat(sprintf("Age bin counts in Proportional (~%d) dataset:\n", PROPORTIONAL_SIZE), TRUE)
heartbeat(capture.output(print(grp_counts(sce_prop))), TRUE)

heartbeat("Age bin counts in BalancedAugmented (Full) dataset:\n", TRUE)
heartbeat(capture.output(print(grp_counts(sce_combined))), TRUE)

heartbeat("Age bin counts in BalancedUpsampled (real-only) dataset:\n", TRUE)
heartbeat(capture.output(print(grp_counts(sce_up))), TRUE)

heartbeat("Age bin counts in Downsampled (real-only) dataset:\n", TRUE)
heartbeat(capture.output(print(grp_counts(sce_down))), TRUE)

# ============================================================
# 16. FINAL REPORT
# ============================================================

heartbeat("\n=== AGE AUGMENTATION V5 (PILOT) COMPLETE ===\n", TRUE)
heartbeat("Output files:\n", TRUE)
heartbeat(sprintf(" 1. %s (BalancedAugmented Full)\n",          out_full),    TRUE)
heartbeat(sprintf(" 2. %s (RealOnly component of Full)\n",      out_real),    TRUE)
heartbeat(sprintf(" 3. %s (SyntheticOnly component of Full)\n", out_syn),     TRUE)
heartbeat(sprintf(" 4. %s (Summary CSV)\n",                     out_sum),     TRUE)
heartbeat(sprintf(" 5. %s (Proportional ~%d, real-only)\n",     out_prop, PROPORTIONAL_SIZE), TRUE)
heartbeat(sprintf(" 6. %s (Balanced Augmented)\n",              out_bal_aug), TRUE)
heartbeat(sprintf(" 7. %s (Balanced Upsampled, real-only)\n",   out_up),      TRUE)
heartbeat(sprintf(" 8. %s (Downsampled, real-only)\n",          out_down),    TRUE)
heartbeat(sprintf("\nConfig summary:\n"), TRUE)
heartbeat(sprintf(" AGE_COL              = %s\n", AGE_COL),              TRUE)
heartbeat(sprintf(" BIN_COL              = %s\n", BIN_COL),              TRUE)
heartbeat(sprintf(" VALIDATION_H5AD      = %s\n", VALIDATION_H5AD),     TRUE)
heartbeat(sprintf(" AGE_LABELS           = %s\n", paste(AGE_LABELS, collapse = ", ")), TRUE)
heartbeat(sprintf(" PROPORTIONAL_SIZE    = %d\n", PROPORTIONAL_SIZE),    TRUE)
heartbeat(sprintf(" TARGET_PER_BIN       = %d\n", TARGET_PER_BIN),       TRUE)
heartbeat(sprintf(" CHUNK_SIZE_DEFAULT   = %d\n", CHUNK_SIZE_DEFAULT),   TRUE)
heartbeat(sprintf(" MAX_CELLS_PER_BIN    = %d\n", MAX_CELLS_PER_BIN),    TRUE)
heartbeat(sprintf(" ANCHOR_PER_CT_PER_BIN= %d\n", ANCHOR_PER_CT_PER_BIN), TRUE)
heartbeat(sprintf(" MIN_CELLS_INTERACTION= %d\n", MIN_CELLS_INTERACTION), TRUE)
heartbeat(sprintf(" FAMILY_USE           = zinb [V5 5]\n"),              TRUE)
heartbeat(sprintf(" N_BINS_DETECTED      = %d (%s)\n", length(all_bins),
                  paste(all_bins, collapse = ", ")),                      TRUE)
heartbeat("\nAll done! Check augmentation_age_pilot_anchors_log.txt for details.\n", TRUE)