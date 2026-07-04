#!/usr/bin/env Rscript
# ============================================================
# STEP 0B: ETHNICITY AUGMENTATION WITH scDesign3 (PILOT, v5)
# Geneformer workflow — augmented/ethnicity_Geneformer_workflow
# ============================================================
# * Group by self_reported_ethnicity (N groups detected dynamically)
# * Majority group: real-only, down-sampled to TARGET_PER_BIN
# * Minority groups: real down-sampled proportionally, then
#   scDesign3-augmented up to TARGET_PER_BIN
# * 1,000 HVGs, stratified training sets
# * Model:
#   - family_use = "zinb"           [V5 5]
#   - assay = "counts"
#   - mu_formula via build_mu_formula() [V5 6/7/8]:
#       "cell_type + self_reported_ethnicity
#        + cell_type:self_reported_ethnicity
#        + sex + age_bin_10yr + offset(log(library))"
#       sex / age_bin_10yr dropped if <2 levels (soft-degrade)
#   - sigma_formula = "cell_type"
#   - corr_formula  = "cell_type"
#   - copula        = "gaussian"
#
# * ANCHOR-CELL ARCHITECTURE [FIX 17]
# * LOW-LEVEL API, TIER 1 ONLY [FIX 21]
# * Fallback: naive resampling if scDesign3 fails after MAX_RETRIES.
#
# ============================================================
# V5 CHANGES vs. augmentedv4 ethnicity script
# ============================================================
#   [V5 1]  VALIDATION_H5AD: ILD_Ethnicity_External_Validation_12500.h5ad
#   [V5 2]  load_validation_barcodes() + exclude_validation_cells()
#   [V5 3]  exclude_validation_cells() at all readH5AD(INPUT_H5AD) sites
#   [V5 4]  LIBRARY_SIZES global: full 31,432-gene colSums pre-computed.
#           load_bin_fresh() attaches colData$library by barcode lookup.
#   [V5 5]  family_use = "zinb" at all 4 sites in run_sd3_lowlevel().
#   [V5 6]  mu_formula extended: + sex + age_bin_10yr + offset(log(library))
#   [V5 7]  construct_data other_covariates: c(BIN_COL, "sex", "age_bin_10yr", "library")
#   [V5 8]  build_mu_formula() soft-degrade: drops sex and/or age_bin_10yr
#           if <2 levels in the joint training set. Always includes library.
#   [V5 9]  add_age_bins() helper: parses development_stage -> age_bin_10yr.
#           Called after every readH5AD() so age is available as a covariate.
#   [GF 1]  BASE PATH: augmentedv4/ethnicity_scfoundation_workflow
#                   -> Geneformer/augmented/ethnicity_Geneformer_workflow
#
# All FIX patches (1-22) from prior scripts preserved verbatim.
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
cat(sprintf("Parallelization disabled: using %d core (serial)\n", N_CORES), file = stderr())
blas_set_num_threads(N_CORES)
message("[INFO] OpenBLAS/MKL now using ", N_CORES, " threads")
register(SerialParam(progressbar = FALSE))

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_H5AD      <- "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"
VALIDATION_H5AD <- "ILD_Ethnicity_External_Validation_12500.h5ad"   # [V5 1]
OUTPUT_BASE     <- "ILD_Ethnicity_Pilot"
CKPTDIR         <- "checkpoints_ethnicity_augmentation_pilot_v5"
LOGFILE         <- "augmentation_ethnicity_pilot_v5_log.txt"

BIN_COL      <- "self_reported_ethnicity"
CELLTYPE_COL <- "cell_type"
AGE_COL      <- "development_stage"    # [V5 9] source text column for age
AGE_BIN_COL  <- "age_bin_10yr"         # [V5 9] derived bin column

UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                    "multiethnic", "na na", "not applicable", "prefer not to say")
CT_UNKNOWN     <- c("unknown", "na", "n/a", "not reported", "", "nan")

# [V5 9] Age bin configuration (matching age workflow exactly)
AGE_UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                         "not applicable", "adult", "child", "infant",
                         "embryonic", "fetal", "newborn")
AGE_BREAKS <- seq(10, 90, by = 10)
AGE_LABELS <- paste0(AGE_BREAKS[-length(AGE_BREAKS)], "_", AGE_BREAKS[-1] - 1)

PROPORTIONAL_SIZE  <- 2500L
CHUNK_SIZE_DEFAULT <- 200L
CHUNK_SIZE_MIN     <- 100L
MAX_RETRIES        <- 3L
BACKOFF_FACTOR     <- 0.7

N_HVG            <- 1000L
MIN_CELLS        <- 5L
MIN_COUNTS       <- 10L
HVG_SAMPLE_SIZE  <- 5000L

MIN_CT_CELLS      <- 20L
MAX_CELLS_PER_BIN <- 500L
MIN_CELLS_INTERACTION <- 50L
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

heartbeat("\n=== STARTING ETHNICITY AUGMENTATION v5 (Geneformer workflow) ===\n", TRUE)
heartbeat(sprintf("Memory protection: MAX_CELLS_PER_BIN = %d\n", MAX_CELLS_PER_BIN), TRUE)
heartbeat(sprintf("Anchor cells per CT per group: %d\n", ANCHOR_PER_CT_PER_BIN), TRUE)
heartbeat(sprintf("Min cells for interaction: %d\n", MIN_CELLS_INTERACTION), TRUE)

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
# [V5 9] ADD_AGE_BINS HELPER
# Parses development_stage text -> 10yr bin. Adds AGE_BIN_COL to colData.
# Does NOT filter out cells with missing age — just sets NA for those.
# This is a control covariate; build_mu_formula() will soft-degrade it
# if <2 levels are present.
# ============================================================

add_age_bins <- function(sce_obj) {
  if (!AGE_COL %in% colnames(colData(sce_obj))) {
    heartbeat(sprintf(" [V5 9] '%s' not found; age_bin_10yr will be NA for all cells.\n", AGE_COL), TRUE)
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
    cut_result          <- cut(age_num[in_range], breaks = AGE_BREAKS,
                               labels = AGE_LABELS, include.lowest = TRUE, right = FALSE)
    age_bin[in_range]   <- as.character(droplevels(cut_result))
  }

  colData(sce_obj)[[AGE_BIN_COL]] <- age_bin
  n_with_age <- sum(!is.na(age_bin))
  heartbeat(sprintf(" [V5 9] age_bin_10yr: %d cells have valid age bin, %d NA.\n",
                    n_with_age, ncol(sce_obj) - n_with_age), TRUE)
  sce_obj
}

# ------------------------------------------------------------
# INTEGER-COUNT + ASSAY HELPERS
# ------------------------------------------------------------

assert_raw_counts <- function(sce_obj, assay_name = "counts") {
  if (!assay_name %in% assayNames(sce_obj)) {
    stop(sprintf("Assay '%s' not found.", assay_name))
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
    vals_all <- mat[cbind(sample.int(nr, samp_n, replace = TRUE),
                          sample.int(nc, samp_n, replace = TRUE))]
  }
  if (length(vals_all) == 0L) {
    message(sprintf("Assay '%s' is all zeros or extremely sparse.", assay_name))
    return(invisible(TRUE))
  }
  n    <- min(10000L, length(vals_all))
  vals <- vals_all[seq_len(n)]
  if (any(abs(vals - round(vals)) > .Machine$double.eps^0.5)) {
    bad <- vals[which.max(abs(vals - round(vals)))]
    stop(sprintf("Assay '%s' contains non-integer values (e.g., %0.6f).", assay_name, bad))
  }
  message(sprintf("Assay '%s' passed integer-count sanity check (%d entries).", assay_name, n))
  invisible(TRUE)
}

ensure_counts_assay <- function(sce_obj) {
  a <- assayNames(sce_obj)
  if ("counts" %in% a) {
    assert_raw_counts(sce_obj, "counts")
  } else if ("X" %in% a) {
    assert_raw_counts(sce_obj, "X")
    assay(sce_obj, "counts") <- assay(sce_obj, "X")
    message("No 'counts' assay; using 'X' after integer check.")
    assert_raw_counts(sce_obj, "counts")
  } else {
    stop("Neither 'counts' nor 'X' assay present.")
  }
  assays(sce_obj) <- SimpleList(counts = assay(sce_obj, "counts"))
  mat <- assay(sce_obj, "counts")
  if (!inherits(mat, "dgCMatrix")) {
    message(sprintf("Coercing '%s' to dgCMatrix.", class(mat)[1]))
    assay(sce_obj, "counts") <- as(mat, "dgCMatrix")
  }
  sce_obj
}

# ============================================================
# [V5 4] PRE-COMPUTE FULL-GENE LIBRARY SIZES
# Must be after ensure_counts_assay().
# ============================================================

heartbeat("[V5 4] Pre-computing full-gene library sizes...\n", TRUE)
sce_libsize_tmp <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_libsize_tmp <- ensure_counts_assay(sce_libsize_tmp)
LIBRARY_SIZES   <- Matrix::colSums(assay(sce_libsize_tmp, "counts"))
heartbeat(sprintf("[V5 4] lib size (full genes) mean=%.0f sd=%.0f (%d cells)\n",
                  mean(LIBRARY_SIZES), sd(LIBRARY_SIZES), length(LIBRARY_SIZES)), TRUE)
rm(sce_libsize_tmp); gc()

# ============================================================
# 1. AUTO-DETECT ETHNICITY GROUP STRUCTURE
# ============================================================

heartbeat("\n=== AUTO-DETECTING ETHNICITY GROUP STRUCTURE ===\n", TRUE)

sce_detect <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_detect <- ensure_counts_assay(sce_detect)
sce_detect <- exclude_validation_cells(sce_detect)   # [V5 3]

eth_raw    <- tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]])))
valid      <- !(eth_raw %in% UNKNOWN_VALUES) & !is.na(eth_raw)
sce_detect <- sce_detect[, valid]
colData(sce_detect)[[BIN_COL]] <- as.character(
  tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]])))
)

bin_table <- table(colData(sce_detect)[[BIN_COL]])
all_bins  <- sort(names(bin_table))

heartbeat("Real dataset ethnicity group sizes (validation-excluded):\n", TRUE)
for (b in all_bins) heartbeat(sprintf("%s: %d cells", b, bin_table[b]), TRUE)

MAJ_BIN <- all_bins[which.max(bin_table)]
MIN_BIN <- all_bins[which.min(bin_table)]
heartbeat(sprintf("\n>>> MAJORITY: %s (%d cells)\n", MAJ_BIN, bin_table[MAJ_BIN]), TRUE)
heartbeat(sprintf(">>> MINORITY: %s (%d cells)\n", MIN_BIN, bin_table[MIN_BIN]), TRUE)

# ============================================================
# 2. COMPUTE PROPORTIONAL TARGETS
# ============================================================

heartbeat(sprintf("\n=== COMPUTING PROPORTIONAL TARGETS (PILOT, %d CELLS) ===\n", PROPORTIONAL_SIZE), TRUE)

total_real <- sum(as.numeric(bin_table))
heartbeat(sprintf("Total cells (validation-excluded): %d\n", total_real), TRUE)

if (total_real < PROPORTIONAL_SIZE) {
  stop(sprintf("Not enough cells: have %d, need %d", total_real, PROPORTIONAL_SIZE))
}

prop_targets <- sapply(as.numeric(bin_table), function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins

heartbeat(sprintf("Proportional targets (%d sample):\n", PROPORTIONAL_SIZE), TRUE)
for (b in all_bins) heartbeat(sprintf("%s: %d cells", b, prop_targets[b]), TRUE)

TARGET_PER_BIN <- prop_targets[MAJ_BIN]
TARGET_DOWN    <- prop_targets[MIN_BIN]
if (TARGET_PER_BIN < TARGET_DOWN) {
  tmp <- TARGET_PER_BIN; TARGET_PER_BIN <- TARGET_DOWN; TARGET_DOWN <- tmp
}

heartbeat(sprintf("\n>>> AUGMENTATION TARGET (per group): %d cells\n", TARGET_PER_BIN), TRUE)
heartbeat(sprintf(">>> DOWN-SAMPLING TARGET (per group): %d cells\n", TARGET_DOWN), TRUE)

# ============================================================
# 3. DIRECTORIES
# ============================================================

if (!dir.exists(CKPTDIR)) dir.create(CKPTDIR, recursive = TRUE)

# ============================================================
# 4. HVG SELECTION
# ============================================================

heartbeat(sprintf("\n=== HVG SELECTION (target %d cells) ===\n", HVG_SAMPLE_SIZE), TRUE)

sce_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_full <- ensure_counts_assay(sce_full)
sce_full <- exclude_validation_cells(sce_full)   # [V5 3]

eth_full   <- tolower(trimws(as.character(colData(sce_full)[[BIN_COL]])))
valid_full <- !(eth_full %in% UNKNOWN_VALUES) & !is.na(eth_full)
sce_full   <- sce_full[, valid_full]
colData(sce_full)[[BIN_COL]] <- as.character(
  tolower(trimws(as.character(colData(sce_full)[[BIN_COL]])))
)

original_genes  <- rownames(sce_full)
heartbeat(sprintf("Full cleaned dataset: %d cells x %d genes\n", ncol(sce_full), nrow(sce_full)), TRUE)

bin_labels_full <- colData(sce_full)[[BIN_COL]]
bin_counts_full <- table(bin_labels_full)

target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
n_per_bin_hvg      <- pmin(as.numeric(bin_counts_full), target_per_bin_hvg)
names(n_per_bin_hvg) <- names(bin_counts_full)

sample_idx_hvg <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_full == b)
  if (length(idx) > n_per_bin_hvg[b]) sample(idx, n_per_bin_hvg[b]) else idx
}))
heartbeat(sprintf("Sampled %d cells across %d groups\n", length(sample_idx_hvg), length(all_bins)), TRUE)
rm(bin_counts_full, bin_labels_full); gc()

sce_sample <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_sample <- ensure_counts_assay(sce_sample)
sce_sample <- exclude_validation_cells(sce_sample)   # [V5 3]
eth_sample   <- tolower(trimws(as.character(colData(sce_sample)[[BIN_COL]])))
valid_sample <- !(eth_sample %in% UNKNOWN_VALUES) & !is.na(eth_sample)
sce_sample   <- sce_sample[, valid_sample]
sce_sample   <- sce_sample[, sample_idx_hvg]

count_mat    <- assay(sce_sample, ASSAY_USE)
gene_sums    <- Matrix::rowSums(count_mat)
gene_nonzero <- Matrix::rowSums(count_mat > 0)
keep_genes   <- (gene_sums >= MIN_COUNTS) & (gene_nonzero >= MIN_CELLS)
n_kept       <- sum(keep_genes)
heartbeat(sprintf("Genes after filtering: %d / %d\n", n_kept, length(gene_sums)), TRUE)
if (n_kept == 0) stop("ERROR: No genes passed filtering!")

sce_filt   <- sce_sample[keep_genes, ]
filt_mat   <- assay(sce_filt, ASSAY_USE)
gene_means <- Matrix::rowMeans(filt_mat)
filt_mat_sq <- filt_mat; filt_mat_sq@x <- filt_mat_sq@x^2
gene_means_sq <- Matrix::rowMeans(filt_mat_sq)
gene_vars  <- pmax(gene_means_sq - gene_means^2, 0)
cv2        <- gene_vars / (gene_means^2 + 1e-8)
cv2[is.na(cv2) | is.infinite(cv2)] <- 0

n_select  <- min(N_HVG, length(cv2))
top_idx   <- order(cv2, decreasing = TRUE)[seq_len(n_select)]
hvg_names <- rownames(sce_filt)[top_idx]
heartbeat(sprintf("Selected %d HVGs (requested %d)\n", length(hvg_names), N_HVG), TRUE)
if (length(hvg_names) == 0) stop("ERROR: HVG selection failed!")

# [GF VOCAB] Restrict HVGs to Geneformer vocabulary genes only
gf_vocab_path <- file.path(dirname(INPUT_H5AD), "geneformer_vocab_genes.txt")
if (file.exists(gf_vocab_path)) {
  gf_vocab     <- readLines(gf_vocab_path)
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
   gene_means, gene_means_sq, gene_vars, cv2, top_idx, keep_genes, filt_mat_sq); gc()

# ============================================================
# 5. COLDATA SANITIZER
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
      if (is.null(xi) || length(xi) == 0) {
        result[[i]] <- NA_character_
      } else if (length(xi) == 1 && is.na(xi)) {
        result[[i]] <- NA_character_
      } else {
        result[[i]] <- tryCatch(paste(as.character(xi), collapse = ";"),
                                error = function(e) NA_character_)
      }
    }
    result
  })
  names(new_cols) <- colnames(cd)
  colData(sce_obj) <- DataFrame(new_cols, row.names = rn)
  sce_obj
}

# ============================================================
# 6. LOAD BIN FRESH
# ============================================================

load_bin_fresh <- function(bin_label) {
  heartbeat(sprintf("Loading group '%s' (HVG subset)...", bin_label), TRUE)

  sce_raw <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
  common_early <- intersect(hvg_names, rownames(sce_raw))
  if (length(common_early) == 0) stop(sprintf("No HVGs found for %s", bin_label))
  sce_raw <- sce_raw[common_early, ]
  sce_raw <- ensure_counts_assay(sce_raw)
  sce_raw <- exclude_validation_cells(sce_raw)   # [V5 3]
  sce_raw <- add_age_bins(sce_raw)               # [V5 9]

  eth_raw_vec <- tolower(trimws(as.character(colData(sce_raw)[[BIN_COL]])))
  valid_eth   <- !(eth_raw_vec %in% UNKNOWN_VALUES) & !is.na(eth_raw_vec)
  sce_raw     <- sce_raw[, valid_eth]

  ct_raw   <- as.character(colData(sce_raw)[[CELLTYPE_COL]])
  valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
  sce_raw  <- sce_raw[, valid_ct]

  eth_vec  <- tolower(trimws(as.character(colData(sce_raw)[[BIN_COL]])))
  bin_mask <- eth_vec == bin_label
  sce_bin  <- sce_raw[, bin_mask]

  if (ncol(sce_bin) == 0) stop(sprintf("No cells for group '%s'", bin_label))

  common  <- intersect(hvg_names, rownames(sce_bin))
  if (length(common) == 0) stop(sprintf("No HVGs in group '%s'", bin_label))
  sce_bin <- sce_bin[common, ]

  ct_vec  <- as.character(colData(sce_bin)[[CELLTYPE_COL]])
  tab_ct  <- table(ct_vec)
  keep_ct <- names(tab_ct)[tab_ct >= MIN_CT_CELLS]
  if (length(keep_ct) == 0) {
    stop(sprintf("No cell types in group '%s' with >= %d cells", bin_label, MIN_CT_CELLS))
  }

  cells_keep <- ct_vec %in% keep_ct
  sce_bin    <- sce_bin[, cells_keep]
  ct_vec     <- ct_vec[cells_keep]

  ct_vec[is.na(ct_vec)] <- "unknown_cleaned"
  colData(sce_bin)[[CELLTYPE_COL]] <- droplevels(factor(ct_vec))
  colData(sce_bin)[[BIN_COL]]      <- as.character(colData(sce_bin)[[BIN_COL]])

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
# [V5 8] BUILD_MU_FORMULA for ETHNICITY
# Primary covariate is self_reported_ethnicity (always included).
# Control covariates sex and age_bin_10yr soft-degrade if <2 levels.
# ============================================================

build_mu_formula <- function(train_sce) {
  formula_parts <- c("cell_type", "self_reported_ethnicity",
                     "cell_type:self_reported_ethnicity")
  other_extra   <- character(0)

  for (covar in c("sex", AGE_BIN_COL)) {
    if (!covar %in% colnames(colData(train_sce))) {
      heartbeat(sprintf(" [V5 8] '%s' absent from colData -> skipped.\n", covar), TRUE)
      next
    }
    n_levels <- length(unique(na.omit(as.character(colData(train_sce)[[covar]]))))
    if (n_levels < 2L) {
      heartbeat(sprintf(" [V5 8] '%s' has %d level(s) -> soft-degraded.\n",
                        covar, n_levels), TRUE)
      next
    }
    formula_parts <- c(formula_parts, covar)
    other_extra   <- c(other_extra, covar)
    heartbeat(sprintf(" [V5 8] '%s' has %d levels -> included.\n", covar, n_levels), TRUE)
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
  if (is.null(new_count)) stop("scDesign3 returned NULL count matrix")
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(new_count, nrow = length(new_count), ncol = 1,
                        dimnames = list(names(new_count), NULL))
  }
  if (!is.null(rownames(new_count)) && !is.null(ref_genes)) {
    if (ncol(new_count) == length(ref_genes) && nrow(new_count) != length(ref_genes)) {
      new_count <- t(new_count)
    }
  }
  if (ncol(new_count) == 0) stop("scDesign3 returned empty count matrix")
  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(0, nrow = length(missing), ncol = ncol(new_count),
                  dimnames = list(missing, colnames(new_count)))
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop = FALSE]
  colnames(new_count) <- paste0("synthetic_", bin_label, "_chunk", chunk_id, "_",
                                seq_len(ncol(new_count)))
  new_count
}

# ============================================================
# 8. H5AD WRITE HELPERS
# ============================================================

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
    heartbeat(sprintf(" Dropped %d all-NA colData columns\n", sum(!not_all_na)), TRUE)
  }

  rn_safe        <- make.unique(as.character(colnames(sce_obj)), sep = "_dup")
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
# 8b. PRE-CACHE ANCHOR CELLS  [FIX 17]
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
    heartbeat(sprintf(" WARNING: anchor cache FAILED for '%s': %s\n", b, conditionMessage(e)), TRUE)
  })
}

build_joint_train <- function(sub_train_target, target_bin) {
  other_anchors <- anchor_cache[setdiff(names(anchor_cache), target_bin)]
  if (length(other_anchors) == 0) {
    heartbeat(" WARNING: no anchor groups; using target only.\n", TRUE)
    return(sub_train_target)
  }

  # [FIX 20]
  target_cts    <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
  other_anchors <- lapply(other_anchors, function(a) {
    ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
    a[, ct_a %in% target_cts, drop = FALSE]
  })
  other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]
  if (length(other_anchors) == 0) {
    heartbeat(" WARNING: no anchors share CTs with target; using target only.\n", TRUE)
    return(sub_train_target)
  }
  heartbeat(sprintf(" [FIX 20] Anchors subsetted to %d CTs; %d groups remain.\n",
                    length(target_cts), length(other_anchors)), TRUE)

  common_genes <- rownames(sub_train_target)
  for (sce_a in other_anchors) common_genes <- intersect(common_genes, rownames(sce_a))
  if (length(common_genes) == 0) {
    heartbeat(" WARNING: no common genes with anchors; skipping injection.\n", TRUE)
    return(sub_train_target)
  }

  pieces <- c(list(sub_train_target[common_genes, ]),
              lapply(other_anchors, function(a) a[common_genes, ]))
  joint  <- do.call(cbind, pieces)

  # [FIX 19] singleton CT guard
  ct_joint  <- as.character(colData(joint)[[CELLTYPE_COL]])
  tab_joint <- table(ct_joint)
  keep_ct   <- names(tab_joint)[tab_joint >= 2L]
  if (length(keep_ct) < length(tab_joint)) {
    n_dropped <- sum(tab_joint[tab_joint < 2L])
    heartbeat(sprintf(" [FIX 19] Dropping %d cells from %d singleton CT(s).\n",
                      n_dropped, sum(tab_joint < 2L)), TRUE)
    joint    <- joint[, ct_joint %in% keep_ct]
    ct_clean <- as.character(colData(joint)[[CELLTYPE_COL]])
    colData(joint)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
  }

  colData(joint)[[BIN_COL]] <- as.character(colData(joint)[[BIN_COL]])

  n_target <- ncol(sub_train_target)
  n_anchor <- ncol(joint) - n_target
  n_grps   <- length(unique(as.character(colData(joint)[[BIN_COL]])))
  n_cts    <- length(unique(as.character(colData(joint)[[CELLTYPE_COL]])))
  heartbeat(sprintf(" Joint training: %d target + %d anchor = %d total, %d groups, %d CTs\n",
                    n_target, n_anchor, ncol(joint), n_grps, n_cts), TRUE)
  joint
}

# ============================================================
# 9. AUGMENTATION FOR ONE ETHNICITY GROUP
# ============================================================

augment_one_bin <- function(bin_label) {
  ckpt_file <- file.path(CKPTDIR,
    paste0("ethnicity_", gsub("[^a-zA-Z0-9_]", "_", bin_label), ".rds"))

  if (file.exists(ckpt_file)) {
    heartbeat(sprintf("Checkpoint found for '%s' -- loading.\n", bin_label), TRUE)
    sce_ckpt <- readRDS(ckpt_file)
    sce_ckpt <- sanitize_coldata_for_h5ad(sce_ckpt)
    return(sce_ckpt)
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

  target_real_min <- prop_targets[bin_label]
  if (is.na(target_real_min) || target_real_min <= 0) {
    target_real_min <- min(n_full, TARGET_PER_BIN)
  }

  if (n_full > target_real_min) {
    set.seed(123)
    sub_real <- sub_real_full[, sample(n_full, target_real_min)]
  } else {
    sub_real <- sub_real_full
  }
  assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")

  n_real <- ncol(sub_real)
  need   <- TARGET_PER_BIN - n_real

  if (need <= 0) {
    heartbeat(sprintf("Group '%s' already at target (n=%d).\n", bin_label, n_real), TRUE)
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  heartbeat(sprintf("Group '%s': augmenting %d -> %d (need %d synthetic).\n",
                    bin_label, n_real, TARGET_PER_BIN, need), TRUE)

  sims      <- list()
  remaining <- need
  chunk_id  <- 1L

  while (remaining > 0) {
    chunk   <- min(CHUNK_SIZE_DEFAULT, remaining)
    attempt <- 1L
    success <- FALSE

    while (attempt <= MAX_RETRIES && !success) {
      heartbeat(sprintf("Augment '%s' | chunk %d | attempt %d | n=%d | remaining=%d",
                        bin_label, chunk_id, attempt, chunk, remaining), TRUE)

      sub_train      <- load_bin_fresh(bin_label)
      n_train_loaded <- ncol(sub_train)

      n_ct_train <- length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
      min_chunk  <- 2L * n_ct_train
      if (chunk < min_chunk) {
        chunk <- min_chunk
        heartbeat(sprintf(" Bumping chunk to %d (>= 2 x %d CTs)\n", chunk, n_ct_train), TRUE)
      }

      if (n_train_loaded > MAX_CELLS_PER_BIN) {
        ct_vec <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        tab_ct <- table(ct_vec); tab_ct <- tab_ct[tab_ct > 0]
        cap    <- as.integer(MAX_CELLS_PER_BIN)
        alloc  <- pmax(2L, as.integer(floor(cap * as.numeric(tab_ct) / sum(tab_ct))))
        names(alloc) <- names(tab_ct)
        alloc  <- pmin(alloc, as.integer(tab_ct))
        while (sum(alloc) > cap) {
          red <- names(alloc)[alloc > 2L]
          if (length(red) == 0) break
          lg <- red[which.max(alloc[red])]; alloc[lg] <- alloc[lg] - 1L
        }
        keep_idx  <- unlist(lapply(names(alloc), function(ct) {
          ids <- which(ct_vec == ct); sample(ids, alloc[[ct]], replace = FALSE)
        }))
        sub_train <- sub_train[, keep_idx, drop = FALSE]
        ct_new    <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        ct_new[is.na(ct_new)] <- "unknown_cleaned"
        colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_new))
        heartbeat(sprintf(" -> Stratified sample: %d cells, %d CTs\n",
                          ncol(sub_train),
                          length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))), TRUE)
      }

      ct_final <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      ct_final[is.na(ct_final)] <- "unknown_cleaned"
      colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_final))
      ct_counts_train <- table(as.character(colData(sub_train)[[CELLTYPE_COL]]))
      if (any(ct_counts_train < 2L)) {
        stop(sprintf("Training subset for '%s' has CT(s) with <2 cells: %s",
                     bin_label,
                     paste(names(ct_counts_train)[ct_counts_train < 2L], collapse = ", ")))
      }

      # [FIX 17] Build joint training set
      sub_train_target <- sub_train
      sub_train        <- build_joint_train(sub_train_target, bin_label)
      ct_counts_train  <- table(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))

      # [V5 6/7/8] Build mu_formula with soft-degrade
      mu_spec         <- build_mu_formula(sub_train)
      mu_formula_str  <- mu_spec$formula
      other_covars_v5 <- mu_spec$other_covariates

      # [FIX 18] Adaptive sparsity threshold
      n_target_cells     <- ncol(sub_train_target)
      mat_for_imp        <- assay(sub_train, ASSAY_USE)
      zero_frac          <- Matrix::rowMeans(mat_for_imp == 0)
      sparsity_threshold <- if (n_target_cells < MIN_CELLS_INTERACTION) 0.95 else 0.80
      imp_feat_vec       <- zero_frac <= sparsity_threshold
      if (sum(imp_feat_vec) < 2L) {
        heartbeat(" WARNING: <2 genes pass sparsity filter; using all genes.\n", TRUE)
        imp_feat_vec <- rep(TRUE, length(imp_feat_vec))
      }
      heartbeat(sprintf(" important_feature: %d / %d genes (threshold=%.2f)\n",
                        sum(imp_feat_vec), length(imp_feat_vec), sparsity_threshold), TRUE)

      # Ethnicity variance check
      eth_vals    <- unique(as.character(colData(sub_train)[[BIN_COL]]))
      has_eth_var <- length(eth_vals) > 1L
      if (!has_eth_var) {
        stop(sprintf("[Tier 1 gate] No ethnicity variance for '%s'.", bin_label))
      }
      heartbeat(sprintf(" Groups in joint training: %s\n", paste(eth_vals, collapse = ", ")), TRUE)

      # CT x group gate
      ct_eth_tab <- table(CT  = as.character(colData(sub_train)[[CELLTYPE_COL]]),
                          Grp = as.character(colData(sub_train)[[BIN_COL]]))
      ct_n_grps  <- rowSums(ct_eth_tab > 0)
      single_ct  <- names(ct_n_grps)[ct_n_grps < 2L]
      if (length(single_ct) > 0) {
        stop(sprintf("[Tier 1 gate] %d CT(s) in only 1 group: %s",
                     length(single_ct), paste(single_ct, collapse = ", ")))
      }

      # ----------------------------------------------------------------
      # [FIX 21] LOW-LEVEL API — TIER 1 ONLY
      # ----------------------------------------------------------------
      run_sd3_lowlevel <- function(mu_f, corr_f) {
        heartbeat(sprintf(" [low-level] mu=%-70s corr=%-12s\n", mu_f, corr_f), TRUE)
        tryCatch({
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

          # [FIX 21] new_covariate from TARGET-GROUP ROWS ONLY
          target_dat_rows <- which(
            as.character(joint_data$dat[[BIN_COL]]) == bin_label
          )
          if (length(target_dat_rows) == 0) {
            stop(sprintf("[FIX 21] No target-group rows for '%s'", bin_label))
          }
          new_cov_idx <- sample(target_dat_rows, chunk,
                                replace = (length(target_dat_rows) < chunk))
          new_cov_df  <- joint_data$dat[new_cov_idx, , drop = FALSE]

          # [FIX 22] Ensure >= 2 rows per CT
          cov_cts    <- as.character(new_cov_df[[CELLTYPE_COL]])
          ct_tab_cov <- table(cov_cts)
          thin_cts   <- names(ct_tab_cov)[ct_tab_cov < 2L]
          if (length(thin_cts) > 0) {
            extra_rows <- lapply(thin_cts, function(ct) {
              ct_rows <- target_dat_rows[
                as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct
              ]
              if (length(ct_rows) == 0L) return(NULL)
              joint_data$dat[sample(ct_rows, 2L - ct_tab_cov[[ct]], replace = TRUE), , drop = FALSE]
            })
            extra_rows <- do.call(rbind, Filter(Negate(is.null), extra_rows))
            if (!is.null(extra_rows) && nrow(extra_rows) > 0) {
              new_cov_df <- rbind(new_cov_df, extra_rows)
              heartbeat(sprintf(" [FIX 22] Added %d row(s) to new_cov_df.\n", nrow(extra_rows)), TRUE)
            }
          }

          if (corr_f == "1") {
            new_cov_df$corr_group <- 1L
          } else {
            new_cov_df$corr_group <- as.character(new_cov_df[[CELLTYPE_COL]])
          }
          rownames(new_cov_df) <- paste0("Cell", seq_len(nrow(new_cov_df)))

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

      heartbeat(" [Tier 1] ethnicity+interaction+sex+age+offset(lib), corr=cell_type [V5]\n", TRUE)
      res <- run_sd3_lowlevel(mu_formula_str, "cell_type")

      if (is.null(res)) {
        heartbeat(sprintf(
            " [Tier 1 FAILED] '%s' chunk %d attempt %d returned NULL. Retrying.\n",
            bin_label, chunk_id, attempt
        ), TRUE)
        attempt <- attempt + 1L
        next
    }

      new_counts <- validate_new_counts(res$new_count, rownames(sub_real), chunk_id, bin_label)
      new_cov    <- res$new_covariate
      if (is.null(new_cov)) {
        new_cov <- DataFrame(row.names = colnames(new_counts))
      } else {
        rownames(new_cov) <- colnames(new_counts)
      }

      sim <- SingleCellExperiment(assays = list(counts = new_counts), colData = new_cov)

      # [FIX 13] Align metadata
      ref_cols     <- colnames(colData(sub_real))
      missing_cols <- setdiff(ref_cols, colnames(colData(sim)))
      for (m in missing_cols) {
        ref_val <- colData(sub_real)[[m]]
        if (is.list(ref_val))          colData(sim)[[m]] <- rep(NA_character_, ncol(sim))
        else if (is.double(ref_val))   colData(sim)[[m]] <- rep(NA_real_,      ncol(sim))
        else if (is.integer(ref_val))  colData(sim)[[m]] <- rep(NA_integer_,   ncol(sim))
        else                           colData(sim)[[m]] <- rep(NA_character_, ncol(sim))
      }
      colData(sim) <- colData(sim)[, ref_cols, drop = FALSE]

      # [FIX 15] Stamp ethnicity label on synthetic cells
      colData(sim)[[BIN_COL]] <- rep(bin_label, ncol(sim))
      heartbeat(sprintf(" Set %s='%s' on all %d synthetic cells.\n",
                        BIN_COL, bin_label, ncol(sim)), TRUE)

      if (any(assay(sim, ASSAY_USE) < 0)) warning("Negative counts in synthetic matrix!")

      lib_train <- colSums(assay(sub_train, ASSAY_USE))
      lib_syn   <- colSums(assay(sim, ASSAY_USE))
      heartbeat(sprintf(" Lib size - Training: mean=%.0f sd=%.0f\n", mean(lib_train), sd(lib_train)), TRUE)
      heartbeat(sprintf(" Lib size - Synthetic: mean=%.0f sd=%.0f\n", mean(lib_syn), sd(lib_syn)), TRUE)

      sims[[length(sims) + 1L]] <- sim
      success <- TRUE
      heartbeat(sprintf(" Chunk %d completed: %d cells\n", chunk_id, ncol(sim)), TRUE)

      rm(res, sim, new_counts, new_cov, sub_train, sub_train_target,
         lib_train, lib_syn, ct_counts_train, mu_spec, mu_formula_str, other_covars_v5); gc()
    }

    if (!success) {
      heartbeat(sprintf("scDesign3 FAILED for '%s' after %d attempts; naive resampling.\n",
                        bin_label, MAX_RETRIES), TRUE)
      set.seed(123)
      n_real_cells <- ncol(sub_real)
      resample_idx <- sample.int(n_real_cells, remaining, replace = TRUE)
      real_mat     <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
      res_counts   <- real_mat[, resample_idx, drop = FALSE]
      colnames(res_counts) <- paste0("synthetic_resample_", bin_label, "_",
                                     seq_len(ncol(res_counts)))
      res_cov <- as.data.frame(colData(sub_real))[resample_idx, , drop = FALSE]
      rownames(res_cov) <- colnames(res_counts)
      sim_fallback <- SingleCellExperiment(assays  = list(counts = res_counts),
                                           colData = DataFrame(res_cov))
      colData(sim_fallback)[[BIN_COL]] <- rep(bin_label, ncol(sim_fallback))
      sims[[length(sims) + 1L]] <- sim_fallback
      remaining <- 0L; break
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
  heartbeat(sprintf("\n=== PROCESSING GROUP: %s ===\n", b), TRUE)
  bin_objects[[b]] <- augment_one_bin(b)
}

# ============================================================
# 11. COMBINE AND RESTORE FULL GENE SET
# ============================================================

heartbeat("\n=== COMBINING ALL GROUPS ===\n", TRUE)
sce_combined <- do.call(cbind, bin_objects)
heartbeat(sprintf("Combined: %d cells x %d genes\n", ncol(sce_combined), nrow(sce_combined)), TRUE)

present_genes  <- rownames(sce_combined)
missing_genes  <- setdiff(original_genes, present_genes)
current_counts <- as(assay(sce_combined, ASSAY_USE), "dgCMatrix")

if (length(missing_genes) > 0) {
  heartbeat(sprintf("Adding %d missing genes...\n", length(missing_genes)), TRUE)
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

truly_missing <- original_genes[is.na(gene_order)]
if (length(truly_missing) > 0) {
  extra_pad <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(truly_missing), ncol(full_counts)),
    dimnames = list(truly_missing, colnames(full_counts)))
  full_counts <- rbind(full_counts, extra_pad)
  full_counts <- full_counts[original_genes, , drop = FALSE]
}

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
  rd_pad <- as.data.frame(matrix(NA, nrow = length(rd_genes_missing), ncol = ncol(rd_subset),
    dimnames = list(rd_genes_missing, colnames(rd_subset))))
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
# 12. SOURCE COLUMN + SANITIZATION
# ============================================================

colData(sce_combined)$source <- ifelse(
  grepl("^synthetic_", colnames(sce_combined)), "synthetic", "real"
)
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
  Source    = colData(sce_combined)$source
))
colnames(summary_df) <- c("Ethnicity", "Source", "Count")
write.csv(summary_df, out_sum, row.names = FALSE)
heartbeat(sprintf("Wrote: %s\n", out_sum), TRUE)

# ============================================================
# 14. FAIRNESS DATASETS
# ============================================================

heartbeat("\n=== GENERATING FAIRNESS DATASETS ===\n", TRUE)

grp_counts <- function(sce_obj) {
  bv <- tolower(trimws(as.character(colData(sce_obj)[[BIN_COL]])))
  table(bv)
}

sample_exact <- function(idx, n, replace = FALSE) {
  n <- as.integer(round(as.numeric(n)))
  if (is.na(n) || n < 0) stop("sample_exact(): invalid n")
  if (n == 0) return(integer(0))
  if (!replace && length(idx) < n) stop("sample_exact(): not enough cells")
  sample(idx, n, replace = replace)
}

# Proportional [FIX 16]
bin_labels_det  <- colData(sce_detect)[[BIN_COL]]
prop_sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_det == b)
  n_b <- prop_targets[b]
  if (n_b <= 0 || length(idx) == 0) integer(0) else sample(idx, n_b)
}))

sce_for_prop <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_for_prop <- ensure_counts_assay(sce_for_prop)
sce_for_prop <- exclude_validation_cells(sce_for_prop)   # [V5 3]
eth_raw_prop <- tolower(trimws(as.character(colData(sce_for_prop)[[BIN_COL]])))
valid_prop   <- !(eth_raw_prop %in% UNKNOWN_VALUES) & !is.na(eth_raw_prop)
sce_for_prop <- sce_for_prop[, valid_prop]
sce_prop     <- sce_for_prop[, prop_sample_idx]

prop_sum <- sum(assay(sce_prop, ASSAY_USE))
if (prop_sum == 0) stop("FATAL [FIX 16]: Proportional count matrix is all zeros.")

out_prop <- paste0(OUTPUT_BASE, "_Proportional_", sum(prop_targets), "_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_prop), out_prop, compression = "gzip")
heartbeat(sprintf("Wrote Proportional: %s\n", out_prop), TRUE)
rm(sce_for_prop); gc()

out_bal_aug <- paste0(OUTPUT_BASE, "_BalancedAugmented_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined), out_bal_aug, compression = "gzip")
heartbeat(sprintf("Wrote BalancedAugmented: %s\n", out_bal_aug), TRUE)

bin_real_vec <- tolower(trimws(as.character(colData(sce_real)[[BIN_COL]])))
up_idx  <- unlist(lapply(all_bins, function(b) sample_exact(which(bin_real_vec == b), TARGET_PER_BIN, replace = TRUE)))
sce_up  <- sce_real[, up_idx]
out_up  <- paste0(OUTPUT_BASE, "_BalancedUpsampled_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_up), out_up, compression = "gzip")
heartbeat(sprintf("Wrote BalancedUpsampled: %s\n", out_up), TRUE)

bin_real_tbl <- table(bin_real_vec)
DOWN_TARGET  <- min(as.numeric(bin_real_tbl))
down_idx <- unlist(lapply(all_bins, function(b) sample_exact(which(bin_real_vec == b), DOWN_TARGET, replace = FALSE)))
sce_down <- sce_real[, down_idx]
out_down <- paste0(OUTPUT_BASE, "_Downsampled_", DOWN_TARGET, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_down), out_down, compression = "gzip")
heartbeat(sprintf("Wrote Downsampled: %s\n", out_down), TRUE)

# ============================================================
# 15. FINAL VALIDATION
# ============================================================

heartbeat("\n=== FINAL VALIDATION ===\n", TRUE)
final_counts <- table(colData(sce_combined)[[BIN_COL]])
for (b in names(final_counts)) {
  heartbeat(sprintf("%s: %d cells (target: %d)\n", b, final_counts[b], TARGET_PER_BIN), TRUE)
}

# ============================================================
# 16. FINAL REPORT
# ============================================================

heartbeat("\n=== ETHNICITY AUGMENTATION V5 (Geneformer workflow) COMPLETE ===\n", TRUE)
heartbeat(sprintf(" BIN_COL              = %s\n", BIN_COL),              TRUE)
heartbeat(sprintf(" VALIDATION_H5AD      = %s\n", VALIDATION_H5AD),     TRUE)
heartbeat(sprintf(" PROPORTIONAL_SIZE    = %d\n", PROPORTIONAL_SIZE),    TRUE)
heartbeat(sprintf(" TARGET_PER_BIN       = %d\n", TARGET_PER_BIN),       TRUE)
heartbeat(sprintf(" MAX_CELLS_PER_BIN    = %d\n", MAX_CELLS_PER_BIN),    TRUE)
heartbeat(sprintf(" ANCHOR_PER_CT_PER_BIN= %d\n", ANCHOR_PER_CT_PER_BIN), TRUE)
heartbeat(sprintf(" FAMILY_USE           = zinb [V5 5]\n"),              TRUE)
heartbeat(sprintf(" N_GROUPS_DETECTED    = %d (%s)\n", length(all_bins),
                  paste(all_bins, collapse = ", ")),                      TRUE)
heartbeat("\nAll done! Check augmentation_ethnicity_pilot_v5_log.txt for details.\n", TRUE)