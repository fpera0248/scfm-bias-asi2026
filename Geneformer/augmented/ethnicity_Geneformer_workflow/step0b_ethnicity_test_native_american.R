#!/usr/bin/env Rscript
# ============================================================
# STEP 0B TEST: NATIVE AMERICAN ONLY — LIBRARY SIZE OFFSET FIX
# ============================================================
# Tests the library size offset fix on the smallest group only.
# Runs augmentation for native american, then the distribution
# shift analysis confirms whether library sizes are preserved.
#
# Changes vs full step0b:
#   [TEST 1] Only processes native american group
#   [TEST 2] library size added to colData in load_bin_fresh()
#   [TEST 3] library recomputed in build_joint_train() after cbind
#   [TEST 4] other_covariates includes "library" in construct_data
#   [TEST 5] mu_formula includes offset(log(library)) in fit_marginal
#   [TEST 6] guard for missing library in new_cov_df
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
})

N_CORES <- 1L
blas_set_num_threads(N_CORES)
register(SerialParam(progressbar = FALSE))

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_H5AD   <- "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"
OUTPUT_BASE  <- "ILD_Ethnicity_Pilot_TEST"
CKPTDIR      <- "checkpoints_test_native_american"
LOGFILE      <- "test_native_american_log.txt"

BIN_COL      <- "self_reported_ethnicity"
CELLTYPE_COL <- "cell_type"

UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                    "multiethnic", "na na", "not applicable", "prefer not to say")
CT_UNKNOWN     <- c("unknown", "na", "n/a", "not reported", "", "nan")

PROPORTIONAL_SIZE  <- 2500L
CHUNK_SIZE_DEFAULT <- 200L
MAX_RETRIES        <- 3L

N_HVG            <- 1000L
MIN_CELLS        <- 5L
MIN_COUNTS       <- 10L
HVG_SAMPLE_SIZE  <- 2000L

MIN_CT_CELLS          <- 20L
MAX_CELLS_PER_BIN     <- 500L
MIN_CELLS_INTERACTION <- 50L
ANCHOR_PER_CT_PER_BIN <- 2L

ASSAY_USE <- "counts"

# [TEST 1] Force only native american — majority is european american
TEST_GROUP  <- "native american"
FORCE_MAJ   <- "european american"

# ============================================================
# LOGGING
# ============================================================

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

heartbeat("\n=== TEST: NATIVE AMERICAN LIBRARY SIZE OFFSET FIX ===\n", TRUE)

# ============================================================
# HELPERS
# ============================================================

assert_raw_counts <- function(sce_obj, assay_name = "counts") {
  if (!assay_name %in% assayNames(sce_obj))
    stop(sprintf("Assay '%s' not found.", assay_name))
  mat <- assay(sce_obj, assay_name)
  vals_all <- if (inherits(mat, "dgCMatrix")) mat@x else {
    nr <- nrow(mat); nc <- ncol(mat)
    if (nr == 0 || nc == 0) return(invisible(TRUE))
    samp_n <- min(10000L, nr * nc)
    mat[cbind(sample.int(nr, samp_n, replace=TRUE),
              sample.int(nc, samp_n, replace=TRUE))]
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
    assert_raw_counts(sce_obj, "counts")
  } else {
    stop("Neither 'counts' nor 'X' assay present.")
  }
  assays(sce_obj) <- SimpleList(counts = assay(sce_obj, "counts"))
  mat <- assay(sce_obj, "counts")
  if (!inherits(mat, "dgCMatrix"))
    assay(sce_obj, "counts") <- as(mat, "dgCMatrix")
  sce_obj
}

sanitize_coldata_for_h5ad <- function(sce_obj) {
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
  n      <- ncol(sce_obj)
  cd_raw <- colData(sce_obj)
  col_list <- lapply(colnames(cd_raw), function(col) {
    v <- tryCatch({
      tmp <- cd_raw[[col]]
      vapply(seq_len(n), function(i) {
        xi <- tryCatch(tmp[[i]], error = function(e) NA)
        if (is.null(xi) || length(xi) == 0 || (length(xi)==1 && is.na(xi)))
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
  cd_new <- structure(col_list_clean, class="data.frame",
                      row.names=rn_safe, names=names(col_list_clean))
  colnames(cnt) <- rn_safe
  SingleCellExperiment(assays=list(counts=cnt), colData=DataFrame(cd_new))
}

# ============================================================
# AUTO-DETECT GROUP STRUCTURE
# ============================================================

heartbeat("\n=== AUTO-DETECTING GROUP STRUCTURE ===\n", TRUE)

sce_detect  <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_detect  <- ensure_counts_assay(sce_detect)
eth_raw     <- tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]])))
valid       <- !(eth_raw %in% UNKNOWN_VALUES) & !is.na(eth_raw)
sce_detect  <- sce_detect[, valid]
colData(sce_detect)[[BIN_COL]] <- as.character(
  tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]]))))
bin_table   <- table(colData(sce_detect)[[BIN_COL]])
all_bins    <- sort(names(bin_table))

for (b in all_bins)
  heartbeat(sprintf("  %s: %d cells", b, bin_table[b]), TRUE)

MAJ_BIN <- FORCE_MAJ
MIN_BIN <- TEST_GROUP
heartbeat(sprintf(">>> MAJORITY: %s  MINORITY (test): %s\n", MAJ_BIN, MIN_BIN), TRUE)

# ============================================================
# PROPORTIONAL TARGETS
# ============================================================

total_real   <- sum(as.numeric(bin_table))
prop_targets <- sapply(as.numeric(bin_table),
                       function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins

TARGET_PER_BIN <- prop_targets[MAJ_BIN]
TARGET_DOWN    <- prop_targets[MIN_BIN]
if (TARGET_PER_BIN < TARGET_DOWN) {
  tmp <- TARGET_PER_BIN; TARGET_PER_BIN <- TARGET_DOWN; TARGET_DOWN <- tmp
}
heartbeat(sprintf("TARGET_PER_BIN = %d\n", TARGET_PER_BIN), TRUE)

if (!dir.exists(CKPTDIR)) dir.create(CKPTDIR, recursive = TRUE)

# ============================================================
# HVG SELECTION
# ============================================================

heartbeat("\n=== HVG SELECTION ===\n", TRUE)

sce_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = TRUE)
sce_full <- ensure_counts_assay(sce_full)
eth_full <- tolower(trimws(as.character(colData(sce_full)[[BIN_COL]])))
sce_full <- sce_full[, !(eth_full %in% UNKNOWN_VALUES) & !is.na(eth_full)]
colData(sce_full)[[BIN_COL]] <- as.character(
  tolower(trimws(as.character(colData(sce_full)[[BIN_COL]]))))
original_genes <- rownames(sce_full)

bin_labels_full  <- colData(sce_full)[[BIN_COL]]
bin_counts_full  <- table(bin_labels_full)
target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
n_per_bin_hvg    <- pmin(as.numeric(bin_counts_full), target_per_bin_hvg)
names(n_per_bin_hvg) <- names(bin_counts_full)

sample_idx_hvg <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_full == b)
  if (length(idx) > n_per_bin_hvg[b]) sample(idx, n_per_bin_hvg[b]) else idx
}))

sce_sample <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
sce_sample <- ensure_counts_assay(sce_sample)
eth_sample <- tolower(trimws(as.character(colData(sce_sample)[[BIN_COL]])))
sce_sample <- sce_sample[, !(eth_sample %in% UNKNOWN_VALUES) & !is.na(eth_sample)]
sce_sample <- sce_sample[, sample_idx_hvg]
colData(sce_sample)[[BIN_COL]] <- as.character(
  tolower(trimws(as.character(colData(sce_sample)[[BIN_COL]]))))

count_mat    <- assay(sce_sample, ASSAY_USE)
gene_sums    <- Matrix::rowSums(count_mat)
gene_nonzero <- Matrix::rowSums(count_mat > 0)
keep_genes   <- (gene_sums >= MIN_COUNTS) & (gene_nonzero >= MIN_CELLS)
sce_filt     <- sce_sample[keep_genes, ]
filt_mat     <- assay(sce_filt, ASSAY_USE)

gene_means    <- Matrix::rowMeans(filt_mat)
filt_mat_sq   <- filt_mat; filt_mat_sq@x <- filt_mat_sq@x^2
gene_vars     <- pmax(Matrix::rowMeans(filt_mat_sq) - gene_means^2, 0)
cv2           <- gene_vars / (gene_means^2 + 1e-8)
cv2[is.na(cv2) | is.infinite(cv2)] <- 0

n_select  <- min(N_HVG, length(cv2))
hvg_names <- rownames(sce_filt)[order(cv2, decreasing=TRUE)[seq_len(n_select)]]
heartbeat(sprintf("Selected %d HVGs\n", length(hvg_names)), TRUE)
rm(sce_sample, sce_filt, count_mat, filt_mat, filt_mat_sq); gc()

# ============================================================
# [TEST 2] LOAD BIN FRESH — with library size
# ============================================================

load_bin_fresh <- function(bin_label) {
  heartbeat(sprintf("Loading group '%s' (HVG subset)...", bin_label), TRUE)

  sce_raw <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
  common_early <- intersect(hvg_names, rownames(sce_raw))
  if (length(common_early) == 0)
    stop(sprintf("No HVGs found for %s", bin_label))
  sce_raw <- sce_raw[common_early, ]
  sce_raw <- ensure_counts_assay(sce_raw)

  eth_raw_vec <- tolower(trimws(as.character(colData(sce_raw)[[BIN_COL]])))
  valid_eth   <- !(eth_raw_vec %in% UNKNOWN_VALUES) & !is.na(eth_raw_vec)
  sce_raw     <- sce_raw[, valid_eth]

  ct_raw   <- as.character(colData(sce_raw)[[CELLTYPE_COL]])
  valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
  sce_raw  <- sce_raw[, valid_ct]

  eth_vec  <- tolower(trimws(as.character(colData(sce_raw)[[BIN_COL]])))
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

  # [TEST 2] Add library size from FULL GENE SET (not HVG subset)
  # Summing 1000 HVGs gives ~7; must sum all 31,432 genes to get ~6500
  sce_raw_full <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
  sce_raw_full <- ensure_counts_assay(sce_raw_full)
  eth_raw_full <- tolower(trimws(as.character(colData(sce_raw_full)[[BIN_COL]])))
  sce_raw_full <- sce_raw_full[, !(eth_raw_full %in% UNKNOWN_VALUES) & !is.na(eth_raw_full)]
  eth_raw_full <- tolower(trimws(as.character(colData(sce_raw_full)[[BIN_COL]])))
  sce_raw_full <- sce_raw_full[, eth_raw_full == bin_label]
  lib_full     <- Matrix::colSums(assay(sce_raw_full, ASSAY_USE))
  common_cells <- intersect(colnames(sce_bin), names(lib_full))
  colData(sce_bin)$library <- as.numeric(lib_full[common_cells])
  rm(sce_raw_full); gc()
  heartbeat(sprintf("  -> %d cells, %d CTs, lib size (full genes) mean=%.0f median=%.0f",
                    ncol(sce_bin), length(unique(ct_vec)),
                    mean(colData(sce_bin)$library),
                    median(colData(sce_bin)$library)), TRUE)
  sce_bin
}

# ============================================================
# ANCHOR CACHE
# ============================================================

heartbeat("\n=== PRE-CACHING ANCHOR CELLS ===\n", TRUE)
anchor_cache <- list()
for (b in all_bins) {
  tryCatch({
    sce_b  <- load_bin_fresh(b)
    ct_b   <- as.character(colData(sce_b)[[CELLTYPE_COL]])
    tab_b  <- table(ct_b)
    anch_idx <- unlist(lapply(names(tab_b), function(ct) {
      ids <- which(ct_b == ct)
      sample(ids, min(ANCHOR_PER_CT_PER_BIN, length(ids)))
    }))
    anchor_cache[[b]] <- sce_b[, anch_idx]
    heartbeat(sprintf(" Cached %d anchor cells for '%s'\n", length(anch_idx), b), TRUE)
    rm(sce_b); gc()
  }, error = function(e) {
    heartbeat(sprintf(" WARNING: anchor cache FAILED for '%s': %s\n",
                      b, conditionMessage(e)), TRUE)
  })
}

# ============================================================
# [TEST 3] BUILD JOINT TRAIN — recompute library after cbind
# ============================================================

build_joint_train <- function(sub_train_target, target_bin) {
  other_anchors <- anchor_cache[setdiff(names(anchor_cache), target_bin)]
  if (length(other_anchors) == 0) return(sub_train_target)

  target_cts    <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
  other_anchors <- lapply(other_anchors, function(a) {
    ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
    a[, ct_a %in% target_cts, drop = FALSE]
  })
  other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]
  if (length(other_anchors) == 0) return(sub_train_target)

  common_genes <- rownames(sub_train_target)
  for (sce_a in other_anchors)
    common_genes <- intersect(common_genes, rownames(sce_a))
  if (length(common_genes) == 0) return(sub_train_target)

  pieces <- c(
    list(sub_train_target[common_genes, ]),
    lapply(other_anchors, function(a) a[common_genes, ])
  )
  joint <- do.call(cbind, pieces)

  # [TEST 3] Recompute library size on joint set (anchors have their own library sizes)
  colData(joint)$library <- Matrix::colSums(assay(joint, ASSAY_USE))
  heartbeat(sprintf(" Joint lib size: mean=%.0f median=%.0f\n",
                    mean(colData(joint)$library),
                    median(colData(joint)$library)), TRUE)

  # FIX 19: drop singleton CTs
  ct_joint  <- as.character(colData(joint)[[CELLTYPE_COL]])
  tab_joint <- table(ct_joint)
  keep_ct   <- names(tab_joint)[tab_joint >= 2L]
  if (length(keep_ct) < length(tab_joint)) {
    joint   <- joint[, ct_joint %in% keep_ct]
    ct_clean <- as.character(colData(joint)[[CELLTYPE_COL]])
    colData(joint)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
  }

  colData(joint)[[BIN_COL]] <- as.character(colData(joint)[[BIN_COL]])

  n_target <- ncol(sub_train_target)
  n_anchor <- ncol(joint) - n_target
  heartbeat(sprintf(" Joint: %d target + %d anchor = %d total, %d groups, %d CTs\n",
                    n_target, n_anchor, ncol(joint),
                    length(unique(as.character(colData(joint)[[BIN_COL]]))),
                    length(unique(as.character(colData(joint)[[CELLTYPE_COL]])))), TRUE)
  joint
}

# ============================================================
# VALIDATE NEW COUNTS
# ============================================================

validate_new_counts <- function(new_count, ref_genes, chunk_id, bin_label) {
  if (is.null(new_count)) stop("scDesign3 returned NULL")
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(new_count, nrow=length(new_count), ncol=1,
                        dimnames=list(names(new_count), NULL))
  }
  if (!is.null(rownames(new_count)) && !is.null(ref_genes)) {
    if (ncol(new_count) == length(ref_genes) && nrow(new_count) != length(ref_genes))
      new_count <- t(new_count)
  }
  if (ncol(new_count) == 0) stop("Empty count matrix")
  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(0, nrow=length(missing), ncol=ncol(new_count),
                  dimnames=list(missing, colnames(new_count)))
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop=FALSE]
  colnames(new_count) <- paste0("synthetic_", bin_label, "_chunk", chunk_id,
                                "_", seq_len(ncol(new_count)))
  new_count
}

# ============================================================
# AUGMENT ONE BIN
# ============================================================

augment_one_bin <- function(bin_label) {
  ckpt_file <- file.path(CKPTDIR,
    paste0("ethnicity_", gsub("[^a-zA-Z0-9_]", "_", bin_label), ".rds"))

  if (file.exists(ckpt_file)) {
    heartbeat(sprintf("Checkpoint found for '%s' — loading.\n", bin_label), TRUE)
    return(sanitize_coldata_for_h5ad(readRDS(ckpt_file)))
  }

  sub_real_full <- load_bin_fresh(bin_label)
  n_full        <- ncol(sub_real_full)

  if (bin_label == MAJ_BIN) {
    if (n_full > TARGET_PER_BIN) {
      sub_real <- sub_real_full[, sample(n_full, TARGET_PER_BIN)]
    } else {
      sub_real <- sub_real_full
    }
    assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
    saveRDS(sub_real, ckpt_file)
    return(sub_real)
  }

  target_real_min <- prop_targets[bin_label]
  if (is.na(target_real_min) || target_real_min <= 0)
    target_real_min <- min(n_full, TARGET_PER_BIN)

  sub_real <- if (n_full > target_real_min) {
    sub_real_full[, sample(n_full, target_real_min)]
  } else {
    sub_real_full
  }
  assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")

  n_real <- ncol(sub_real)
  need   <- TARGET_PER_BIN - n_real
  if (need <= 0) {
    saveRDS(sub_real, ckpt_file); return(sub_real)
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
      heartbeat(sprintf("Chunk %d | attempt %d | n=%d | remaining=%d\n",
                        chunk_id, attempt, chunk, remaining), TRUE)

      sub_train      <- load_bin_fresh(bin_label)
      n_train_loaded <- ncol(sub_train)

      n_ct_train <- length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
      if (chunk < 2L * n_ct_train) chunk <- 2L * n_ct_train

      # Stratified cap
      if (n_train_loaded > MAX_CELLS_PER_BIN) {
        ct_vec <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        tab_ct <- table(ct_vec[ct_vec != ""])
        alloc  <- pmax(2L, as.integer(floor(MAX_CELLS_PER_BIN *
                        as.numeric(tab_ct) / sum(tab_ct))))
        names(alloc) <- names(tab_ct)
        alloc <- pmin(alloc, as.integer(tab_ct))
        keep_idx  <- unlist(lapply(names(alloc), function(ct) {
          ids <- which(ct_vec == ct)
          sample(ids, alloc[[ct]], replace=FALSE)
        }))
        sub_train <- sub_train[, keep_idx, drop=FALSE]
        ct_new    <- as.character(colData(sub_train)[[CELLTYPE_COL]])
        ct_new[is.na(ct_new)] <- "unknown_cleaned"
        colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_new))
      }

      ct_final <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      ct_final[is.na(ct_final)] <- "unknown_cleaned"
      colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_final))

      if (any(table(as.character(colData(sub_train)[[CELLTYPE_COL]])) < 2L))
        stop("Training set has CT with <2 cells")

      # Build joint training set
      sub_train_target <- sub_train
      sub_train        <- build_joint_train(sub_train_target, bin_label)

      # FIX 18: adaptive sparsity
      n_target_cells     <- ncol(sub_train_target)
      mat_for_imp        <- assay(sub_train, ASSAY_USE)
      zero_frac          <- Matrix::rowMeans(mat_for_imp == 0)
      sparsity_threshold <- if (n_target_cells < MIN_CELLS_INTERACTION) 0.95 else 0.80
      imp_feat_vec       <- zero_frac <= sparsity_threshold
      if (sum(imp_feat_vec) < 2L) imp_feat_vec <- rep(TRUE, length(imp_feat_vec))

      eth_vals    <- unique(as.character(colData(sub_train)[[BIN_COL]]))
      if (length(eth_vals) < 2L)
        stop("No ethnicity variance in joint set")

      # CT × ethnicity gate
      ct_eth_tab <- table(
        CT  = as.character(colData(sub_train)[[CELLTYPE_COL]]),
        Grp = as.character(colData(sub_train)[[BIN_COL]])
      )
      ct_n_grps <- rowSums(ct_eth_tab > 0)
      single_ct <- names(ct_n_grps)[ct_n_grps < 2L]
      if (length(single_ct) > 0)
        stop(sprintf("CT(s) in only 1 ethnicity group: %s",
                     paste(single_ct, collapse=", ")))

      # --------------------------------------------------------
      # [TEST 4+5] LOW-LEVEL API with library size offset
      # --------------------------------------------------------
      run_sd3_lowlevel <- function(mu_f, corr_f) {
        heartbeat(sprintf(" [low-level] mu=%s corr=%s\n", mu_f, corr_f), TRUE)
        tryCatch({

          # [TEST 4] include "library" in other_covariates
          joint_data <- construct_data(
            sce              = sub_train,
            assay_use        = ASSAY_USE,
            celltype         = CELLTYPE_COL,
            pseudotime       = NULL,
            spatial          = NULL,
            other_covariates = c(BIN_COL, "library"),
            ncell            = ncol(sub_train),
            corr_by          = corr_f,
            parallelization  = "mcmapply",
            BPPARAM          = NULL
          )

          # [TEST 5] offset(log(library)) in mu_formula
          marginal_res <- fit_marginal(
            data            = joint_data,
            predictor       = "gene",
            mu_formula      = mu_f,
            sigma_formula   = "cell_type",
            family_use      = "nb",
            n_cores         = N_CORES,
            usebam          = FALSE,
            parallelization = "mcmapply",
            BPPARAM         = NULL
          )

          copula_res <- fit_copula(
            sce               = sub_train,
            assay_use         = ASSAY_USE,
            marginal_list     = marginal_res,
            family_use        = "nb",
            copula            = "gaussian",
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
          new_cov_df  <- joint_data$dat[new_cov_idx, , drop=FALSE]

          # [TEST 6] Guard for missing library
          if (!"library" %in% colnames(new_cov_df)) {
            new_cov_df$library <- median(colData(sub_train)$library)
            heartbeat(" WARNING: library missing from new_cov_df; using median.\n", TRUE)
          }

          # FIX 22: ensure >=2 rows per CT
          cov_cts    <- as.character(new_cov_df[[CELLTYPE_COL]])
          ct_tab_cov <- table(cov_cts)
          thin_cts   <- names(ct_tab_cov)[ct_tab_cov < 2L]
          if (length(thin_cts) > 0) {
            extra_rows <- lapply(thin_cts, function(ct) {
              ct_rows <- target_dat_rows[
                as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct]
              if (length(ct_rows) == 0L) return(NULL)
              joint_data$dat[sample(ct_rows, 2L - ct_tab_cov[[ct]], replace=TRUE), ]
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
            family_use      = "nb",
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
            family_use        = "nb",
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

      heartbeat(" [Tier 1] mu=cell_type+ethnicity+interaction+offset(library)\n", TRUE)
      res <- run_sd3_lowlevel(
        "cell_type + self_reported_ethnicity + cell_type:self_reported_ethnicity + offset(log(library))",
        "cell_type"
      )

      if (is.null(res))
        stop(sprintf("Tier 1 FAILED for '%s' chunk %d attempt %d",
                     bin_label, chunk_id, attempt))

      new_counts <- validate_new_counts(res$new_count, rownames(sub_real),
                                        chunk_id, bin_label)
      new_cov    <- res$new_covariate
      if (is.null(new_cov)) new_cov <- DataFrame(row.names=colnames(new_counts))
      else rownames(new_cov) <- colnames(new_counts)

      sim <- SingleCellExperiment(assays=list(counts=new_counts), colData=new_cov)

      # Align metadata
      ref_cols     <- colnames(colData(sub_real))
      missing_cols <- setdiff(ref_cols, colnames(colData(sim)))
      for (m in missing_cols) {
        ref_val <- colData(sub_real)[[m]]
        colData(sim)[[m]] <- if (is.double(ref_val)) rep(NA_real_, ncol(sim))
                             else if (is.integer(ref_val)) rep(NA_integer_, ncol(sim))
                             else rep(NA_character_, ncol(sim))
      }
      colData(sim) <- colData(sim)[, ref_cols, drop=FALSE]
      colData(sim)[[BIN_COL]] <- rep(bin_label, ncol(sim))

      # Library size check
      lib_real <- colSums(assay(sub_real, ASSAY_USE))
      lib_syn  <- colSums(assay(sim, ASSAY_USE))
      heartbeat(sprintf(" Real lib:  mean=%.0f median=%.0f\n",
                        mean(lib_real), median(lib_real)), TRUE)
      heartbeat(sprintf(" Synth lib: mean=%.0f median=%.0f\n",
                        mean(lib_syn), median(lib_syn)), TRUE)

      sims[[length(sims) + 1L]] <- sim
      success <- TRUE
      heartbeat(sprintf(" Chunk %d done: %d cells generated\n", chunk_id, ncol(sim)), TRUE)

      rm(res, sim, new_counts, new_cov, sub_train, sub_train_target); gc()
    }

    if (!success) {
      heartbeat(sprintf("FAILED after %d attempts — naive resampling fallback\n",
                        MAX_RETRIES), TRUE)
      resample_idx <- sample.int(ncol(sub_real), remaining, replace=TRUE)
      res_counts   <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")[, resample_idx]
      colnames(res_counts) <- paste0("synthetic_resample_", bin_label,
                                     "_", seq_len(ncol(res_counts)))
      res_cov <- as.data.frame(colData(sub_real))[resample_idx, ]
      rownames(res_cov) <- colnames(res_counts)
      sim_fallback <- SingleCellExperiment(
        assays  = list(counts = res_counts),
        colData = DataFrame(res_cov)
      )
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
# [TEST 1] PROCESS ONLY NATIVE AMERICAN + MAJORITY
# ============================================================

heartbeat("\n=== PROCESSING TEST GROUPS (native american + european american) ===\n", TRUE)

test_bins   <- c(MAJ_BIN, TEST_GROUP)
bin_objects <- list()

for (b in test_bins) {
  heartbeat(sprintf("\n=== PROCESSING: %s ===\n", b), TRUE)
  bin_objects[[b]] <- augment_one_bin(b)
}

# ============================================================
# COMBINE AND WRITE OUTPUT
# ============================================================

heartbeat("\n=== COMBINING AND WRITING OUTPUTS ===\n", TRUE)

sce_combined <- do.call(cbind, bin_objects)
heartbeat(sprintf("Combined: %d cells x %d genes\n",
                  ncol(sce_combined), nrow(sce_combined)), TRUE)

# Add source flag
colData(sce_combined)$source <- ifelse(
  grepl("^synthetic_", colnames(sce_combined)), "synthetic", "real")

sce_combined <- sanitize_coldata_for_h5ad(sce_combined)

out_file <- paste0(OUTPUT_BASE, "_NativeAmerican_LibraryOffset_test.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined, drop_all_na=FALSE),
                         out_file, compression="gzip")
heartbeat(sprintf("Wrote: %s\n", out_file), TRUE)

# Summary
tab <- table(colData(sce_combined)[[BIN_COL]],
             colData(sce_combined)$source)
heartbeat("Cell counts by group and source:\n", TRUE)
heartbeat(capture.output(print(tab)), TRUE)

# Library size summary
real_cells <- sce_combined[, colData(sce_combined)$source == "real"]
syn_cells  <- sce_combined[, colData(sce_combined)$source == "synthetic"]
lib_real   <- Matrix::colSums(assay(real_cells, ASSAY_USE))
lib_syn    <- Matrix::colSums(assay(syn_cells,  ASSAY_USE))
heartbeat(sprintf("\nLibrary size summary:\n"), TRUE)
heartbeat(sprintf("  Real      — mean=%.0f  median=%.0f  sd=%.0f\n",
                  mean(lib_real), median(lib_real), sd(lib_real)), TRUE)
heartbeat(sprintf("  Synthetic — mean=%.0f  median=%.0f  sd=%.0f\n",
                  mean(lib_syn), median(lib_syn), sd(lib_syn)), TRUE)

heartbeat("\n=== TEST COMPLETE ===\n", TRUE)
heartbeat(sprintf("Check %s for full log.\n", LOGFILE), TRUE)
heartbeat("Next: run analyze_distribution_shift.py and check KS p-values.\n", TRUE)