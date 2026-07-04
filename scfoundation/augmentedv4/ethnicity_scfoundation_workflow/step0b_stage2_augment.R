#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 2: PER-GROUP AUGMENTATION
# Args: <bin_label>
#
# CORE OPTIMIZATION:
# Fits joint_data + marginal_res + copula_res ONCE for the group,
# then loops simu_new() per chunk. Marginal+copula is the expensive
# part (~minutes); simu_new is fast (~seconds per chunk).
#
# Reads:  stage1_shared_state.rds, stage1_sce_<group>.rds
# Writes: stage2_synth_<group>.rds
# ============================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("Usage: stage2.R <bin_label>")
bin_label <- args[[1]]
safe_b <- gsub("[^a-zA-Z0-9_]", "_", bin_label)

cat(sprintf("\n=== STAGE 2: %s ===\n", bin_label))

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDesign3)
  library(Matrix)
  library(BiocParallel)
  library(RhpcBLASctl)
})
blas_set_num_threads(1L)
register(SerialParam(progressbar = FALSE))
N_CORES <- 1L

# ============================================================
# CONFIG (must match stage 1)
# ============================================================

BIN_COL      <- "self_reported_ethnicity"
CELLTYPE_COL <- "cell_type"
SEX_COL      <- "sex"
AGE_BIN_COL  <- "age_bin_10yr"
ASSAY_USE    <- "counts"

CHUNK_SIZE_DEFAULT <- 500L   # bigger chunks since marginals are reused
MAX_CELLS_PER_BIN  <- 500L
MIN_CELLS_INTERACTION <- 50L

FAMILY_USE  <- "nb"
COPULA_TYPE <- "gaussian"
USE_BAM     <- FALSE
EDF_FLEX    <- TRUE

# ============================================================
# LOGGING
# ============================================================

log_path <- sprintf("stage2_log_%s.txt", safe_b)
log_con <- file(log_path, open = "at")
on.exit(close(log_con), add = TRUE)
heartbeat <- function(msg, newline = FALSE) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg, if (newline) "\n" else "")
  cat(line, file = log_con); cat(line, file = stderr()); flush(log_con)
}

heartbeat(sprintf("\n=== STAGE 2 START: %s ===\n", bin_label), TRUE)

# ============================================================
# Load shared state
# ============================================================

heartbeat("Loading shared state...\n", TRUE)
ss <- readRDS("stage1_shared_state.rds")
hvg_names      <- ss$hvg_names
prop_targets   <- ss$prop_targets
TARGET_PER_BIN <- ss$TARGET_PER_BIN
MAJ_BIN        <- ss$MAJ_BIN
all_bins       <- ss$all_bins
anchor_cache   <- ss$anchor_cache

heartbeat(sprintf("HVGs: %d, TARGET_PER_BIN: %d, MAJ: %s\n",
                  length(hvg_names), TARGET_PER_BIN, MAJ_BIN), TRUE)

# ============================================================
# EARLY EXIT: majority group needs no synthesis
# ============================================================

sce_path <- sprintf("stage1_sce_%s.rds", safe_b)
if (!file.exists(sce_path)) stop(sprintf("Group SCE not found: %s", sce_path))

sub_real_full <- readRDS(sce_path)
n_full <- ncol(sub_real_full)
heartbeat(sprintf("Loaded %d real cells\n", n_full), TRUE)

if (bin_label == MAJ_BIN) {
  heartbeat("Majority group - downsample only, no synthesis\n", TRUE)
  if (n_full > TARGET_PER_BIN) {
    set.seed(123)
    sub_real <- sub_real_full[, sample(n_full, TARGET_PER_BIN)]
  } else sub_real <- sub_real_full
  assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")
  saveRDS(sub_real, sprintf("stage2_synth_%s.rds", safe_b))
  heartbeat(sprintf("Saved %d real cells\n", ncol(sub_real)), TRUE)
  heartbeat("\n=== STAGE 2 COMPLETE (majority) ===\n", TRUE)
  quit(status = 0)
}

# ============================================================
# Subsample real cells (proportional share)
# ============================================================

target_real_min <- prop_targets[[bin_label]]
if (is.na(target_real_min) || target_real_min <= 0)
  target_real_min <- min(n_full, TARGET_PER_BIN)

sub_real <- if (n_full > target_real_min) {
  set.seed(123)
  sub_real_full[, sample(n_full, target_real_min)]
} else sub_real_full
assay(sub_real, ASSAY_USE) <- as(assay(sub_real, ASSAY_USE), "dgCMatrix")

n_real <- ncol(sub_real)
need <- TARGET_PER_BIN - n_real

if (need <= 0) {
  heartbeat(sprintf("Real already at target (%d), no synthesis needed\n", n_real), TRUE)
  saveRDS(sub_real, sprintf("stage2_synth_%s.rds", safe_b))
  heartbeat("\n=== STAGE 2 COMPLETE (no synthesis) ===\n", TRUE)
  quit(status = 0)
}

heartbeat(sprintf("Need to synthesize %d cells (have %d real, target %d)\n",
                  need, n_real, TARGET_PER_BIN), TRUE)

# ============================================================
# Build joint training set ONCE
# ============================================================

heartbeat("Building joint training set...\n", TRUE)
sub_train <- readRDS(sce_path)

# Stratified subsample if too large (capped at MAX_CELLS_PER_BIN)
n_train_loaded <- ncol(sub_train)
if (n_train_loaded > MAX_CELLS_PER_BIN) {
  ct_vec <- as.character(colData(sub_train)[[CELLTYPE_COL]])
  tab_ct <- table(ct_vec[ct_vec != ""])
  alloc <- pmax(2L, as.integer(floor(MAX_CELLS_PER_BIN * as.numeric(tab_ct) / sum(tab_ct))))
  names(alloc) <- names(tab_ct)
  alloc <- pmin(alloc, as.integer(tab_ct))
  set.seed(789)
  keep_idx <- unlist(lapply(names(alloc), function(ct) {
    ids <- which(ct_vec == ct); sample(ids, alloc[[ct]], replace = FALSE)
  }))
  sub_train <- sub_train[, keep_idx, drop = FALSE]
  ct_new <- as.character(colData(sub_train)[[CELLTYPE_COL]])
  ct_new[is.na(ct_new)] <- "unknown_cleaned"
  colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_new))
}

ct_final <- as.character(colData(sub_train)[[CELLTYPE_COL]])
ct_final[is.na(ct_final)] <- "unknown_cleaned"
colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_final))
sub_train_target <- sub_train

# Add anchors from other groups
other_anchors <- anchor_cache[setdiff(names(anchor_cache), bin_label)]
target_cts <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
other_anchors <- lapply(other_anchors, function(a) {
  ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
  a[, ct_a %in% target_cts, drop = FALSE]
})
other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]

if (length(other_anchors) > 0) {
  common_genes <- rownames(sub_train_target)
  for (sce_a in other_anchors) common_genes <- intersect(common_genes, rownames(sce_a))
  if (length(common_genes) > 0) {
    pieces <- c(list(sub_train_target[common_genes, ]),
                lapply(other_anchors, function(a) a[common_genes, ]))
    sub_train <- do.call(cbind, pieces)
    if (!"library" %in% colnames(colData(sub_train)) || any(is.na(colData(sub_train)$library)))
      colData(sub_train)$library <- Matrix::colSums(assay(sub_train, ASSAY_USE))
    ct_joint <- as.character(colData(sub_train)[[CELLTYPE_COL]])
    tab_joint <- table(ct_joint)
    keep_ct <- names(tab_joint)[tab_joint >= 2L]
    if (length(keep_ct) < length(tab_joint)) {
      sub_train <- sub_train[, ct_joint %in% keep_ct]
      ct_clean <- as.character(colData(sub_train)[[CELLTYPE_COL]])
      colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
    }
    colData(sub_train)[[BIN_COL]] <- as.character(colData(sub_train)[[BIN_COL]])
    colData(sub_train)[[SEX_COL]] <- as.character(colData(sub_train)[[SEX_COL]])
    colData(sub_train)[[AGE_BIN_COL]] <- as.character(colData(sub_train)[[AGE_BIN_COL]])
  }
}

heartbeat(sprintf("Joint: %d cells, %d CTs, %d groups\n",
                  ncol(sub_train),
                  length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]]))),
                  length(unique(as.character(colData(sub_train)[[BIN_COL]])))), TRUE)

# ============================================================
# Build mu formula (additive)
# ============================================================

build_mu_formula <- function(train_sce) {
  parts <- c("cell_type", "self_reported_ethnicity")
  other_extra <- character(0)
  for (covar in c(SEX_COL, AGE_BIN_COL)) {
    if (!covar %in% colnames(colData(train_sce))) next
    n_levels <- length(unique(na.omit(as.character(colData(train_sce)[[covar]]))))
    if (n_levels < 2L) next
    parts <- c(parts, covar)
    other_extra <- c(other_extra, covar)
  }
  formula_str <- paste(paste(parts, collapse = " + "), "+ offset(log(library))")
  list(formula = formula_str, other_covariates = c(BIN_COL, other_extra, "library"))
}

mu_spec <- build_mu_formula(sub_train)
heartbeat(sprintf("mu_formula: %s\n", mu_spec$formula), TRUE)

# ============================================================
# Fit marginals ONCE (the expensive part)
# ============================================================

heartbeat("Fitting joint_data...\n", TRUE)
t0 <- Sys.time()
joint_data <- construct_data(
  sce = sub_train, assay_use = ASSAY_USE, celltype = CELLTYPE_COL,
  pseudotime = NULL, spatial = NULL,
  other_covariates = mu_spec$other_covariates,
  ncell = ncol(sub_train), corr_by = "cell_type",
  parallelization = "mcmapply", BPPARAM = NULL
)
heartbeat(sprintf("  joint_data done in %.1fs\n", as.numeric(Sys.time()-t0, units="secs")), TRUE)

heartbeat("Fitting marginals (this is the slow step)...\n", TRUE)
t0 <- Sys.time()
marginal_res <- fit_marginal(
  data = joint_data, predictor = "gene",
  mu_formula = mu_spec$formula, sigma_formula = "cell_type",
  family_use = FAMILY_USE, n_cores = N_CORES,
  usebam = USE_BAM, edf_flexible = EDF_FLEX,
  parallelization = "mcmapply", BPPARAM = NULL
)
heartbeat(sprintf("  fit_marginal done in %.1fs\n", as.numeric(Sys.time()-t0, units="secs")), TRUE)

# important_feature: explicit logical vector
mat_for_imp <- assay(sub_train, ASSAY_USE)
zero_frac <- Matrix::rowMeans(mat_for_imp == 0)
sparsity_threshold <- if (ncol(sub_train_target) < MIN_CELLS_INTERACTION) 0.95 else 0.80
imp_feat_vec <- zero_frac <= sparsity_threshold
if (sum(imp_feat_vec) < 2L) imp_feat_vec <- rep(TRUE, length(imp_feat_vec))
names(imp_feat_vec) <- rownames(sub_train)
heartbeat(sprintf("important_feature: %d / %d kept (threshold=%.2f)\n",
                  sum(imp_feat_vec), length(imp_feat_vec), sparsity_threshold), TRUE)

heartbeat("Fitting copula...\n", TRUE)
t0 <- Sys.time()
copula_res <- fit_copula(
  sce = sub_train, assay_use = ASSAY_USE,
  marginal_list = marginal_res, family_use = FAMILY_USE,
  copula = COPULA_TYPE, DT = TRUE, pseudo_obs = FALSE,
  n_cores = N_CORES, input_data = joint_data$dat,
  important_feature = imp_feat_vec, if_sparse = FALSE,
  parallelization = "mcmapply", BPPARAM = NULL
)
heartbeat(sprintf("  fit_copula done in %.1fs\n", as.numeric(Sys.time()-t0, units="secs")), TRUE)

target_dat_rows <- which(as.character(joint_data$dat[[BIN_COL]]) == bin_label)
heartbeat(sprintf("Target-group dat rows: %d\n", length(target_dat_rows)), TRUE)

# ============================================================
# Loop simu_new for chunks (the FAST part now)
# ============================================================

validate_new_counts <- function(new_count, ref_genes, chunk_id) {
  if (is.null(new_count)) stop("simu_new returned NULL")
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(new_count, nrow = length(new_count), ncol = 1,
                        dimnames = list(names(new_count), NULL))
  }
  if (!is.null(rownames(new_count)) && !is.null(ref_genes)) {
    if (ncol(new_count) == length(ref_genes) && nrow(new_count) != length(ref_genes))
      new_count <- t(new_count)
  }
  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(0, nrow = length(missing), ncol = ncol(new_count),
                  dimnames = list(missing, colnames(new_count)))
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop = FALSE]
  colnames(new_count) <- paste0("synthetic_", bin_label, "_chunk", chunk_id, "_", seq_len(ncol(new_count)))
  new_count
}

heartbeat(sprintf("Looping simu_new for %d total cells in chunks of %d...\n",
                  need, CHUNK_SIZE_DEFAULT), TRUE)

sims <- list()
remaining <- need
chunk_id <- 1L

while (remaining > 0) {
  chunk <- min(CHUNK_SIZE_DEFAULT, remaining)
  n_ct_train <- length(unique(as.character(colData(sub_train)[[CELLTYPE_COL]])))
  if (chunk < 2L * n_ct_train) chunk <- 2L * n_ct_train

  t0 <- Sys.time()
  new_cov_idx <- sample(target_dat_rows, chunk, replace = (length(target_dat_rows) < chunk))
  new_cov_df <- joint_data$dat[new_cov_idx, , drop = FALSE]
  if (!"library" %in% colnames(new_cov_df))
    new_cov_df$library <- median(colData(sub_train)$library)
  if (!SEX_COL %in% colnames(new_cov_df))
    new_cov_df[[SEX_COL]] <- names(sort(table(as.character(colData(sub_train_target)[[SEX_COL]])), decreasing = TRUE))[1]
  if (!AGE_BIN_COL %in% colnames(new_cov_df))
    new_cov_df[[AGE_BIN_COL]] <- names(sort(table(as.character(colData(sub_train_target)[[AGE_BIN_COL]])), decreasing = TRUE))[1]

  cov_cts <- as.character(new_cov_df[[CELLTYPE_COL]])
  ct_tab_cov <- table(cov_cts)
  thin_cts <- names(ct_tab_cov)[ct_tab_cov < 2L]
  if (length(thin_cts) > 0) {
    extra_rows <- lapply(thin_cts, function(ct) {
      ct_rows <- target_dat_rows[as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct]
      if (length(ct_rows) == 0L) return(NULL)
      joint_data$dat[sample(ct_rows, 2L - ct_tab_cov[[ct]], replace = TRUE), ]
    })
    extra_rows <- do.call(rbind, Filter(Negate(is.null), extra_rows))
    if (!is.null(extra_rows) && nrow(extra_rows) > 0)
      new_cov_df <- rbind(new_cov_df, extra_rows)
  }
  new_cov_df$corr_group <- as.character(new_cov_df[[CELLTYPE_COL]])
  rownames(new_cov_df) <- paste0("Cell", seq_len(nrow(new_cov_df)))

  para_list <- extract_para(
    sce = sub_train, assay_use = ASSAY_USE,
    marginal_list = marginal_res, n_cores = N_CORES,
    family_use = FAMILY_USE, new_covariate = new_cov_df,
    parallelization = "mcmapply", BPPARAM = NULL,
    data = joint_data$dat
  )

  new_count <- simu_new(
    sce = sub_train, assay_use = ASSAY_USE,
    mean_mat = para_list$mean_mat, sigma_mat = para_list$sigma_mat,
    zero_mat = para_list$zero_mat, quantile_mat = NULL,
    copula_list = copula_res$copula_list, n_cores = N_CORES,
    family_use = FAMILY_USE, nonnegative = TRUE, nonzerovar = TRUE,
    input_data = joint_data$dat, new_covariate = new_cov_df,
    important_feature = imp_feat_vec, parallelization = "mcmapply",
    BPPARAM = NULL, filtered_gene = joint_data$filtered_gene
  )

  new_counts <- validate_new_counts(new_count, rownames(sub_real), chunk_id)
  rownames(new_cov_df) <- colnames(new_counts)
  sim <- SingleCellExperiment(assays = list(counts = new_counts), colData = new_cov_df)

  syn_active <- Matrix::colSums(as(assay(sim, ASSAY_USE), "dgCMatrix") > 0)
  syn_lib <- Matrix::colSums(as(assay(sim, ASSAY_USE), "dgCMatrix"))
  real_active <- Matrix::colSums(as(assay(sub_real, ASSAY_USE), "dgCMatrix") > 0)
  real_lib <- Matrix::colSums(as(assay(sub_real, ASSAY_USE), "dgCMatrix"))
  heartbeat(sprintf(" QC HVG-space: synth active=%.0f (real %.0f) lib=%.0f (real %.0f)\n",
                    median(syn_active), median(real_active),
                    median(syn_lib), median(real_lib)), TRUE)

  ref_cols <- colnames(colData(sub_real))
  missing_cols <- setdiff(ref_cols, colnames(colData(sim)))
  for (m in missing_cols) {
    rv <- colData(sub_real)[[m]]
    colData(sim)[[m]] <- if (is.list(rv)) rep(NA_character_, ncol(sim))
                         else if (is.double(rv)) rep(NA_real_, ncol(sim))
                         else if (is.integer(rv)) rep(NA_integer_, ncol(sim))
                         else rep(NA_character_, ncol(sim))
  }
  colData(sim) <- colData(sim)[, ref_cols, drop = FALSE]
  colData(sim)[[BIN_COL]] <- rep(bin_label, ncol(sim))

  sims[[length(sims) + 1L]] <- sim
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")
  heartbeat(sprintf("Chunk %d done: %d cells in %.1fs\n", chunk_id, ncol(sim), elapsed), TRUE)

  remaining <- remaining - chunk
  chunk_id <- chunk_id + 1L
}

# ============================================================
# Combine real + synthetic and save
# ============================================================

out_bin <- do.call(cbind, c(list(sub_real), sims))
heartbeat(sprintf("Final: %d cells (%d real + %d synth)\n",
                  ncol(out_bin), ncol(sub_real),
                  sum(sapply(sims, ncol))), TRUE)
saveRDS(out_bin, sprintf("stage2_synth_%s.rds", safe_b))

heartbeat(sprintf("\n=== STAGE 2 COMPLETE: %s ===\n", bin_label), TRUE)
