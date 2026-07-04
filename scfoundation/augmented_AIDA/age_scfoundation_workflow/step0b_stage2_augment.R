#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 2: PER-BIN scDesign3 AUGMENTATION
# Age workflow (v6 parallel architecture)
# ============================================================
# Reads BIN_LABEL_ENCODED env var; one job per bin.
# Outputs per-bin synthetic SCE to ./stage2_outputs/{bin}.rds
# ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDesign3)
  library(Matrix)
  library(BiocParallel)
  library(RhpcBLASctl)
})

N_CORES <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))
blas_set_num_threads(N_CORES)
register(SerialParam(progressbar = FALSE))

BIN_LABEL <- Sys.getenv("BIN_LABEL_ENCODED", "")
if (BIN_LABEL == "") stop("BIN_LABEL_ENCODED env var not set")

BIN_COL      <- "age_bin_10yr"
CELLTYPE_COL <- "cell_type"
ASSAY_USE    <- "counts"
MAX_CELLS_PER_BIN <- 500L
MIN_CELLS_INTERACTION <- 50L
CHUNK_SIZE_DEFAULT <- 200L
MAX_RETRIES <- 3L

IN_DIR  <- "stage1_outputs"
OUT_DIR <- "stage2_outputs"
dir.create(OUT_DIR, showWarnings = FALSE)

log_msg <- function(msg) {
  cat(sprintf("[%s][%s] %s\n", format(Sys.time(), "%H:%M:%S"), BIN_LABEL, msg))
  flush.console()
}

log_msg(sprintf("STAGE 2 START for bin '%s'", BIN_LABEL))

bin_sces       <- readRDS(file.path(IN_DIR, "bin_sces.rds"))
anchor_cache   <- readRDS(file.path(IN_DIR, "anchor_cache.rds"))
hvg_names      <- readRDS(file.path(IN_DIR, "hvg_names.rds"))
prop_targets   <- readRDS(file.path(IN_DIR, "prop_targets.rds"))
target_per_bin <- readRDS(file.path(IN_DIR, "target_per_bin.rds"))
maj_bin        <- readRDS(file.path(IN_DIR, "maj_bin.rds"))

if (!BIN_LABEL %in% names(bin_sces)) {
  stop(sprintf("Bin '%s' not in stage1 bin_sces (have: %s)",
               BIN_LABEL, paste(names(bin_sces), collapse=", ")))
}

sub_real_full <- bin_sces[[BIN_LABEL]]
n_full <- ncol(sub_real_full)
log_msg(sprintf("Loaded bin SCE: %d cells x %d genes", n_full, nrow(sub_real_full)))

# Majority bin: real-only, downsample
if (BIN_LABEL == maj_bin) {
  if (n_full > target_per_bin) {
    log_msg(sprintf("MAJORITY: downsampling %d -> %d", n_full, target_per_bin))
    set.seed(123)
    keep_idx <- sample(n_full, target_per_bin)
    sub_real <- sub_real_full[, keep_idx]
  } else {
    sub_real <- sub_real_full
  }
  saveRDS(sub_real, file.path(OUT_DIR, sprintf("%s.rds", BIN_LABEL)))
  log_msg(sprintf("MAJORITY done, saved %d real cells", ncol(sub_real)))
  quit(save="no", status=0)
}

# Minority bin: proportional downsample, then augment
target_real_min <- prop_targets[BIN_LABEL]
if (n_full > target_real_min) {
  log_msg(sprintf("MINORITY: downsampling %d -> %d real cells (proportional)",
                  n_full, target_real_min))
  set.seed(123)
  keep_idx <- sample(n_full, target_real_min)
  sub_real <- sub_real_full[, keep_idx]
} else {
  sub_real <- sub_real_full
}
n_real <- ncol(sub_real)
need <- target_per_bin - n_real
log_msg(sprintf("Need %d synthetic cells (%d real -> %d target)", need, n_real, target_per_bin))

if (need <= 0) {
  saveRDS(sub_real, file.path(OUT_DIR, sprintf("%s.rds", BIN_LABEL)))
  log_msg("No augmentation needed, saved real-only")
  quit(save="no", status=0)
}

# Build joint training set (target + anchors from other bins)
sub_train_target <- bin_sces[[BIN_LABEL]]
n_target_loaded  <- ncol(sub_train_target)

# Stratified subsample if too large
if (n_target_loaded > MAX_CELLS_PER_BIN) {
  ct_vec <- as.character(colData(sub_train_target)[[CELLTYPE_COL]])
  tab_ct <- table(ct_vec); tab_ct <- tab_ct[tab_ct > 0]
  cap <- as.integer(MAX_CELLS_PER_BIN)
  base_alloc <- floor(cap * as.numeric(tab_ct) / sum(tab_ct))
  alloc <- pmax(2L, as.integer(base_alloc))
  names(alloc) <- names(tab_ct)
  alloc <- pmin(alloc, as.integer(tab_ct))
  while (sum(alloc) > cap) {
    reducible <- names(alloc)[alloc > 2L]
    if (length(reducible) == 0) break
    largest <- reducible[which.max(alloc[reducible])]
    alloc[largest] <- alloc[largest] - 1L
  }
  if (sum(alloc) < cap) {
    room <- as.integer(tab_ct) - alloc
    names(room) <- names(tab_ct)
    while (sum(alloc) < cap && any(room > 0)) {
      addable <- names(room)[room > 0]
      pick <- addable[which.max(room[addable])]
      alloc[pick] <- alloc[pick] + 1L
      room[pick] <- room[pick] - 1L
    }
  }
  set.seed(42)
  keep_idx <- unlist(lapply(names(alloc), function(ct) {
    ids <- which(ct_vec == ct)
    sample(ids, alloc[[ct]], replace = FALSE)
  }))
  sub_train_target <- sub_train_target[, keep_idx]
  log_msg(sprintf("Stratified subsample: %d cells", ncol(sub_train_target)))
}

# Anchor injection
other_anchors <- anchor_cache[setdiff(names(anchor_cache), BIN_LABEL)]
target_cts <- unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]]))
other_anchors <- lapply(other_anchors, function(a) {
  ct_a <- as.character(colData(a)[[CELLTYPE_COL]])
  a[, ct_a %in% target_cts, drop=FALSE]
})
other_anchors <- other_anchors[vapply(other_anchors, ncol, integer(1)) > 0L]

if (length(other_anchors) == 0) {
  sub_train <- sub_train_target
  log_msg("WARN: no anchor cells available, using target only")
} else {
  common_genes <- rownames(sub_train_target)
  for (a in other_anchors) common_genes <- intersect(common_genes, rownames(a))
  pieces <- c(list(sub_train_target[common_genes, ]),
              lapply(other_anchors, function(a) a[common_genes, ]))
  sub_train <- do.call(cbind, pieces)
  
  # FIX: ensure library covariate is set (used in offset(log(library)) in mu_formula)
  if (!"library" %in% colnames(colData(sub_train)) || any(is.na(colData(sub_train)$library)))
    colData(sub_train)$library <- Matrix::colSums(assay(sub_train, ASSAY_USE))
  
  # Drop singletons in joint set
  ct_j <- as.character(colData(sub_train)[[CELLTYPE_COL]])
  tab_j <- table(ct_j); keep_ct <- names(tab_j)[tab_j >= 2L]
  sub_train <- sub_train[, ct_j %in% keep_ct]
  ct_clean <- as.character(colData(sub_train)[[CELLTYPE_COL]])
  colData(sub_train)[[CELLTYPE_COL]] <- droplevels(factor(ct_clean))
  colData(sub_train)[[BIN_COL]] <- as.character(colData(sub_train)[[BIN_COL]])
  log_msg(sprintf("Joint training set: %d target + %d anchor = %d total",
                  ncol(sub_train_target), ncol(sub_train) - ncol(sub_train_target),
                  ncol(sub_train)))
}

# Build mu_formula with sex/ethnicity covariates if available
build_mu_formula <- function(train_sce) {
  parts <- c("cell_type", "age_bin_10yr")
  extra <- character(0)
  for (covar in c()) {
    if (!covar %in% colnames(colData(train_sce))) next
    nl <- length(unique(na.omit(as.character(colData(train_sce)[[covar]]))))
    if (nl < 2L) {
      log_msg(sprintf("  [V5 8] '%s' has %d level -> dropped", covar, nl))
      next
    }
    parts <- c(parts, covar)
    extra <- c(extra, covar)
    log_msg(sprintf("  [V5 8] '%s' has %d levels -> included", covar, nl))
  }
  list(formula = paste(parts, collapse=" + "),
       other_covariates = c(BIN_COL, extra, "library"))
}

mu_spec <- build_mu_formula(sub_train)
log_msg(sprintf("mu_formula: %s", mu_spec$formula))

# Fit joint_data + marginals + copula ONCE, then loop simu_new per chunk
n_target_cells <- ncol(sub_train_target)
mat_for_imp <- assay(sub_train, ASSAY_USE)
zero_frac <- Matrix::rowMeans(mat_for_imp == 0)
sparsity_threshold <- if (n_target_cells < MIN_CELLS_INTERACTION) 0.95 else 0.80
imp_feat_vec <- zero_frac <= sparsity_threshold
if (sum(imp_feat_vec) < 2L) imp_feat_vec <- rep(TRUE, length(imp_feat_vec))
log_msg(sprintf("important_feature: %d / %d genes (threshold=%.2f)",
                sum(imp_feat_vec), length(imp_feat_vec), sparsity_threshold))

t0 <- Sys.time()
log_msg("Fitting construct_data...")
joint_data <- construct_data(
  sce = sub_train, assay_use = ASSAY_USE, celltype = CELLTYPE_COL,
  pseudotime = NULL, spatial = NULL,
  other_covariates = mu_spec$other_covariates,
  ncell = ncol(sub_train), corr_by = "cell_type",
  parallelization = "mcmapply", BPPARAM = NULL
)
log_msg(sprintf("construct_data done (%.1f sec)", as.numeric(Sys.time()-t0, units="secs")))

t0 <- Sys.time()
log_msg("Fitting marginals (ZINB)...")
marginal_res <- fit_marginal(
  data = joint_data, predictor = "gene",
  mu_formula = mu_spec$formula, sigma_formula = "cell_type",
  family_use = "nb", n_cores = N_CORES, usebam = FALSE,
  edf_flexible = TRUE,
  parallelization = "mcmapply", BPPARAM = NULL
)
log_msg(sprintf("fit_marginal done (%.1f sec)", as.numeric(Sys.time()-t0, units="secs")))

t0 <- Sys.time()
log_msg("Fitting copula...")
copula_res <- fit_copula(
  sce = sub_train, assay_use = ASSAY_USE, marginal_list = marginal_res,
  family_use = "nb", copula = "gaussian", n_cores = N_CORES,
  input_data = joint_data$dat, important_feature = imp_feat_vec,
  if_sparse = FALSE, parallelization = "mcmapply", BPPARAM = NULL
)
log_msg(sprintf("fit_copula done (%.1f sec)", as.numeric(Sys.time()-t0, units="secs")))

# Loop simu_new per chunk
target_dat_rows <- which(as.character(joint_data$dat[[BIN_COL]]) == BIN_LABEL)
sims <- list()
remaining <- need
chunk_id <- 1L

while (remaining > 0) {
  chunk <- min(CHUNK_SIZE_DEFAULT, remaining)
  n_ct_train <- length(unique(as.character(colData(sub_train_target)[[CELLTYPE_COL]])))
  if (chunk < 2L * n_ct_train) chunk <- 2L * n_ct_train
  
  log_msg(sprintf("Chunk %d: simulating %d cells (remaining=%d)", chunk_id, chunk, remaining))
  
  set.seed(123 + chunk_id)
  new_cov_idx <- sample(target_dat_rows, chunk,
                        replace = (length(target_dat_rows) < chunk))
  new_cov_df <- joint_data$dat[new_cov_idx, , drop=FALSE]
  
  # FIX: ensure library is in new_cov_df (offset uses it)
  if (!"library" %in% colnames(new_cov_df))
    new_cov_df$library <- median(colData(sub_train)$library)
  
  # Ensure each CT has >=2 rows (FIX 22)
  cov_cts <- as.character(new_cov_df[[CELLTYPE_COL]])
  ct_tab_cov <- table(cov_cts)
  thin_cts <- names(ct_tab_cov)[ct_tab_cov < 2L]
  if (length(thin_cts) > 0) {
    extras <- lapply(thin_cts, function(ct) {
      rows <- target_dat_rows[as.character(joint_data$dat[[CELLTYPE_COL]][target_dat_rows]) == ct]
      if (length(rows) == 0) return(NULL)
      ex_idx <- sample(rows, 2L - ct_tab_cov[[ct]], replace = TRUE)
      joint_data$dat[ex_idx, , drop=FALSE]
    })
    extras <- do.call(rbind, Filter(Negate(is.null), extras))
    if (!is.null(extras) && nrow(extras) > 0) new_cov_df <- rbind(new_cov_df, extras)
  }
  new_cov_df$corr_group <- as.character(new_cov_df[[CELLTYPE_COL]])
  rownames(new_cov_df) <- paste0("Cell", seq_len(nrow(new_cov_df)))
  
  para_list <- extract_para(
    sce = sub_train, assay_use = ASSAY_USE, marginal_list = marginal_res,
    n_cores = N_CORES, family_use = "nb",
    new_covariate = new_cov_df, parallelization = "mcmapply", BPPARAM = NULL,
    data = joint_data$dat
  )
  
  t_chunk <- Sys.time()
  new_count <- simu_new(
    sce = sub_train, assay_use = ASSAY_USE,
    mean_mat = para_list$mean_mat, sigma_mat = para_list$sigma_mat,
    zero_mat = para_list$zero_mat, quantile_mat = NULL,
    copula_list = copula_res$copula_list, n_cores = N_CORES,
    family_use = "nb", nonnegative = TRUE, nonzerovar = TRUE,
    input_data = joint_data$dat, new_covariate = new_cov_df,
    important_feature = copula_res$important_feature,
    parallelization = "mcmapply", BPPARAM = NULL,
    filtered_gene = joint_data$filtered_gene
  )
  log_msg(sprintf("  simu_new done (%.1f sec)", as.numeric(Sys.time()-t_chunk, units="secs")))
  
  if (is.vector(new_count) || is.null(dim(new_count))) {
    new_count <- matrix(new_count, ncol=1, dimnames=list(names(new_count), NULL))
  }
  if (!is.null(rownames(new_count)) && ncol(new_count) == nrow(sub_real) &&
      nrow(new_count) != nrow(sub_real)) {
    new_count <- t(new_count)
  }
  
  ref_genes <- rownames(sub_real)
  missing <- setdiff(ref_genes, rownames(new_count))
  if (length(missing) > 0) {
    pad <- matrix(0, nrow=length(missing), ncol=ncol(new_count),
                  dimnames=list(missing, colnames(new_count)))
    new_count <- rbind(new_count, pad)
  }
  new_count <- new_count[ref_genes, , drop=FALSE]
  colnames(new_count) <- paste0("synthetic_", BIN_LABEL, "_chunk", chunk_id, "_",
                                seq_len(ncol(new_count)))
  
  sim_cd <- new_cov_df
  rownames(sim_cd) <- colnames(new_count)
  sim <- SingleCellExperiment(assays=list(counts=as(new_count, "dgCMatrix")),
                              colData=DataFrame(sim_cd))
  
  # Align colData with sub_real
  ref_cols <- colnames(colData(sub_real))
  for (m in setdiff(ref_cols, colnames(colData(sim)))) {
    rv <- colData(sub_real)[[m]]
    if (is.double(rv))   colData(sim)[[m]] <- rep(NA_real_, ncol(sim))
    else if (is.integer(rv)) colData(sim)[[m]] <- rep(NA_integer_, ncol(sim))
    else colData(sim)[[m]] <- rep(NA_character_, ncol(sim))
  }
  colData(sim) <- colData(sim)[, ref_cols, drop=FALSE]
  colData(sim)[[BIN_COL]] <- rep(BIN_LABEL, ncol(sim))
  
  sims[[length(sims)+1]] <- sim
  log_msg(sprintf("Chunk %d complete: %d cells generated", chunk_id, ncol(sim)))
  
  remaining <- remaining - chunk
  chunk_id <- chunk_id + 1L
}

log_msg(sprintf("Combining %d chunks + real...", length(sims)))
out_bin <- do.call(cbind, c(list(sub_real), sims))
log_msg(sprintf("Final bin '%s': %d cells x %d genes", BIN_LABEL,
                ncol(out_bin), nrow(out_bin)))

saveRDS(out_bin, file.path(OUT_DIR, sprintf("%s.rds", BIN_LABEL)))
log_msg("STAGE 2 COMPLETE")
