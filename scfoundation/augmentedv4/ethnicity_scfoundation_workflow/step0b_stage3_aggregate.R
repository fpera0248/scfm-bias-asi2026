#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 3: AGGREGATOR
# Reads stage2_synth_<group>.rds for all groups, restores
# full gene set, generates 4 fairness datasets + summary CSV.
# ============================================================

cat("\n=== STAGE 3: AGGREGATE ===\n")

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

# CONFIG
INPUT_H5AD   <- "InterstitialLungDisease_RawCounts_ETHNICITY.h5ad"
OUTPUT_BASE  <- "ILD_Ethnicity_Pilot"
BIN_COL      <- "self_reported_ethnicity"
SEX_COL      <- "sex"
ASSAY_USE    <- "counts"
UNKNOWN_VALUES <- c("unknown", "na", "n/a", "not reported", "", "nan",
                    "multiethnic", "na na", "not applicable", "prefer not to say")

log_con <- file("stage3_log.txt", open = "at")
on.exit(close(log_con), add = TRUE)
heartbeat <- function(msg, newline = FALSE) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg, if (newline) "\n" else "")
  cat(line, file = log_con); cat(line, file = stderr()); flush(log_con)
}

heartbeat("\n=== STAGE 3 START ===\n", TRUE)

ss <- readRDS("stage1_shared_state.rds")
all_bins       <- ss$all_bins
prop_targets   <- ss$prop_targets
TARGET_PER_BIN <- ss$TARGET_PER_BIN
MAJ_BIN        <- ss$MAJ_BIN
original_genes <- ss$original_genes
rd_full        <- ss$rd_full

# ============================================================
# 1. Load all per-group outputs
# ============================================================

heartbeat("Loading per-group outputs...\n", TRUE)
bin_objects <- list()
for (b in all_bins) {
  safe_b <- gsub("[^a-zA-Z0-9_]", "_", b)
  path <- sprintf("stage2_synth_%s.rds", safe_b)
  if (!file.exists(path)) {
    heartbeat(sprintf("MISSING: %s\n", path), TRUE)
    next
  }
  bin_objects[[b]] <- readRDS(path)
  heartbeat(sprintf("  %s: %d cells\n", b, ncol(bin_objects[[b]])), TRUE)
}

if (length(bin_objects) == 0L) stop("No per-group outputs loaded")

# ============================================================
# 2. Combine and restore full gene set
# ============================================================

heartbeat("Combining all groups...\n", TRUE)
sce_combined <- do.call(cbind, bin_objects)
heartbeat(sprintf("Combined: %d cells x %d genes\n", ncol(sce_combined), nrow(sce_combined)), TRUE)

present_genes <- rownames(sce_combined)
missing_genes <- setdiff(original_genes, present_genes)
current_counts <- as(assay(sce_combined, ASSAY_USE), "dgCMatrix")

if (length(missing_genes) > 0) {
  pad_mat <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(missing_genes), ncol(sce_combined)),
    dimnames = list(missing_genes, colnames(sce_combined)))
  full_counts <- rbind(current_counts, pad_mat)
} else full_counts <- current_counts

full_counts <- as(full_counts, "dgCMatrix")
gene_order <- match(original_genes, rownames(full_counts))
valid_order <- gene_order[!is.na(gene_order)]
full_counts <- full_counts[valid_order, , drop = FALSE]
rownames(full_counts) <- original_genes[!is.na(gene_order)]

# Restore real cells' full-gene counts from source
heartbeat("Restoring full-gene counts for real cells...\n", TRUE)
sce_full <- readRDS("stage1_sce_full.rds")
src_counts <- as(as.matrix(assay(sce_full, ASSAY_USE)), "dgCMatrix")
source_names <- colnames(sce_full)
combined_names <- colnames(sce_combined)
is_synthetic <- grepl("^synthetic_", combined_names)
real_positions <- which(!is_synthetic)
real_names <- combined_names[real_positions]
src_lookup <- match(real_names, source_names)
found_mask <- !is.na(src_lookup)

heartbeat(sprintf("Real cells found in source: %d/%d\n",
                  sum(found_mask), length(real_positions)), TRUE)
if (sum(found_mask) > 0) {
  # Step 1: pull just the real cells (2574 columns) from src_counts. Sparse
  # column subsetting is O(nnz_of_selected) - cheap.
  heartbeat("  Subsetting real cells from source...\n", TRUE)
  t0 <- Sys.time()
  src_real <- src_counts[, src_lookup[found_mask], drop = FALSE]
  rm(src_counts); gc()
  heartbeat(sprintf("  subset done in %.1fs (dim %d x %d)\n",
                    as.numeric(Sys.time()-t0, units="secs"),
                    nrow(src_real), ncol(src_real)), TRUE)
  
  # Step 2: row-reorder + pad missing genes via direct triplet construction.
  # Avoids the O(n^2) sparse in-place assignment entirely.
  heartbeat("  Building real_block via triplet construction...\n", TRUE)
  t0 <- Sys.time()
  src_gene_idx <- match(original_genes, rownames(src_real))
  valid_gene <- !is.na(src_gene_idx)
  src_trip <- as(src_real, "TsparseMatrix")
  # src_trip@i is 0-indexed source row; map to 0-indexed target row in original_genes
  # First build an inverse lookup: src_row -> original_genes_row (or NA)
  inv_lookup <- match(rownames(src_real), original_genes)  # length nrow(src_real)
  new_i <- inv_lookup[src_trip@i + 1L] - 1L                # 0-indexed target rows
  keep <- !is.na(new_i)
  real_block <- Matrix::sparseMatrix(
    i = new_i[keep] + 1L,                                  # back to 1-indexed for sparseMatrix()
    j = src_trip@j[keep] + 1L,
    x = src_trip@x[keep],
    dims = c(length(original_genes), ncol(src_real)),
    dimnames = list(original_genes, colnames(src_real))
  )
  rm(src_real, src_trip); gc()
  heartbeat(sprintf("  triplet build done in %.1fs (dim %d x %d, nnz %d)\n",
                    as.numeric(Sys.time()-t0, units="secs"),
                    nrow(real_block), ncol(real_block), length(real_block@x)), TRUE)
  
  # Step 3: split full_counts into synthetic columns, then cbind syn + real, then reorder.
  heartbeat("  Splitting and recombining columns...\n", TRUE)
  t0 <- Sys.time()
  syn_positions <- which(is_synthetic)
  if (length(syn_positions) > 0) {
    syn_block <- full_counts[, syn_positions, drop = FALSE]
  } else {
    syn_block <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
      dims = c(nrow(full_counts), 0L),
      dimnames = list(rownames(full_counts), character(0)))
  }
  colnames(real_block) <- combined_names[real_positions[found_mask]]
  combined_block <- cbind(syn_block, real_block)
  reorder_idx <- match(combined_names, colnames(combined_block))
  full_counts <- combined_block[, reorder_idx, drop = FALSE]
  heartbeat(sprintf("  recombine done in %.1fs\n", as.numeric(Sys.time()-t0, units="secs")), TRUE)
  rm(syn_block, real_block, combined_block); gc()
}

cd <- colData(sce_combined)
rd_genes_present <- intersect(original_genes, rownames(rd_full))
rd_subset <- if (length(rd_genes_present) > 0) { rd_full[rd_genes_present, , drop = FALSE] } else { rd_full[integer(0), , drop = FALSE] }
rd_genes_missing <- setdiff(original_genes, rownames(rd_full))
if (length(rd_genes_missing) > 0) {
  rd_pad <- as.data.frame(matrix(NA, nrow = length(rd_genes_missing),
                                 ncol = ncol(rd_subset),
                                 dimnames = list(rd_genes_missing, colnames(rd_subset))))
  rd_combined <- rbind(as.data.frame(rd_subset), rd_pad)
} else rd_combined <- as.data.frame(rd_subset)
rd_combined <- rd_combined[original_genes, , drop = FALSE]

sce_combined <- SingleCellExperiment(assays = list(counts = full_counts),
                                     colData = cd, rowData = DataFrame(rd_combined))
rownames(sce_combined) <- original_genes
colData(sce_combined)$source <- ifelse(grepl("^synthetic_", colnames(sce_combined)),
                                        "synthetic", "real")
rm(sce_full); gc()

# ============================================================
# 3. Sanitize and write
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
        if (is.null(xi) || length(xi) == 0 || (length(xi) == 1 && is.na(xi))) return(NA_character_)
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
    col_list <- col_list[not_all_na]
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

sce_combined <- sanitize_coldata_for_h5ad(sce_combined)

heartbeat("Writing output files...\n", TRUE)

out_full <- paste0(OUTPUT_BASE, "_Full_BalancedAugmented_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined), out_full, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_full), TRUE)

sce_real <- sce_combined[, colData(sce_combined)$source == "real"]
out_real <- paste0(OUTPUT_BASE, "_RealOnly_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_real), out_real, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_real), TRUE)

sce_syn <- sce_combined[, colData(sce_combined)$source == "synthetic"]
out_syn <- paste0(OUTPUT_BASE, "_SyntheticOnly_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_syn, drop_all_na = TRUE),
                         out_syn, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_syn), TRUE)

out_sum <- paste0(OUTPUT_BASE, "_Summary_ETHNICITY.csv")
write.csv(as.data.frame(table(
  Ethnicity = colData(sce_combined)[[BIN_COL]],
  Source = colData(sce_combined)$source)),
  out_sum, row.names = FALSE)
heartbeat(sprintf("Wrote: %s\n", out_sum), TRUE)

# ============================================================
# 4. Fairness datasets: Proportional, Upsampled, Downsampled
# ============================================================

sample_exact <- function(idx, n, replace = FALSE) {
  n <- as.integer(round(as.numeric(n)))
  if (is.na(n) || n < 0) stop("invalid n")
  if (n == 0) return(integer(0))
  if (!replace && length(idx) < n) stop("not enough cells")
  sample(idx, n, replace = replace)
}

# Proportional - rebuild from fresh h5ad read since use_hdf5=TRUE didn't survive .rds
heartbeat("Building Proportional dataset (fresh h5ad read)...\n", TRUE)
sce_detect <- zellkonverter::readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
if ("counts" %in% assayNames(sce_detect)) {
  # ok
} else if ("X" %in% assayNames(sce_detect)) {
  assay(sce_detect, "counts") <- assay(sce_detect, "X")
}
assays(sce_detect) <- SimpleList(counts = as(assay(sce_detect, "counts"), "dgCMatrix"))
keep_val <- !colnames(sce_detect) %in% ss$VALIDATION_BARCODES
sce_detect <- sce_detect[, keep_val]
eth_clean <- tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]])))
sex_clean <- tolower(trimws(as.character(colData(sce_detect)[[SEX_COL]])))
unknown_eth <- c("unknown", "na", "n/a", "not reported", "", "nan",
                 "multiethnic", "na na", "not applicable", "prefer not to say")
unknown_sex <- c("unknown", "na", "n/a", "not reported", "", "nan")
keep_demo <- !(eth_clean %in% unknown_eth) & !is.na(eth_clean) &
             !(sex_clean %in% unknown_sex) & !is.na(sex_clean)
sce_detect <- sce_detect[, keep_demo]
colData(sce_detect)[[BIN_COL]] <- tolower(trimws(as.character(colData(sce_detect)[[BIN_COL]])))
heartbeat(sprintf("  sce_detect rebuilt: %d cells x %d genes\n", ncol(sce_detect), nrow(sce_detect)), TRUE)
bin_labels_det <- colData(sce_detect)[[BIN_COL]]
set.seed(123)
prop_sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_labels_det == b)
  n_b <- prop_targets[[b]]
  if (n_b <= 0 || length(idx) == 0) integer(0) else sample(idx, n_b)
}))
sce_prop <- sce_detect[, prop_sample_idx]
out_prop <- paste0(OUTPUT_BASE, "_Proportional_2497_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_prop), out_prop, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_prop), TRUE)

# BalancedAugmented (alias of full)
out_bal_aug <- paste0(OUTPUT_BASE, "_BalancedAugmented_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_combined), out_bal_aug, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_bal_aug), TRUE)

# Upsampled (real with replacement)
bin_real_vec <- tolower(trimws(as.character(colData(sce_real)[[BIN_COL]])))
up_idx <- unlist(lapply(all_bins, function(b)
  sample_exact(which(bin_real_vec == b), TARGET_PER_BIN, replace = TRUE)))
sce_up <- sce_real[, up_idx]
out_up <- paste0(OUTPUT_BASE, "_BalancedUpsampled_", TARGET_PER_BIN, "Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_up), out_up, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_up), TRUE)

# Downsampled
bin_real_tbl <- table(bin_real_vec)
DOWN_TARGET <- as.integer(min(bin_real_tbl))
down_idx <- unlist(lapply(all_bins, function(b)
  sample_exact(which(bin_real_vec == b), DOWN_TARGET, replace = FALSE)))
sce_down <- sce_real[, down_idx]
out_down <- paste0(OUTPUT_BASE, "_Downsampled_48Each_ETHNICITY.h5ad")
zellkonverter::writeH5AD(rebuild_for_write(sce_down), out_down, compression = "gzip")
heartbeat(sprintf("Wrote: %s\n", out_down), TRUE)

heartbeat("\n=== STAGE 3 COMPLETE ===\n", TRUE)
heartbeat("Output files:\n", TRUE)
heartbeat(sprintf("  %s\n", out_full), TRUE)
heartbeat(sprintf("  %s\n", out_real), TRUE)
heartbeat(sprintf("  %s\n", out_syn), TRUE)
heartbeat(sprintf("  %s\n", out_sum), TRUE)
heartbeat(sprintf("  %s\n", out_prop), TRUE)
heartbeat(sprintf("  %s\n", out_bal_aug), TRUE)
heartbeat(sprintf("  %s\n", out_up), TRUE)
heartbeat(sprintf("  %s\n", out_down), TRUE)
