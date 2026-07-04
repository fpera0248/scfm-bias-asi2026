#!/usr/bin/env Rscript
# ============================================================
# STEP 0B STAGE 1: SETUP + HVG SELECTION + ANCHOR PRE-CACHE
# Age workflow (v6 parallel architecture)
# ============================================================
# Outputs to ./stage1_outputs/:
#   - hvg_names.rds            (~988 HVGs intersected with Geneformer vocab)
#   - prop_targets.rds         (per-bin proportional targets)
#   - bin_table.rds            (per-bin real cell counts)
#   - all_bins.rds             (sorted bin labels)
#   - target_per_bin.rds       (1262)
#   - prop_total.rds           (2495)
#   - down_target.rds          (25)
#   - validation_barcodes.rds  (10500 cell IDs to exclude)
#   - library_sizes.rds        (full-gene per-cell colSums by barcode)
#   - anchor_cache.rds         (per-bin SCE of anchor cells, CT-stratified)
#   - sce_detect.rds           (full validation-excluded SCE for downstream slicing)
# ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
  library(BiocParallel)
  library(RhpcBLASctl)
  library(scran)
  library(scuttle)
})

N_CORES <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))
blas_set_num_threads(N_CORES)
register(SerialParam(progressbar = FALSE))

INPUT_H5AD      <- "InterstitialLungDisease_RawCounts_AGE.h5ad"
VALIDATION_H5AD <- "ILD_Age_External_Validation_10500.h5ad"
OUTPUT_BASE     <- "ILD_Age_Pilot"

BIN_COL      <- "age_bin_10yr"
CELLTYPE_COL <- "cell_type"

PROPORTIONAL_SIZE <- 2500L
N_HVG             <- 1000L
MIN_CELLS         <- 5L
MIN_COUNTS        <- 10L
HVG_SAMPLE_SIZE   <- 2000L
MIN_CT_CELLS      <- 20L
ANCHOR_PER_CT_PER_BIN <- 2L
CT_UNKNOWN <- c("unknown","na","n/a","not reported","","nan")

OUT_DIR <- "stage1_outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

log_msg <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
  flush.console()
}

# ============================================================
# 1. Load validation barcodes
# ============================================================
log_msg("Loading validation barcodes...")
sce_val  <- readH5AD(VALIDATION_H5AD, use_hdf5 = TRUE)
val_barcodes <- colnames(sce_val)
rm(sce_val); gc()
log_msg(sprintf("  %d validation barcodes", length(val_barcodes)))
saveRDS(val_barcodes, file.path(OUT_DIR, "validation_barcodes.rds"))

# ============================================================
# 2. Pre-compute full-gene library sizes (by barcode)
# ============================================================
log_msg("Pre-computing full-gene library sizes...")
sce_lib <- readH5AD(INPUT_H5AD, use_hdf5 = FALSE)
if (!"counts" %in% assayNames(sce_lib)) {
  if ("X" %in% assayNames(sce_lib)) {
    assay(sce_lib, "counts") <- assay(sce_lib, "X")
  } else stop("No 'counts' or 'X' assay")
}
mat <- assay(sce_lib, "counts")
if (!inherits(mat, "dgCMatrix")) mat <- as(mat, "dgCMatrix")
LIBRARY_SIZES <- Matrix::colSums(mat)
log_msg(sprintf("  lib mean=%.0f sd=%.0f n=%d",
                mean(LIBRARY_SIZES), sd(LIBRARY_SIZES), length(LIBRARY_SIZES)))
saveRDS(LIBRARY_SIZES, file.path(OUT_DIR, "library_sizes.rds"))
rm(mat); gc()

# Reuse sce_lib as detect SCE (counts assay already attached)
sce_detect <- sce_lib
rm(sce_lib); gc()

# ============================================================
# 3. Materialize counts assay properly + exclude validation
# ============================================================
log_msg("Excluding validation cells...")
keep <- !colnames(sce_detect) %in% val_barcodes
sce_detect <- sce_detect[, keep]
log_msg(sprintf("  After validation exclude: %d cells", ncol(sce_detect)))

# Drop cells with NA / unknown bin label
bin_vec <- as.character(colData(sce_detect)[[BIN_COL]])
valid_bin_mask <- !is.na(bin_vec) & bin_vec != "" & !tolower(bin_vec) %in% c("nan","none")
sce_detect <- sce_detect[, valid_bin_mask]
log_msg(sprintf("  After dropping invalid bins: %d cells", ncol(sce_detect)))

# Drop unknown CTs
ct_raw <- as.character(colData(sce_detect)[[CELLTYPE_COL]])
valid_ct <- !is.na(ct_raw) & !(tolower(ct_raw) %in% CT_UNKNOWN) & nzchar(trimws(ct_raw))
sce_detect <- sce_detect[, valid_ct]
log_msg(sprintf("  After dropping unknown CTs: %d cells", ncol(sce_detect)))

# Force counts assay materialization
mat <- assay(sce_detect, "counts")
if (!inherits(mat, "dgCMatrix")) {
  log_msg("  Materializing counts to dgCMatrix...")
  assay(sce_detect, "counts") <- as(mat, "dgCMatrix")
}
assays(sce_detect) <- SimpleList(counts = assay(sce_detect, "counts"))

# ============================================================
# 4. Compute proportional targets
# ============================================================
log_msg("Computing proportional targets...")
bin_table <- table(colData(sce_detect)[[BIN_COL]])
all_bins  <- sort(names(bin_table))
log_msg(sprintf("  Detected bins: %s", paste(all_bins, collapse=", ")))
for (b in all_bins) log_msg(sprintf("    %s: %d", b, bin_table[b]))

total_real   <- sum(as.numeric(bin_table))
prop_targets <- sapply(as.numeric(bin_table),
                       function(n) floor(n * PROPORTIONAL_SIZE / total_real))
names(prop_targets) <- all_bins
log_msg("  Proportional targets:")
for (b in all_bins) log_msg(sprintf("    %s -> %d", b, prop_targets[b]))

prop_total      <- sum(prop_targets)
target_per_bin  <- max(prop_targets)
down_target     <- min(prop_targets)
maj_bin         <- all_bins[which.max(prop_targets)]
log_msg(sprintf("  Sum=%d  TARGET_PER_BIN=%d  DOWN_TARGET=%d  MAJ_BIN=%s",
                prop_total, target_per_bin, down_target, maj_bin))

saveRDS(bin_table, file.path(OUT_DIR, "bin_table.rds"))
saveRDS(all_bins, file.path(OUT_DIR, "all_bins.rds"))
saveRDS(prop_targets, file.path(OUT_DIR, "prop_targets.rds"))
saveRDS(prop_total, file.path(OUT_DIR, "prop_total.rds"))
saveRDS(target_per_bin, file.path(OUT_DIR, "target_per_bin.rds"))
saveRDS(down_target, file.path(OUT_DIR, "down_target.rds"))
saveRDS(maj_bin, file.path(OUT_DIR, "maj_bin.rds"))

# ============================================================
# 5. HVG selection (stratified sample) + Geneformer vocab filter
# ============================================================
log_msg("HVG selection from stratified sample...")
target_per_bin_hvg <- floor(HVG_SAMPLE_SIZE / length(all_bins))
sample_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(as.character(colData(sce_detect)[[BIN_COL]]) == b)
  if (length(idx) > target_per_bin_hvg) sample(idx, target_per_bin_hvg) else idx
}))
sce_hvg <- sce_detect[, sample_idx]
log_msg(sprintf("  HVG sample: %d cells x %d genes", ncol(sce_hvg), nrow(sce_hvg)))

count_mat    <- assay(sce_hvg, "counts")
gene_sums    <- Matrix::rowSums(count_mat)
gene_nonzero <- Matrix::rowSums(count_mat > 0)
keep_genes   <- (gene_sums >= MIN_COUNTS) & (gene_nonzero >= MIN_CELLS)
log_msg(sprintf("  Genes after filtering: %d / %d", sum(keep_genes), length(gene_sums)))

filt <- sce_hvg[keep_genes, ]
filt <- scuttle::logNormCounts(filt)
var_model <- scran::modelGeneVar(filt, BPPARAM = BiocParallel::SerialParam())
hvg_names <- scran::getTopHVGs(var_model, n = N_HVG)
log_msg(sprintf("  Selected %d HVGs", length(hvg_names)))

# Geneformer vocab filter
gf_vocab_path <- "geneformer_vocab_genes.txt"
if (file.exists(gf_vocab_path)) {
  gf_vocab     <- readLines(gf_vocab_path)
  hvg_stripped <- sub("\\.[0-9]+$", "", hvg_names)
  in_vocab     <- hvg_stripped %in% gf_vocab
  n_orig       <- length(hvg_names)
  hvg_names    <- hvg_names[in_vocab]
  log_msg(sprintf("  [GF VOCAB] %d / %d HVGs in vocab", length(hvg_names), n_orig))
  if (length(hvg_names) == 0) stop("No HVGs after Geneformer vocab filter")
} else {
  log_msg("  [GF VOCAB] WARNING: vocab file not found, skipping filter")
}

saveRDS(hvg_names, file.path(OUT_DIR, "hvg_names.rds"))
rm(sce_hvg, filt, fmat, fmsq, count_mat, gene_sums, gene_nonzero,
   var_model, keep_genes); gc()

# ============================================================
# 6. Pre-load per-group HVG SCEs + cache anchors
# ============================================================
log_msg("Pre-loading per-bin HVG SCEs and caching anchors...")
common_genes <- intersect(hvg_names, rownames(sce_detect))
log_msg(sprintf("  HVGs found in detect SCE: %d / %d", length(common_genes), length(hvg_names)))

bin_sces <- list()
anchor_cache <- list()
for (b in all_bins) {
  bin_mask <- as.character(colData(sce_detect)[[BIN_COL]]) == b
  sce_b <- sce_detect[common_genes, bin_mask]
  
  # Filter rare CTs
  ct_b <- as.character(colData(sce_b)[[CELLTYPE_COL]])
  tab_ct <- table(ct_b)
  keep_ct <- names(tab_ct)[tab_ct >= MIN_CT_CELLS]
  if (length(keep_ct) == 0) {
    log_msg(sprintf("  WARN: bin '%s' has no CTs with >=%d cells, skipping anchor cache", b, MIN_CT_CELLS))
    next
  }
  cells_keep <- ct_b %in% keep_ct
  sce_b <- sce_b[, cells_keep]
  ct_b  <- ct_b[cells_keep]
  
  ct_b[is.na(ct_b)] <- "unknown_cleaned"
  colData(sce_b)[[CELLTYPE_COL]] <- droplevels(factor(ct_b))
  colData(sce_b)[[BIN_COL]] <- as.character(colData(sce_b)[[BIN_COL]])
  
  # Attach library size by barcode lookup
  lib_vec <- LIBRARY_SIZES[colnames(sce_b)]
  lib_vec[is.na(lib_vec)] <- median(LIBRARY_SIZES, na.rm = TRUE)
  colData(sce_b)$library <- as.numeric(lib_vec)
  
  bin_sces[[b]] <- sce_b
  log_msg(sprintf("  bin '%s' SCE: %d cells x %d genes", b, ncol(sce_b), nrow(sce_b)))
  
  # Anchor cache: stratified by CT
  ct_now <- as.character(colData(sce_b)[[CELLTYPE_COL]])
  tab_now <- table(ct_now)
  anch_idx <- unlist(lapply(names(tab_now), function(ct) {
    ids <- which(ct_now == ct)
    n   <- min(ANCHOR_PER_CT_PER_BIN, length(ids))
    sample(ids, n)
  }))
  anchor_cache[[b]] <- sce_b[, anch_idx]
  log_msg(sprintf("    cached %d anchor cells (%d CTs)", length(anch_idx), length(tab_now)))
}

saveRDS(bin_sces, file.path(OUT_DIR, "bin_sces.rds"))
saveRDS(anchor_cache, file.path(OUT_DIR, "anchor_cache.rds"))

# Save the validation-excluded full SCE for stage 3 (Proportional dataset construction)
saveRDS(sce_detect, file.path(OUT_DIR, "sce_detect.rds"))

# Save original gene set (for stage 3 zero-padding)
original_genes <- rownames(sce_detect)
saveRDS(original_genes, file.path(OUT_DIR, "original_genes.rds"))

log_msg("STAGE 1 COMPLETE")
log_msg(sprintf("Outputs in: %s/", OUT_DIR))
