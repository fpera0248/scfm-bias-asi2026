suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

CKPTDIR     <- "checkpoints_age_augmentation_pilot_anchors"
OUTPUT_BASE <- "ILD_Age_Pilot"
all_bins    <- c("10_19","20_29","30_39","40_49","50_59","60_69","70_79")

cat("Loading checkpoints...\n")
bin_objects <- list()
for (b in all_bins) {
  f <- file.path(CKPTDIR, paste0("age_bin_", b, ".rds"))
  cat(sprintf("  Loading %s\n", f))
  bin_objects[[b]] <- readRDS(f)
  cat(sprintf("  -> %d cells x %d genes\n", ncol(bin_objects[[b]]), nrow(bin_objects[[b]])))
}

# Union of all genes across bins
all_hvg_genes <- Reduce(union, lapply(bin_objects, rownames))
cat(sprintf("Union of HVG genes across all bins: %d\n", length(all_hvg_genes)))

# Pad each bin to the union gene set with zeros
cat("Padding each bin to union gene set...\n")
bin_objects <- lapply(bin_objects, function(sce) {
  present  <- rownames(sce)
  missing  <- setdiff(all_hvg_genes, present)
  cnt      <- as(assay(sce, "counts"), "dgCMatrix")
  if (length(missing) > 0) {
    pad <- Matrix::sparseMatrix(
      i = integer(0), j = integer(0), x = numeric(0),
      dims = c(length(missing), ncol(sce)),
      dimnames = list(missing, colnames(sce))
    )
    cnt <- rbind(cnt, pad)
  }
  cnt <- cnt[all_hvg_genes, , drop = FALSE]
  rowData(sce) <- NULL
  SingleCellExperiment(
    assays  = list(counts = cnt),
    colData = colData(sce)
  )
})

cat("Combining...\n")
sce_combined <- do.call(cbind, bin_objects)
cat(sprintf("Combined: %d cells x %d genes\n", ncol(sce_combined), nrow(sce_combined)))

# Load full gene set from Proportional h5ad
cat("Loading full gene set from Proportional h5ad...\n")
sce_ref        <- zellkonverter::readH5AD(
  paste0(OUTPUT_BASE, "_Proportional_2495_AGE.h5ad"), use_hdf5 = TRUE
)
original_genes <- rownames(sce_ref)
cat(sprintf("Full gene set: %d genes\n", length(original_genes)))
rm(sce_ref); gc()

cat("Zero-padding to full gene set...\n")
present_genes  <- rownames(sce_combined)
missing_genes  <- setdiff(original_genes, present_genes)
current_counts <- as(assay(sce_combined, "counts"), "dgCMatrix")

if (length(missing_genes) > 0) {
  cat(sprintf("Padding %d missing genes\n", length(missing_genes)))
  pad_mat <- Matrix::sparseMatrix(
    i = integer(0), j = integer(0), x = numeric(0),
    dims = c(length(missing_genes), ncol(sce_combined)),
    dimnames = list(missing_genes, colnames(sce_combined))
  )
  full_counts <- rbind(current_counts, pad_mat)
} else {
  full_counts <- current_counts
}
full_counts <- full_counts[original_genes, , drop = FALSE]

cat("Adding source column...\n")
cd         <- colData(sce_combined)
source_col <- ifelse(grepl("^synthetic_", colnames(sce_combined)), "synthetic", "real")

sce_out <- SingleCellExperiment(
  assays  = list(counts = full_counts),
  colData = cd
)
colData(sce_out)$source <- source_col
rownames(sce_out)       <- original_genes

rebuild_for_write <- function(sce_obj) {
  cnt <- as(assay(sce_obj, "counts"), "dgCMatrix")
  n   <- ncol(sce_obj)
  cd_raw <- colData(sce_obj)
  col_list <- lapply(colnames(cd_raw), function(col) {
    v <- tryCatch({
      tmp <- cd_raw[[col]]
      vapply(seq_len(n), function(i) {
        xi <- tryCatch(tmp[[i]], error = function(e) NA)
        if (is.null(xi) || length(xi)==0 || (length(xi)==1 && is.na(xi)))
          return(NA_character_)
        paste(as.character(xi), collapse=";")
      }, character(1))
    }, error = function(e) rep(NA_character_, n))
    num <- suppressWarnings(as.numeric(v))
    if (!any(is.na(num) & !is.na(v))) return(num)
    v
  })
  names(col_list) <- colnames(cd_raw)
  rn_safe <- make.unique(as.character(colnames(sce_obj)), sep="_dup")
  colnames(cnt) <- rn_safe
  cd_new <- as.data.frame(col_list, row.names=rn_safe)
  SingleCellExperiment(assays=list(counts=cnt), colData=DataFrame(cd_new))
}

cat("Writing BalancedAugmented h5ad...\n")
zellkonverter::writeH5AD(
  rebuild_for_write(sce_out),
  paste0(OUTPUT_BASE, "_BalancedAugmented_1262Each_AGE.h5ad"),
  compression = "gzip"
)

cat("Writing BalancedUpsampled h5ad...\n")
sce_real     <- sce_out[, colData(sce_out)$source == "real"]
bin_real_vec <- as.character(colData(sce_real)$age_bin_10yr)
TARGET       <- 1262L
up_idx <- unlist(lapply(all_bins, function(b) {
  idx <- which(bin_real_vec == b)
  sample(idx, TARGET, replace = TRUE)
}))
sce_up <- sce_real[, up_idx]
zellkonverter::writeH5AD(
  rebuild_for_write(sce_up),
  paste0(OUTPUT_BASE, "_BalancedUpsampled_1262Each_AGE.h5ad"),
  compression = "gzip"
)

cat("Done.\n")
cat(sprintf("Active HVG genes in synthetic cells: %d / %d\n",
            length(all_hvg_genes), length(original_genes)))
cat("NOTE: Synthetic cells only express HVG genes (HVG-only limitation).\n")
