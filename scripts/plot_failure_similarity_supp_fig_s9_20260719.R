#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
repo <- normalizePath(".", mustWork = TRUE)
out_dir <- file.path(
  repo,
  "outputs/paper_protocolb_v1/manuscript_reproduction"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

manifest_status <- system2(
  "python",
  c("scripts/validate_active_locked_run.py", "--manifest", "config/active_locked_run.json"),
  stdout = TRUE,
  stderr = TRUE
)
if (!any(grepl('"status": "ok"', manifest_status, fixed = TRUE))) {
  stop("Active locked-run validation did not return status ok:\n", paste(manifest_status, collapse = "\n"))
}

base_dir <- file.path(
  repo,
  "outputs/paper_protocolb_v1/manuscript_reproduction"
)
all_hm450_path <- file.path(
  base_dir,
  "failure_pair_all_hm450_tumor/failure_pair_confusion_30cpg_vs_all_hm450_tumor.tsv"
)
normal_path <- file.path(
  base_dir,
  "failure_pair_similarity_30cpg_tcga_normal/failure_pair_confusion_methylation_similarity_30cpg.tsv"
)
stats_path <- file.path(
  base_dir,
  "failure_pair_euclidean_review/failure_pair_euclidean_primary_spearman_stats.tsv"
)

for (p in c(all_hm450_path, normal_path, stats_path)) {
  if (!file.exists(p)) stop("Missing input: ", p)
}

all_hm450 <- read.delim(all_hm450_path, check.names = FALSE)
normal <- read.delim(normal_path, check.names = FALSE)
stats <- read.delim(stats_path, check.names = FALSE)

scale_similarity <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) {
    return(rep(NA_real_, length(x)))
  }
  1 - ((x - rng[1]) / (rng[2] - rng[1]))
}

fmt_rho <- function(x) sprintf("rho = %.2f", x)

fmt_p <- function(p) {
  if (is.na(p)) return("p = NA")
  if (p < 1e-14) return("p < 1e-14")
  if (p < 0.01) return(sprintf("p = %.3f", p))
  sprintf("p = %.3f", p)
}

extract_stat <- function(name) {
  row <- stats[stats$comparison == name, , drop = FALSE]
  if (nrow(row) != 1) stop("Could not find unique stats row: ", name)
  row
}

all_stat <- extract_stat("confusion_vs_tumor_all_hm450_euclidean_similarity")
normal_stat <- extract_stat("confusion_vs_normal_30_euclidean_similarity")

plot_all <- all_hm450[is.finite(all_hm450$tumor_all_hm450_euclidean) &
                        is.finite(all_hm450$pair_confusion_rate), ]
plot_all$plot_similarity <- scale_similarity(plot_all$tumor_all_hm450_euclidean)
plot_all$plot_confusion_percent <- 100 * plot_all$pair_confusion_rate
plot_all$panel <- "A"

plot_normal <- normal[is.finite(normal$normal_euclidean_30) &
                        is.finite(normal$pair_confusion_rate), ]
plot_normal$plot_similarity <- scale_similarity(plot_normal$normal_euclidean_30)
plot_normal$plot_confusion_percent <- 100 * plot_normal$pair_confusion_rate
plot_normal$panel <- "B"

plot_panel <- function(df, letter, xlab, stat_row) {
  par(mar = c(4.45, 4.05, 0.9, 0.7), mgp = c(2.75, 0.55, 0), tcl = -0.22)
  plot(
    df$plot_similarity,
    df$plot_confusion_percent,
    xlim = c(-0.03, 1.03),
    ylim = c(0, max(df$plot_confusion_percent, na.rm = TRUE) * 1.08),
    pch = 21,
    bg = adjustcolor("#2E8C8C", alpha.f = 0.55),
    col = adjustcolor("#23424A", alpha.f = 0.72),
    cex = 1.15,
    lwd = 0.45,
    xlab = xlab,
    ylab = "Pair confusion rate (%)",
    cex.lab = 0.88,
    cex.axis = 0.85,
    las = 1,
    bty = "l"
  )
  grid(nx = NA, ny = NULL, col = adjustcolor("#BCC7C9", alpha.f = 0.35), lwd = 0.5)
  fit <- lm(plot_confusion_percent ~ plot_similarity, data = df)
  xs <- seq(0, 1, length.out = 100)
  lines(xs, predict(fit, newdata = data.frame(plot_similarity = xs)),
        col = "#353A40", lwd = 2.1)
  text(
    x = par("usr")[1] + 0.04 * diff(par("usr")[1:2]),
    y = par("usr")[4] - 0.08 * diff(par("usr")[3:4]),
    labels = paste0(
      fmt_rho(stat_row$spearman_rho),
      "; ",
      fmt_p(stat_row$spearman_p_scipy)
    ),
    adj = c(0, 1),
    cex = 0.82,
    font = 2,
    col = "#202427"
  )
  mtext(letter, side = 3, adj = -0.08, line = 0.0, font = 2, cex = 1.15)
}

png_path <- file.path(out_dir, "supp_fig_s9_failure_similarity_robustness.png")
pdf_path <- file.path(out_dir, "supp_fig_s9_failure_similarity_robustness.pdf")

png(png_path, width = 7.0, height = 3.45, units = "in", res = 300, type = "cairo", pointsize = 9)
par(family = "Helvetica")
layout(matrix(c(1, 2), nrow = 1), widths = c(1, 1))
plot_panel(
  plot_all,
  "A",
  "Tumor methylation similarity\n(shared HM450 methylome)",
  all_stat
)
plot_panel(
  plot_normal,
  "B",
  "Normal-adjacent methylation similarity\n(30 MiniCpG probes)",
  normal_stat
)
dev.off()

pdf(pdf_path, width = 7.0, height = 3.45, useDingbats = FALSE, pointsize = 9)
par(family = "Helvetica")
layout(matrix(c(1, 2), nrow = 1), widths = c(1, 1))
plot_panel(
  plot_all,
  "A",
  "Tumor methylation similarity\n(shared HM450 methylome)",
  all_stat
)
plot_panel(
  plot_normal,
  "B",
  "Normal-adjacent methylation similarity\n(30 MiniCpG probes)",
  normal_stat
)
dev.off()

plot_data <- rbind(
  data.frame(
    panel = "A",
    class_a = plot_all$class_a,
    class_b = plot_all$class_b,
    confusion_rate = plot_all$pair_confusion_rate,
    confusion_percent = plot_all$plot_confusion_percent,
    source_distance = plot_all$tumor_all_hm450_euclidean,
    plotted_similarity = plot_all$plot_similarity,
    source = "TCGA tumor shared HM450 methylome"
  ),
  data.frame(
    panel = "B",
    class_a = plot_normal$class_a,
    class_b = plot_normal$class_b,
    confusion_rate = plot_normal$pair_confusion_rate,
    confusion_percent = plot_normal$plot_confusion_percent,
    source_distance = plot_normal$normal_euclidean_30,
    plotted_similarity = plot_normal$plot_similarity,
    source = "TCGA normal-adjacent 30 MiniCpG probes"
  )
)
write.table(
  plot_data,
  file.path(out_dir, "supp_fig_s9_failure_similarity_robustness_plot_data.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

summary_lines <- c(
  "{",
  sprintf('  "png": "%s",', png_path),
  sprintf('  "pdf": "%s",', pdf_path),
  sprintf('  "panel_a_n_pairs": %d,', nrow(plot_all)),
  sprintf('  "panel_a_spearman_rho": %.4f,', all_stat$spearman_rho),
  sprintf('  "panel_a_spearman_p": %.12g,', all_stat$spearman_p_scipy),
  sprintf('  "panel_b_n_pairs": %d,', nrow(plot_normal)),
  sprintf('  "panel_b_spearman_rho": %.4f,', normal_stat$spearman_rho),
  sprintf('  "panel_b_spearman_p": %.12g', normal_stat$spearman_p_scipy),
  "}"
)
writeLines(summary_lines, file.path(out_dir, "supp_fig_s9_failure_similarity_robustness_summary.json"))

message("Wrote: ", png_path)
message("Wrote: ", pdf_path)
