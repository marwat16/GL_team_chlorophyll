# run_all.R
# Runs the whole Section 6 analysis in order. Each step can also be run on its own.
#   Rscript analysis/run_all.R        (from the package folder, the one with README.md)
#
# Input:  data/processed/runs_all.csv
# Output: output/figures/*.pdf|png (+ captions.csv) and output/tables/*.csv|tex, named after the
#         step that made them

if (!file.exists("analysis/00_common.R")) {
  stop("run this from the package folder (the one with README.md and analysis/)", call. = FALSE)
}
options(warn = 1)  # print each warning under the step that raised it
steps <- c("01_load.R", "02_explore.R", "03_normality.R", "04_rq1_energy.R", "05_rq2_performance.R",
           "06_rq3_correlations.R", "07_models.R", "08_summary.R")
for (step in steps) {
  cat("\n==== ", step, " ====\n", sep = "")
  source(file.path("analysis", step))
}
cat("\nDone. Figures in output/figures, tables in output/tables.\n")
