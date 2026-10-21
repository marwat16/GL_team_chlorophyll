# 01_load.R
# Lab 3 step 1 (1.1-1.4): load the data, set the factors and print an overview. The completeness
# checks are done by processing/3_build_runs_all.py.
#
# Input:  data/processed/runs_all.csv, one row per run (energy recomputed from EnergiBridge,
#         measured quality), written by the processing scripts
# Output: the data frame `runs` that every later script uses, and an overview on the console

if (!file.exists("analysis/00_common.R")) {
  stop("run the scripts from the package folder (the one with README.md and analysis/)", call. = FALSE)
}
source("analysis/00_common.R")

runs <- read_csv("data/processed/runs_all.csv", show_col_types = FALSE) |>
  mutate(task = factor(task, levels = TASK_KEYS, labels = TASKS),
         treatment = factor(treatment, levels = c("ai", "human"), labels = c("AI", "Human")),
         rep = factor(rep),
         energy_kJ = energy_J / 1000,
         power_W = energy_J / time_s,
         mem_MiB = .data[[MEM_COLUMN]])

# Ten groups (5 tasks x 2 treatments) with 10 or 20 runs each
glimpse(runs)
print(count(runs, task, treatment))
print(summary(select(runs, energy_J, time_s, cpu_pct_mean, gpu_pct_mean, mem_MiB, quality)))
