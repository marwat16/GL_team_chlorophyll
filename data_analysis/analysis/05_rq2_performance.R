# 05_rq2_performance.R
# RQ2 (Section 4.5): execution time, CPU and GPU utilization and memory, AI versus human per task.
# 17 tests: time, CPU and memory in all 5 tasks, GPU only in Geospatial and ZAPBench (Section 4.2),
# the two tasks that use the GPU in both treatments. Holm and Benjamini-Hochberg over these 17.
#
# Input:  runs (01_load.R)
# Output: table 05_rq2_performance_tests (the CSV is also read by 08_summary.R) and figure
#         05_rq2_by_task

if (!exists("runs")) source("analysis/01_load.R")
set.seed(SEED)

GPU_TESTED <- c("Geospatial", "ZAPBench")


# ---- Tests and report table ----------------------------------------------------------------------
rq2 <- run_tests(runs, c(time = "time_s", cpu = "cpu_pct_mean", gpu = "gpu_pct_mean", memory = "mem_MiB"),
                 tasks_for = function(v) if (v == "gpu") GPU_TESTED else TASKS) |>
  mutate(p_holm = p.adjust(p, "holm"), p_bh = p.adjust(p, "BH"))

t2 <- tex_tests(rq2) |>
  mutate(Variable = recode(Variable, time = "Time (s)", cpu = "CPU (%)", gpu = "GPU (%)",
                           memory = "Memory increase (MiB)"))
t2 <- squeeze_tests(t2, paste(
  "RQ2, execution time, mean CPU and GPU utilization and memory increase per run, AI versus human in",
  "each task.",
  centre_sentence(rq2),
  "GPU utilization is tested only in the two tasks that use the GPU in both treatments."))
save_table(rq2, "05_rq2_performance_tests", t2$caption, tex = t2$tex, wide = TRUE)


# ---- Report figure, all four metrics -------------------------------------------------------------
# One row per metric in the layout of the energy figure of 02, stacked so that the columns are the
# tasks. Utilization in percent keeps the fixed 0-100 scale in every panel (lecture 09), time and
# memory get their own axis per task. Tasks that run on the CPU only keep an empty GPU panel.
ONE_THREAD <- 100 / THREADS
cpu_only <- runs |>
  group_by(task) |>
  filter(all(gpu_pct_mean == 0)) |>
  pull(task) |>
  unique() |>
  as.character()
gpu_runs <- filter(runs, !task %in% cpu_only)

time_row <- function(...) plot_by_task(runs, "time_s", unit = "s", phrase = "%s %s longer", ...)
cpu_row <- function(...) {
  plot_by_task(runs, "cpu_pct_mean", unit = "%", percent = TRUE, phrase = "%s %s higher",
               ref_line = ONE_THREAD, ...)
}
gpu_row <- function(...) {
  plot_by_task(gpu_runs, "gpu_pct_mean", unit = "%", percent = TRUE, phrase = "%s %s higher",
               zero = "AI uses no GPU", empty_title = "CPU only", empty_note = "no GPU use", ...)
}
memory_row <- function(...) plot_by_task(runs, "mem_MiB", unit = "MiB", ...)

p <- plot_grid(time_row(ylab = "Time (s)"),
               cpu_row(ylab = "CPU (%)", show_task = FALSE),
               gpu_row(ylab = "GPU (%)", show_task = FALSE),
               memory_row(ylab = "Memory increase (MiB)", show_task = FALSE),
               ncol = 1, align = "v", axis = "lr", rel_heights = c(1.2, 1, 1, 1),
               labels = c("(a)", "(b)", "(c)", "(d)"), label_size = 8, label_fontfamily = "Helvetica",
               label_colour = INK, hjust = 0, label_x = 0.002)
save_plot(p, "05_rq2_by_task", h = 6.2, use = "report", caption = paste(
  "RQ2 per task, (a) execution time, (b) mean CPU utilization, (c) mean GPU utilization and (d) memory",
  "increase per run of the AI and the human implementation.",
  "The bar and its label give the median, each point is one run and the panel title says which",
  "treatment has the higher median and by what factor.",
  "Time and memory have their own y-axis from 0 in every panel, CPU and GPU utilization share one 0 to",
  "100% scale.",
  sprintf(paste("The dashed line in (b) marks one fully busy hardware thread (%s%% of the %d threads of",
                "the Intel Core i9-11900KF)."),
          fmt_sig(ONE_THREAD), THREADS),
  sprintf("%s run on the CPU only, so their GPU panels in (c) are empty.", and_list(cpu_only)),
  "Memory is the increase of the used main memory of the whole machine from the start to the peak of a",
  "run, without GPU memory."))
