# 08_summary.R
# Lab 3 steps 5 and 6: the summary across tasks (forest plot of Section 4.5), the count of tasks in
# which AI is lower or higher and the descriptive tables.
#
# Input:  runs (01_load.R) and the test tables of 04 and 05 (output/tables/*.csv)
# Output: figure 08_forest_ratios, tables 08_direction_counts, 08_descriptives_energy and
#         08_descriptives_performance

if (!exists("runs")) source("analysis/01_load.R")
set.seed(SEED)

test_files <- c(rq11 = "output/tables/04_rq1_1_energy_tests.csv",
                rq12 = "output/tables/04_rq1_2_eqr.csv",
                rq2 = "output/tables/05_rq2_performance_tests.csv")
stopifnot("run 04_rq1_energy.R and 05_rq2_performance.R first" = all(file.exists(test_files)))
tests <- map(test_files, \(f) read_csv(f, show_col_types = FALSE))


# ---- Forest plot (Section 4.5) -------------------------------------------------------------------
# Every AI/Human ratio with its 95 % bootstrap interval, one panel per variable and one row per task.
# All panels share one log axis, so a factor of 10 has the same length everywhere and the panels can
# be compared. Mean power is not tested (Section 4.2) but shown because energy = power x time.
power <- map_dfr(TASKS, function(t) {
  d <- filter(runs, task == t)
  ai <- d$power_W[d$treatment == "AI"]
  human <- d$power_W[d$treatment == "Human"]
  boot <- replicate(B, median(sample(ai, replace = TRUE)) / median(sample(human, replace = TRUE)))
  tibble(task = t, variable = "power", ratio = median(ai) / median(human),
         ratio_lo = quantile(boot, 0.025), ratio_hi = quantile(boot, 0.975))
})

# Power is not tested, so it comes last and the tested variables stay together
PANELS <- c(energy = "Energy\nRQ1.1", eqr = "EQR\nRQ1.2", time = "Time\nRQ2", cpu = "CPU\nRQ2",
            gpu = "GPU\nRQ2", memory = "Memory\nRQ2", power = "Power\nnot tested")
forest <- bind_rows(
  tests$rq12 |>
    filter(!is.na(EQR)) |>
    transmute(task, variable = "eqr", ratio = EQR, ratio_lo = EQR_lo, ratio_hi = EQR_hi),
  tests$rq11 |> select(task, variable, ratio, ratio_lo, ratio_hi),
  power,
  tests$rq2 |> filter(!is.na(ratio)) |> select(task, variable, ratio, ratio_lo, ratio_hi)
) |>
  mutate(task = factor(task, levels = rev(TASKS)), variable = factor(variable, levels = names(PANELS)),
         label = fmt_sig(ratio, 3),
         # labels near the ends of the axis are moved inwards
         hjust = case_when(ratio < 0.05 ~ 0.15, ratio > 5 ~ 0.85, TRUE ~ 0.5))

# Cells without a ratio get a short reason instead of an empty row
gpu_med <- runs |>
  group_by(task, treatment) |>
  summarise(m = median(gpu_pct_mean), .groups = "drop") |>
  pivot_wider(names_from = treatment, values_from = m)
gpu_reason <- function(t) {
  g <- filter(gpu_med, task == t)
  if (g$AI == 0 && g$Human == 0) return("no GPU use")
  sprintf("AI %s%%, Human %s%%", fmt_sig(g$AI, 2), fmt_sig(g$Human, 2))
}
no_ratio <- expand_grid(task = TASKS, variable = names(PANELS)) |>
  anti_join(mutate(forest, task = as.character(task), variable = as.character(variable)),
            by = c("task", "variable")) |>
  mutate(reason = ifelse(variable == "eqr", "no quality data", map_chr(task, gpu_reason)),
         task = factor(task, levels = rev(TASKS)), variable = factor(variable, levels = names(PANELS)))

LIMITS <- c(0.01, 20)
MID <- sqrt(prod(LIMITS))  # middle of the log axis, where the reasons are written
p <- ggplot(forest, aes(ratio, task)) +
  annotate("rect", xmin = 1, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#ecebe5") +
  geom_vline(xintercept = 1, colour = AXIS, linewidth = 0.4) +
  geom_errorbar(aes(xmin = ratio_lo, xmax = ratio_hi), width = 0, linewidth = 0.6, colour = INK,
                orientation = "y") +
  geom_point(shape = 21, fill = INK, colour = "white", stroke = 0.4, size = 2) +
  geom_label(aes(label = label, hjust = hjust), vjust = -0.55, size = 2.3, colour = INK2, fill = "white",
             border.colour = NA, label.padding = unit(0.6, "pt"), label.r = unit(0, "pt")) +
  geom_label(data = no_ratio, aes(x = MID, label = reason), size = 2.1, colour = MUTED, fontface = "italic",
             fill = "white", border.colour = NA, label.padding = unit(0.8, "pt"), label.r = unit(0, "pt")) +
  facet_wrap(~variable, nrow = 1, labeller = as_labeller(PANELS)) +
  scale_x_log10(limits = LIMITS, breaks = c(0.01, 0.1, 1, 10), labels = c("0.01", "0.1", "1", "10"),
                expand = expansion(0)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 0.8))) +
  labs(x = "AI / Human ratio (log scale). Below 1 means AI is lower, above 1 (shaded) means AI is higher",
       y = NULL) +
  theme_report() +
  theme(panel.grid.major.y = element_blank(), panel.spacing = unit(14, "pt"),
        axis.text.y = element_text(colour = INK, size = rel(0.95)))

# Whether the interval bars are visible next to the points. A point is about 2 mm wide and a panel
# about 19 mm wide for 3.3 decades of the log axis, so a point covers about 0.35 decades.
POINT_DECADES <- 0.35
ci_width <- log10(forest$ratio_hi / forest$ratio_lo)
ci_note <- if (all(ci_width < POINT_DECADES)) {
  "All 95% bootstrap intervals are narrower than the points and are given in the test tables."
} else {
  "The bar is the 95% bootstrap interval where it is wider than the point."
}
any_log <- any(c(tests$rq11$ratio_of, tests$rq2$ratio_of) == "geometric means", na.rm = TRUE)
save_plot(p, "08_forest_ratios", h = 2.6, use = "report", caption = paste(
  "Ratio of the AI value to the human value for every variable and task, on one shared log axis.",
  "For energy, time, CPU, GPU, memory and power the ratio is that of the medians",
  if (any_log) "or, where the test ran on log values, of the geometric means." else "per run.",
  "EQR is the energy-to-quality ratio built from the means, and a value below 1 means AI needs less",
  "energy for the quality it reaches.",
  ci_note,
  "Mean power is shown because energy is power times time, but it is not tested."))


# ---- Direction count (Section 4.5) ---------------------------------------------------------------
# In how many tasks AI is significantly lower, significantly higher or not different (Holm-adjusted),
# with the tasks named in each cell
direction <- bind_rows(
  tests$rq11 |> select(task, variable, ratio, p_holm),
  tests$rq12 |> filter(!is.na(EQR)) |> transmute(task, variable = "eqr", ratio = EQR, p_holm),
  tests$rq2 |> select(task, variable, ratio, p_holm)
) |>
  mutate(result = case_when(is.na(p_holm) ~ "not tested",
                            p_holm >= ALPHA ~ "no significant difference",
                            ratio < 1 ~ "AI lower",
                            TRUE ~ "AI higher"),
         task = factor(task, levels = TASKS))

RESULTS <- c("AI lower", "AI higher", "no significant difference")
counts <- direction |>
  group_by(variable, result) |>
  summarise(n = n(), tasks = paste(sort(task), collapse = ", "), .groups = "drop") |>
  complete(variable, result = RESULTS, fill = list(n = 0L, tasks = "")) |>
  mutate(variable = factor(variable, levels = names(PANELS))) |>
  arrange(variable)

# LaTeX version (replication package): one cell "n (tasks)" per variable and result, without the
# columns that are empty
empty_cols <- RESULTS[map_lgl(RESULTS, function(r) all(filter(counts, result == r)$n == 0))]
cells <- counts |>
  mutate(cell = ifelse(n == 0, "0", paste0(n, " (", tasks, ")"))) |>
  select(variable, result, cell) |>
  pivot_wider(names_from = result, values_from = cell) |>
  select(variable, any_of(setdiff(RESULTS, empty_cols))) |>
  mutate(variable = recode(as.character(variable), eqr = "EQR", energy = "Energy", time = "Execution time",
                           cpu = "CPU utilization", gpu = "GPU utilization", memory = "Memory increase")) |>
  rename(Variable = variable)

all_significant <- if ("no significant difference" %in% empty_cols) {
  "Every comparison is significant, so there is no column for no significant difference."
} else {
  ""
}
empty_directions <- intersect(c("AI lower", "AI higher"), empty_cols)
no_direction <- if (length(empty_directions) > 0) {
  sprintf("No comparison falls in %s.", paste(empty_directions, collapse = " or "))
} else {
  ""
}
save_table(counts, "08_direction_counts", paste(
  "Number of tasks in which the AI implementation is significantly lower or higher than the human one",
  "(Holm-adjusted p < 0.05), with the tasks in brackets.",
  all_significant, no_direction),
  tex = cells)


# ---- Descriptive tables (Lab 3 steps 6.1-6.2, lecture 07-A and 09) -------------------------------
# Decimals follow the standard deviation, so the SD has two significant digits and the other columns
# are given to the same decimal place
desc_energy <- runs |>
  group_by(task, treatment) |>
  summarise(n = n(), mean_kJ = mean(energy_kJ), sd_kJ = sd(energy_kJ), cv_pct = 100 * sd_kJ / mean_kJ,
            min_kJ = min(energy_kJ), median_kJ = median(energy_kJ), max_kJ = max(energy_kJ),
            .groups = "drop")
decimals <- pmax(0, 1 - floor(log10(desc_energy$sd_kJ)))
save_table(desc_energy, "08_descriptives_energy", paste(
  "Energy per run in kJ per task and treatment. CV is the coefficient of variation.",
  "Mean, SD, minimum, median and maximum are given to the decimal place of the second significant",
  "digit of the SD."),
  wide = TRUE,
  tex = desc_energy |>
    transmute(Task = task, Treatment = treatment, n = as.character(n),
              `Mean (kJ)` = fmt_dec(mean_kJ, decimals), `SD (kJ)` = fmt_dec(sd_kJ, decimals),
              `CV (%)` = fmt_dec(cv_pct, 2), `Min (kJ)` = fmt_dec(min_kJ, decimals),
              `Median (kJ)` = fmt_dec(median_kJ, decimals), `Max (kJ)` = fmt_dec(max_kJ, decimals)))

# cpu_pct_mean and gpu_pct_mean are overwritten by their group mean inside summarise(), so later
# expressions see the mean. That is why gpu_used comes first.
desc_perf <- runs |>
  group_by(task, treatment) |>
  summarise(time_s_mean = mean(time_s), time_s_sd = sd(time_s), time_cv_pct = 100 * time_s_sd / time_s_mean,
            power_W_mean = mean(power_W), cpu_pct_mean = mean(cpu_pct_mean), gpu_used = any(gpu_pct_mean > 0),
            gpu_pct_mean = mean(gpu_pct_mean), mem_MiB_mean = mean(mem_MiB), mem_MiB_sd = sd(mem_MiB),
            gpu_mem_MiB_mean = mean(gpu_mem_peak_MiB), quality_metric = first(quality_metric),
            quality_mean = mean(quality), .groups = "drop")
# GPU memory of the treatments that do not use the GPU, i.e. what is in use when the GPU is idle
gpu_idle <- runs |>
  group_by(task, treatment) |>
  filter(all(gpu_pct_mean == 0)) |>
  pull(gpu_mem_peak_MiB) |>
  median()
q_tasks <- desc_perf |> filter(!is.na(quality_mean)) |> distinct(task, quality_metric)
stopifnot("the caption below assumes lower-is-better quality scores" =
            all(filter(runs, quality_metric %in% q_tasks$quality_metric)$quality_lower_is_better))
unmeasured <- unique(as.character(desc_perf$task[is.na(desc_perf$quality_mean)]))
save_table(desc_perf, "08_descriptives_performance", paste(
  "Performance metrics and quality per task and treatment, as means over the runs.",
  "CPU and GPU give the mean utilization.",
  "Time gives the coefficient of variation in brackets.",
  "Memory is the increase of the used main memory of the whole machine during a run.",
  "GPU memory is the peak memory in use on the GPU during a run, which includes the",
  sprintf("%s MiB in use when the GPU is idle.", fmt_sig(gpu_idle)),
  "A dash marks a treatment that does not use the GPU.",
  sprintf("Quality is the measured score, %s, and lower is better for all of them.",
          and_list(paste(QUALITY_LONG[q_tasks$quality_metric], "for", q_tasks$task))),
  sprintf("It was not measured for %s in this experiment.", and_list(unmeasured))),
  tex = desc_perf |>
    transmute(Task = task, Treatment = treatment,
              `Time (s), mean (CV)` = paste0(fmt_sig(time_s_mean), " (", fmt_sig(time_cv_pct, 2), "%)"),
              `Power (W)` = fmt_sig(power_W_mean), `CPU (%)` = fmt_sig(cpu_pct_mean, 2),
              `GPU (%)` = ifelse(gpu_used, fmt_sig(gpu_pct_mean, 2), "–"),
              `Memory increase (MiB)` = fmt_sig(mem_MiB_mean),
              `GPU memory (MiB)` = ifelse(gpu_used, fmt_sig(gpu_mem_MiB_mean), "–"),
              Quality = ifelse(is.na(quality_mean), "–",
                               paste(QUALITY_LABELS[quality_metric], fmt_sig(quality_mean)))),
  wide = TRUE)
