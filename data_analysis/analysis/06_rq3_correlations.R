# 06_rq3_correlations.R
# RQ3 (Section 4.5): correlation of energy with each performance metric, per task and per treatment.
# Pearson if both variables are normal, Spearman otherwise, with Kendall's tau next to it (lecture 07-A).
# Pooling the two treatments would turn the treatment effect into a correlation, so they stay separate.
# Section 4.5 reports the coefficients with intervals and does not test them, so the p-values (raw,
# Holm and Benjamini-Hochberg over all coefficients) are kept in the CSV only.
#
# Input:  runs (01_load.R)
# Output: table 06_rq3_correlations, figures 06_rq3_coefficients (full width and one column, _col) and
#         06_energy_vs_* (one per metric)

if (!exists("runs")) source("analysis/01_load.R")

# Metrics (name = column) with their axis title, short name for tables and name in captions
METRICS <- c(time_s = "Execution time (s)", cpu_pct_mean = "Mean CPU utilization (%)",
             gpu_pct_mean = "Mean GPU utilization (%)", mem_MiB = "Memory increase (MiB)")
METRIC_SHORT <- c(time_s = "Time", cpu_pct_mean = "CPU", gpu_pct_mean = "GPU", mem_MiB = "Memory")
METRIC_WORDS <- c(time_s = "execution time", cpu_pct_mean = "mean CPU utilization",
                  gpu_pct_mean = "mean GPU utilization", mem_MiB = "memory increase")


# ---- Coefficients --------------------------------------------------------------------------------
# 95 % interval via Fisher's z, with the Bonett-Wright standard error for Spearman
fisher_ci <- function(r, n, method) {
  se <- switch(method,
               pearson = 1 / sqrt(n - 3),
               spearman = sqrt((1 + r^2 / 2) / (n - 3)))
  z <- atanh(r) + c(-1, 1) * qnorm(0.975) * se
  tanh(z)
}

# Correlation of the energy with one metric in one group. A metric with fewer than three distinct
# values (for example 0% GPU in every run) gets a note instead of a coefficient.
correlate <- function(energy, y, metric) {
  if (n_distinct(y) < 3) {
    note <- ifelse(n_distinct(y) == 1, "constant", "fewer than 3 distinct values")
    return(tibble(metric = metric, note = note))
  }
  method <- if (isTRUE(sw_p(energy) > ALPHA && sw_p(y) > ALPHA)) "pearson" else "spearman"
  main <- suppressWarnings(cor.test(energy, y, method = method))
  kendall <- suppressWarnings(cor.test(energy, y, method = "kendall"))
  ci <- fisher_ci(unname(main$estimate), length(energy), method)
  tibble(metric = metric, method = method, r = unname(main$estimate), r_lo = ci[1], r_hi = ci[2],
         p = main$p.value, kendall_tau = unname(kendall$estimate), kendall_p = kendall$p.value)
}

rq3 <- map_dfr(names(METRICS), function(v) {
  runs |>
    group_by(task, treatment) |>
    group_modify(function(d, key) correlate(d$energy_J, d[[v]], v)) |>
    ungroup()
}) |>
  mutate(p_holm = p.adjust(p, "holm"), p_bh = p.adjust(p, "BH"))


# ---- Report table, one row per task and treatment and one column per metric ----------------------
rq3_tex <- rq3 |>
  mutate(cell = ifelse(is.na(r), "–",
                       paste0(fmt_r(r, r_lo, r_hi), ifelse(method == "spearman", "†", ""))),
         metric = factor(METRIC_SHORT[metric], levels = METRIC_SHORT)) |>
  select(task, treatment, metric, cell) |>
  arrange(metric) |>
  pivot_wider(names_from = metric, values_from = cell) |>
  arrange(task, treatment) |>
  rename(Task = task, Treatment = treatment)
save_table(rq3, "06_rq3_correlations", paste(
  "RQ3, correlation of energy with each performance metric per task and treatment, with its 95% interval.",
  "The coefficient is Pearson's $r$, or Spearman's $\\rho$ where marked with $^\\dagger$.",
  "A dash marks a metric with fewer than three distinct values."),
  tex = rq3_tex, wide = TRUE)


# ---- Report figure, all coefficients at once -----------------------------------------------------
# One panel per metric and one row per task with AI above Human, the line is the 95 % interval.
# Filled points are Pearson's r and open points Spearman's rho. The shared x-axis runs from -1 to 1.
# The rows are placed by hand (task position +-0.17) so that a group without a coefficient keeps its
# place and the other treatment does not move to the middle of the row.
row_y <- function(task, treatment) {
  as.numeric(factor(task, levels = rev(TASKS))) + ifelse(treatment == "AI", 0.17, -0.17)
}
coef_data <- rq3 |>
  filter(!is.na(r)) |>
  mutate(y = row_y(task, treatment),
         metric = factor(METRIC_SHORT[metric], levels = METRIC_SHORT),
         shape = ifelse(method == "spearman", "Spearman", "Pearson"))
# Groups without a coefficient get a note in their own row
gaps <- rq3 |>
  filter(is.na(r)) |>
  mutate(metric = factor(METRIC_SHORT[metric], levels = METRIC_SHORT)) |>
  group_by(task, metric) |>
  summarise(note = if (n() == 2) "constant in both" else paste(first(treatment), "constant"),
            y = if (n() == 2) row_y(first(task), "AI") - 0.17 else row_y(first(task), first(treatment)),
            .groups = "drop")

p <- ggplot(coef_data, aes(r, y, colour = treatment)) +
  geom_vline(xintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_linerange(aes(xmin = r_lo, xmax = r_hi), linewidth = 0.6) +
  geom_point(data = filter(coef_data, method == "pearson"), aes(fill = treatment), shape = 21,
             colour = "white", stroke = 0.35, size = 1.9) +
  geom_point(data = filter(coef_data, method == "spearman"), shape = 21, fill = "white", stroke = 0.7,
             size = 1.7) +
  geom_text(data = gaps, aes(x = 0, y = y, label = note), colour = MUTED, size = 2.1,
            fontface = "italic", inherit.aes = FALSE) +
  facet_wrap(~metric, nrow = 1, drop = FALSE) +
  scale_x_continuous(limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1),
                     labels = c(paste0(MINUS_SIGN, "1"), paste0(MINUS_SIGN, "0.5"), "0", "0.5", "1"),
                     expand = expansion(add = 0.04)) +
  scale_y_continuous(breaks = seq_along(TASKS), labels = rev(TASKS), limits = c(0.55, length(TASKS) + 0.45),
                     expand = expansion(0)) +
  scale_colour_manual(values = COLORS, guide = "none") +
  scale_fill_manual(values = COLORS) +
  guides(fill = guide_legend(override.aes = list(size = 2.4))) +
  labs(x = "Correlation with energy per run (Pearson r or Spearman \u03c1, 95% interval)", y = NULL) +
  theme_report() +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(colour = INK, size = rel(0.95)))
coef_caption <- paste(
  "Correlation of the energy per run with each performance metric per task and treatment, with its 95%",
  "interval. Filled points are Pearson's r and open points Spearman's rho.")
save_plot(p, "06_rq3_coefficients", h = 2.6, use = "report", caption = coef_caption)
# The same figure in one column, with two metrics per row and the axis title on two lines
p <- p + facet_wrap(~metric, nrow = 2, drop = FALSE) +
  labs(x = "Correlation with energy per run\n(Pearson r or Spearman \u03c1, 95% interval)")
save_plot(p, "06_rq3_coefficients_col", w = COL_W, h = 3.9, use = "report", caption = coef_caption)


# ---- Scatter plots (Lab 3 step 5.4, lecture 07-A) ------------------------------------------------
# Energy against each metric per task and treatment. The panel title gives the coefficient of the
# table above with its interval. The least-squares line is drawn only where the coefficient is
# Pearson's r, since Spearman's rho measures a monotonic and not a linear relation.
fmt_coef <- function(x) sub("^-", MINUS_SIGN, sprintf("%.2f", x))

scatter_plot <- function(v) {
  coefs <- rq3 |>
    filter(metric == v) |>
    group_panels() |>
    mutate(coef_text = paste0(ifelse(method == "pearson", "r = ", "ρ = "), fmt_coef(r),
                              " [", fmt_coef(r_lo), ", ", fmt_coef(r_hi), "]"),
           title = paste0(panel, "\n", coef_text))
  # Groups without a coefficient are left out (named in the caption), the rest keep the
  # AI-then-Human order
  shown <- as.character(coefs$panel[!is.na(coefs$r)])
  d <- runs |>
    mutate(x = .data[[v]]) |>
    group_panels() |>
    filter(panel %in% shown)
  pearson_runs <- semi_join(d, filter(coefs, method == "pearson"), by = "panel")

  ggplot(d, aes(x, energy_kJ)) +
    geom_smooth(data = pearson_runs, aes(colour = treatment, fill = treatment), method = "lm",
                formula = y ~ x, linewidth = 0.45, alpha = 0.12) +
    geom_point(aes(fill = treatment), shape = 21, colour = "white", stroke = 0.3, size = 1.6) +
    facet_wrap(~panel, nrow = if (length(shown) > 5) 2 else 1, scales = "free", drop = TRUE,
               labeller = as_labeller(setNames(coefs$title, as.character(coefs$panel)))) +
    scale_x_continuous(breaks = few_breaks(2), labels = num_lab) +
    scale_y_continuous(breaks = few_breaks(3), labels = num_lab, expand = expansion(mult = 0.08)) +
    scale_colour_manual(values = COLORS, guide = "none") +
    scale_fill_manual(values = COLORS, guide = "none") +
    labs(x = METRICS[[v]], y = "Energy per run (kJ)") +
    theme_report() +
    THEME_GROUPS +
    theme(axis.text.x = element_text(size = rel(0.85)), axis.text.y = element_text(size = rel(0.85)))
}
scatter_caption <- function(v, extra = "") paste(
  "Energy per run against", METRIC_WORDS[[v]], "for each task and treatment, each panel on its own scales.",
  "The panel title gives Pearson's r or Spearman's rho with its 95% interval, and the line is a",
  "least-squares fit with its 95% band where the coefficient is Pearson's r.", extra)

# Short runs last only a few EnergiBridge samples, so their lengths fall on a few steps of the
# sampling grid, and one sample more also adds one more interval of energy
grid_steps <- runs |>
  group_by(task, treatment) |>
  summarise(k = n_distinct(round(time_s / SAMPLE_S)), .groups = "drop") |>
  filter(k <= 4)
time_note <- ""
if (nrow(grid_steps) > 0) {
  time_note <- sprintf(paste(
    "Execution time is measured on the %s s sampling grid of EnergiBridge. In %s the run lengths fall",
    "on only %d to %d steps of this grid, and one sample more also adds one more interval of energy,",
    "so part of the correlation in these groups comes from the grid."),
    SAMPLE_S, and_list(paste(grid_steps$task, treatment_word(grid_steps$treatment))),
    min(grid_steps$k), max(grid_steps$k))
}

save_plot(scatter_plot("time_s"), "06_energy_vs_time", h = 3.6,
          caption = scatter_caption("time_s", time_note))
save_plot(scatter_plot("cpu_pct_mean"), "06_energy_vs_cpu", h = 3.6,
          caption = scatter_caption("cpu_pct_mean"))

# Groups that never use the GPU have no coefficient and no panel
gpu_missing <- rq3 |> filter(metric == "gpu_pct_mean", is.na(r))
save_plot(scatter_plot("gpu_pct_mean"), "06_energy_vs_gpu", h = if (nrow(gpu_missing) >= 5) 2.1 else 3.6,
          caption = scatter_caption("gpu_pct_mean", sprintf(
            "%s do not use the GPU, stay at 0%% in every run and are not shown.",
            and_list(paste(gpu_missing$task, treatment_word(gpu_missing$treatment))))))
save_plot(scatter_plot("mem_MiB"), "06_energy_vs_mem", h = 3.6, caption = scatter_caption("mem_MiB"))
