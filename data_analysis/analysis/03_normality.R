# 03_normality.R
# Lab 3 step 3 (3.1-3.4): normality per task and treatment, normalization, Q-Q and density plots.
# The same Shapiro-Wilk check runs again inside compare() (00_common.R), where it picks the test of
# 04 and 05 (rule of Section 4.5).
#
# Input:  runs (01_load.R)
# Output: tables 03_shapiro_wilk and 03_bestnormalize_energy, figures 03_qq_energy_time_memory,
#         03_qq_cpu_gpu and 03_energy_density (full width and one column, _col)

if (!exists("runs")) source("analysis/01_load.R")
set.seed(SEED)

# Variables checked (name = short name, value = column) and how captions and axis titles name them
VARS <- c(energy = "energy_J", time = "time_s", cpu = "cpu_pct_mean", gpu = "gpu_pct_mean",
          memory = "mem_MiB")
LABELS <- c(energy = "energy", time = "execution time", cpu = "CPU utilization", gpu = "GPU utilization",
            memory = "memory increase")


# ---- Shapiro-Wilk per task and treatment (Lab 3 step 3.1) ----------------------------------------
normality <- map_dfr(names(VARS), function(v) {
  runs |>
    mutate(x = .data[[VARS[[v]]]]) |>
    group_by(task, treatment) |>
    summarise(variable = v, distinct = n_distinct(x), sw_p_raw = sw_p(x), sw_p_log = sw_p(log(x)),
              .groups = "drop")
}) |>
  mutate(normal_raw = sw_p_raw > ALPHA, normal_log = sw_p_log > ALPHA)

# LaTeX version (replication package), one row per group and one column per variable with the
# raw-scale p-value. The log-scale p-values stay in the CSV. A dash marks groups with fewer than three
# distinct values.
sw_tex <- normality |>
  mutate(cell = ifelse(is.na(sw_p_raw), "–", fmt_p(sw_p_raw)),
         variable = factor(variable, levels = names(VARS))) |>
  select(task, treatment, variable, cell) |>
  arrange(variable) |>
  pivot_wider(names_from = variable, values_from = cell) |>
  arrange(task, treatment) |>
  rename(Task = task, Treatment = treatment, Energy = energy, Time = time, CPU = cpu, GPU = gpu,
         Memory = memory)

# Groups in which the log scale gives a different normality decision
log_changes <- normality |> filter(!is.na(sw_p_raw), normal_raw != normal_log | is.na(normal_log))
log_sentence <- if (nrow(log_changes) == 0) {
  "On the log scale the decision is the same in every group."
} else {
  sprintf("On the log scale the decision changes in %d of the %d tested groups.",
          nrow(log_changes), sum(!is.na(normality$sw_p_raw)))
}
save_table(normality, "03_shapiro_wilk", paste(
  "Shapiro-Wilk p-values per task and treatment on the raw scale. A p-value below 0.05 rejects normality.",
  "A dash marks a group with fewer than three distinct values (the GPU usage of implementations that",
  "run on the CPU only).",
  log_sentence),
  tex = sw_tex)


# ---- Normalization of energy (Lab 3 steps 3.2-3.3, replication package) --------------------------
# bestNormalize picks a transformation per task and treatment, then Shapiro-Wilk is run again.
# With 10 or 20 runs per group the cross-validation of bestNormalize has too few runs per fold,
# so it chooses on the in-sample normality statistic (out_of_sample = FALSE). The test rule of
# Section 4.5 uses the log transform instead. The energy is standardized first. On raw joules
# (up to 1.6 million) Yeo-Johnson picks a power near -5, which maps every run to the same number.
TRANSFORM_NAMES <- c(no_transform = "none", center_scale = "centre and scale", yeojohnson = "Yeo-Johnson",
                     boxcox = "Box-Cox", log_x = "log", sqrt_x = "square root", exp_x = "exponential",
                     arcsinh_x = "arcsinh", double_reverse_log = "double reverse log")
norm_check <- runs |>
  group_by(task, treatment) |>
  group_modify(function(d, key) {
    x <- as.numeric(scale(d$energy_J))
    fit <- bestNormalize(x, allow_orderNorm = FALSE, out_of_sample = FALSE, quiet = TRUE)
    chosen <- class(fit$chosen_transform)[1]
    tibble(transform = coalesce(unname(TRANSFORM_NAMES[chosen]), chosen),
           sw_p_before = sw_p(x), sw_p_after = sw_p(predict(fit)))
  }) |>
  ungroup()
save_table(norm_check, "03_bestnormalize_energy",
           paste("Shapiro-Wilk p-value of the standardized energy per run before and after the",
                 "transformation chosen by bestNormalize. bestNormalize ranks the transformations by the",
                 "Pearson P statistic, so the Shapiro-Wilk p-value can also go down. Every group is normal",
                 "without a transformation, so the analysis uses the raw values."),
           tex = norm_check |>
             transmute(Task = task, Treatment = treatment, Transformation = transform,
                       `p before` = fmt_p(sw_p_before), `p after` = fmt_p(sw_p_after)))


# ---- Q-Q plots (Lab 3 step 3.4, lecture 07-A) ----------------------------------------------------
# One row per variable and one panel per task, with AI and Human in the same panel. The values are
# standardized within each task and treatment, which keeps the shape of the distribution but puts
# every group on one scale, so the reference is the identity line. Points inside the band are what a
# normal sample of this size would give. Both treatments of a task have the same number of runs and
# so the same band, which is drawn once. The text gives the Shapiro-Wilk p-value of each treatment.
qq_plot <- function(vars) {
  row_titles <- setNames(paste0(toupper(substr(LABELS[vars], 1, 1)), substring(LABELS[vars], 2)), vars)
  d <- map_dfr(vars, function(v) mutate(runs, variable = v, x = .data[[VARS[[v]]]])) |>
    group_by(variable, task, treatment) |>
    filter(n_distinct(x) >= 3) |>
    mutate(z = (x - mean(x)) / sd(x)) |>
    ungroup() |>
    mutate(variable = factor(variable, levels = vars))
  band <- d |>
    group_by(variable, task) |>
    filter(treatment == first(treatment)) |>
    ungroup()
  lim <- max(3.3, ceiling(10 * max(abs(d$z))) / 10 + 0.3)

  sw_text <- normality |>
    filter(variable %in% vars) |>
    mutate(variable = factor(variable, levels = vars),
           label = paste(treatment, case_when(distinct == 1 ~ "constant",
                                              is.na(sw_p_raw) ~ "not tested",
                                              sw_p_raw < 0.001 ~ "p < 0.001",
                                              TRUE ~ paste("p =", fmt_p(sw_p_raw)))),
           face = ifelse(!is.na(sw_p_raw) & sw_p_raw < ALPHA, "bold", "plain"),
           y = ifelse(treatment == "AI", 0.95, 0.74) * lim)

  ggplot(d, aes(sample = z)) +
    stat_qq_band(data = band, fill = MUTED, alpha = 0.2, bandType = "pointwise", identity = TRUE) +
    geom_abline(slope = 1, intercept = 0, colour = MUTED, linewidth = 0.35) +
    stat_qq_point(aes(fill = treatment), shape = 21, colour = "white", stroke = 0.25, size = 1.4) +
    geom_text(data = sw_text, aes(x = -2.2, y = y, label = label, fontface = face), hjust = 0,
              size = 2.2, colour = INK2, inherit.aes = FALSE) +
    facet_grid(variable ~ task, labeller = labeller(variable = as_labeller(row_titles))) +
    scale_x_continuous(labels = minus_labels) +
    scale_y_continuous(labels = minus_labels) +
    scale_fill_manual(values = COLORS) +
    guides(fill = guide_legend(override.aes = list(size = 2.2))) +
    coord_cartesian(xlim = c(-2.3, 2.3), ylim = c(-lim, lim)) +
    labs(x = "Theoretical normal quantiles", y = "Standardized value") +
    theme_report() +
    THEME_GROUPS +
    theme(strip.text.y = element_text(angle = -90, hjust = 0.5, size = rel(0.85)))
}
qq_caption <- function(vars) paste(
  "Normal Q-Q plots of the", and_list(LABELS[vars]), "per run in each task, with the Shapiro-Wilk",
  "p-value of each treatment (in bold where p < 0.05).")

save_plot(qq_plot(c("energy", "time", "memory")), "03_qq_energy_time_memory", h = 4.4, use = "report",
          caption = qq_caption(c("energy", "time", "memory")))
save_plot(qq_plot(c("cpu", "gpu")), "03_qq_cpu_gpu", h = 3.1, caption = paste(
  qq_caption(c("cpu", "gpu")),
  "Implementations that do not use the GPU stay at 0% in every run and are not tested."))


# ---- Density of the energy (lecture 07-A, report) ------------------------------------------------
# One panel per task and treatment, each on its own scale. It shows the shape that the bars of the
# energy figure hide, and the label gives the coefficient of variation with the same two decimals as
# the descriptive energy table. The x-range is widened on each side by half the range of the runs or
# by three bandwidths of the kernel, whichever is more, so that the tails are not cut off.
density_info <- runs |>
  group_by(task, treatment) |>
  summarise(label = paste0("CV ", fmt_dec(100 * sd(energy_J) / mean(energy_J), 2), "%"),
            pad = max(0.5 * diff(range(energy_kJ)), 3 * 1.2 * bw.nrd0(energy_kJ)),
            lo = min(energy_kJ) - pad,
            hi = max(energy_kJ) + pad,
            .groups = "drop") |>
  group_panels()

# Two or three round breaks that keep clear of the panel edges, so that the labels of neighbouring
# panels do not run into each other
inner_breaks <- function(lims) {
  pad <- 0.08 * diff(lims)
  for (k in 2:8) {
    b <- pretty(lims, n = k)
    b <- b[b >= lims[1] + pad & b <= lims[2] - pad]
    if (length(b) >= 2) break
  }
  if (length(b) > 3) b <- b[seq(1, length(b), by = 2)]
  b
}

p <- ggplot(group_panels(runs), aes(energy_kJ, colour = treatment, fill = treatment)) +
  geom_blank(data = pivot_longer(density_info, c(lo, hi), values_to = "energy_kJ")) +
  geom_density(alpha = 0.12, linewidth = 0.45, adjust = 1.2) +
  geom_rug(length = unit(4, "pt"), linewidth = 0.3, alpha = 0.5) +
  geom_text(data = density_info, aes(x = -Inf, y = Inf, label = label), hjust = -0.1, vjust = 1.4,
            size = 2.3, colour = INK2, inherit.aes = FALSE) +
  facet_wrap(~panel, nrow = 2, scales = "free") +
  scale_x_continuous(breaks = inner_breaks, labels = scales::label_comma()) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  scale_colour_manual(values = COLORS, guide = "none") +
  scale_fill_manual(values = COLORS, guide = "none") +
  labs(x = "Energy per run (kJ)", y = "Density") +
  theme_report() +
  THEME_GROUPS +
  theme(axis.text.y = element_blank(), panel.grid.major.y = element_blank())
density_caption <- paste(
  "Distribution of the energy per run in each task and treatment as a kernel density estimate.",
  "The ticks are the single runs and CV is the coefficient of variation.")
save_plot(p, "03_energy_density", h = 3.2, use = "report", caption = density_caption)
# The same figure in one column, one task per row with AI on the left and human on the right
save_plot(p + facet_wrap(~panel, ncol = 2, dir = "v", scales = "free"), "03_energy_density_col", w = COL_W,
          h = 4.4, use = "report", caption = density_caption)
