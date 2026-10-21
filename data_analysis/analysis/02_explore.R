# 02_explore.R
# Lab 3 step 2 (2.1-2.3): visual exploration of energy, plus two checks for the threats to validity
# (drift over the execution order and heat carried over from the previous run).
#
# Input:  runs (01_load.R)
# Output: figures 02_energy_by_task and 02_energy_vs_start_temp,
#         table 02_carry_over
#
# Tasks differ by up to 1,500x in energy, so the energy figure has one panel per task with its own
# y-axis. The temperature figure shows the deviation from the group mean in percent, which puts all
# groups on one scale. Tasks are blocks and are never compared with each other directly.

if (!exists("runs")) source("analysis/01_load.R")

energy_cv <- runs |>
  group_by(task, treatment) |>
  summarise(cv = 100 * sd(energy_J) / mean(energy_J), .groups = "drop")

# Axis labels for a deviation in percent, with an explicit sign
signed_pct <- function(x) paste0(ifelse(x > 0, "+", ifelse(x < 0, MINUS_SIGN, "")), abs(x), "%")


# ---- Energy of every run per task (Lab 3 step 2.2, report) ---------------------------------------
p <- plot_by_task(runs, "energy_kJ", "Energy per run (kJ)", unit = "kJ")
save_plot(p, "02_energy_by_task", h = 2.5, use = "report", caption = paste(
  "Energy per run of the AI and the human implementation in each task.", BY_TASK_NOTE,
  sprintf(paste("Within a group the runs differ very little (coefficient of variation at most %s%%),",
                "so the points overlap and a box plot would shrink to a line. The shape of each",
                "distribution is shown in the density figure."),
          fmt_sig(max(energy_cv$cv), 3))))


# ---- Drift over the execution order (Lab 3 step 2.3, threats) ------------------------------------
# Energy as the deviation from the group mean in percent, so all ten groups share one scale.
deviation <- runs |>
  group_by(task, treatment) |>
  mutate(dev = 100 * (energy_J / mean(energy_J) - 1)) |>
  ungroup()

# Monotonic trend over the execution order per group (Spearman), Holm-corrected over the ten groups
spearman_trend <- function(x, y) {
  test <- suppressWarnings(cor.test(x, y, method = "spearman"))
  tibble(rho = test$estimate, p = test$p.value)
}
drift <- deviation |>
  group_by(task, treatment) |>
  summarise(spearman_trend(order, dev), .groups = "drop") |>
  mutate(p_holm = p.adjust(p, "holm"))


# ---- Carry-over of heat from the previous run (threats) ------------------------------------------
# The GPU temperature at the start of a run depends on what ran before it, and a warmer machine draws
# more power. R2 and slope of a straight line per group.
temp_fit <- function(dev, temp) {
  fit <- lm(dev ~ temp)
  tibble(r2 = summary(fit)$r.squared, slope = unname(coef(fit)[2]))
}
heat <- deviation |>
  group_by(task, treatment) |>
  summarise(temp_fit(dev, gpu_temp_start_C), .groups = "drop")

heat_titles <- heat |>
  group_by(task) |>
  summarise(label = paste0(first(task), "\nR² AI ", sprintf("%.2f", r2[treatment == "AI"]),
                           ", Human ", sprintf("%.2f", r2[treatment == "Human"])))
p <- ggplot(deviation, aes(gpu_temp_start_C, dev, colour = treatment, fill = treatment)) +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.5, show.legend = FALSE) +
  geom_point(shape = 21, colour = "white", stroke = 0.3, size = 1.7) +
  facet_wrap(~task, nrow = 1, labeller = as_labeller(setNames(heat_titles$label, heat_titles$task))) +
  scale_x_continuous(breaks = few_breaks(2)) +
  scale_y_continuous(labels = signed_pct) +
  scale_colour_manual(values = COLORS, guide = "none") +
  scale_fill_manual(values = COLORS) +
  guides(fill = guide_legend(override.aes = list(size = 2.2))) +
  labs(x = "GPU temperature at the start of the run (°C)", y = "Deviation from group mean energy") +
  theme_report()

strongest <- heat |> arrange(desc(r2)) |> slice(1)
save_plot(p, "02_energy_vs_start_temp", h = 2.5, use = "threats", caption = paste0(
  "Deviation of the energy of each run from its group mean against the GPU temperature at the start ",
  "of the run, with a least-squares line per group and its R² in the panel title. ",
  "The start temperature ranged from ", min(runs$gpu_temp_start_C), " to ", max(runs$gpu_temp_start_C),
  " °C and depends on the run before. ",
  "It explains up to ", round(100 * strongest$r2), "% of the variation within a group (",
  strongest$task, " ", treatment_word(strongest$treatment), ", ", sprintf("%+.2f", strongest$slope),
  "% energy per °C)."))

carry_over <- heat |> mutate(rho_order = drift$rho, p_order = drift$p, p_order_holm = drift$p_holm)
save_table(carry_over, "02_carry_over", paste(
  "Heat carried over from the previous run and drift over the execution order, per task and treatment.",
  "$R^2$ and slope belong to a least-squares line of the energy deviation from the group mean on the",
  "GPU temperature at the start of the run. $\\rho$ is the Spearman correlation of the same deviation",
  "with the position in the execution order, with its p-value raw and Holm-adjusted over the ten groups.",
  sprintf("Every run lies within %s%% of its group mean.",
          fmt_sig(ceiling(10 * max(abs(deviation$dev))) / 10, 2))),
  tex = carry_over |>
    transmute(Task = task, Treatment = treatment, `$R^2$` = sprintf("%.2f", r2),
              `Slope (\\%/$^\\circ$C)` = sprintf("%+.2f", round(slope, 2) + 0),  # + 0 turns -0.00 into +0.00
              `$\\rho$ order` = sprintf("%.2f", rho_order),
              `p order` = fmt_p(p_order), `p order (Holm)` = fmt_p(p_order_holm)))
