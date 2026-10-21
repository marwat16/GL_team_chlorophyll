# 04_rq1_energy.R
# RQ1 (Lab 3 step 4.1, Section 4.5): energy consumption (RQ1.1) and energy-to-quality ratio (RQ1.2).
# Both sub-questions form one family for the Holm-Bonferroni correction, so they stay in this script.
#
# Input:  runs (01_load.R)
# Output: tables 04_rq1_1_energy_tests and 04_rq1_2_eqr (the CSVs are also read by 08_summary.R)

if (!exists("runs")) source("analysis/01_load.R")
set.seed(SEED)


# ---- RQ1.1: energy, AI vs Human in each of the 5 tasks -------------------------------------------
rq11 <- run_tests(runs, c(energy = "energy_J"))


# ---- RQ1.2: EQR = R_E / R_Q with a bootstrap over runs (B = 10,000) ------------------------------
# R_Q > 1 means the AI result is better. For errors (lower is better) the reciprocal is used.
# Quality is the measured one, per run. Tasks without a measured score are skipped and reported.
eqr_task <- function(t) {
  d <- filter(runs, task == t)
  if (all(is.na(d$quality))) return(tibble(task = t, note = "no measured quality"))

  lower_is_better <- d$quality_lower_is_better[1]
  ai <- filter(d, treatment == "AI")
  human <- filter(d, treatment == "Human")
  r_q <- function(q_ai, q_human) {
    if (lower_is_better) mean(q_human) / mean(q_ai) else mean(q_ai) / mean(q_human)
  }
  # EQR of the AI runs i_ai and the human runs i_human (all runs, or one bootstrap resample)
  eqr <- function(i_ai, i_human) {
    r_e <- mean(ai$energy_J[i_ai]) / mean(human$energy_J[i_human])
    r_e / r_q(ai$quality[i_ai], human$quality[i_human])
  }
  boots <- replicate(B, eqr(sample(nrow(ai), replace = TRUE), sample(nrow(human), replace = TRUE)))
  # Two-sided bootstrap p-value for EQR = 1
  p <- 2 * min((1 + sum(boots <= 1)) / (B + 1), (1 + sum(boots >= 1)) / (B + 1))

  tibble(task = t, quality_metric = d$quality_metric[1], quality_ai = mean(ai$quality),
         quality_human = mean(human$quality),
         R_E = mean(ai$energy_J) / mean(human$energy_J), R_Q = r_q(ai$quality, human$quality),
         EQR = eqr(seq_len(nrow(ai)), seq_len(nrow(human))),
         EQR_lo = quantile(boots, 0.025), EQR_hi = quantile(boots, 0.975),
         p = min(p, 1))
}
rq12 <- map_dfr(TASKS, eqr_task)


# ---- Multiple comparisons ------------------------------------------------------------------------
# Holm-Bonferroni over all RQ1 tests (energy + EQR), with Benjamini-Hochberg reported next to it.
# The p-values of RQ1.1 come first, then those of the tasks with an EQR.
rq1_p <- c(rq11$p, rq12$p[!is.na(rq12$p)])
rq1_holm <- p.adjust(rq1_p, "holm")
rq1_bh <- p.adjust(rq1_p, "BH")
rq11 <- rq11 |> mutate(p_holm = rq1_holm[seq_len(n())], p_bh = rq1_bh[seq_len(n())])

eqr_tested <- which(!is.na(rq12$p))
rq12 <- rq12 |> mutate(p_holm = NA_real_, p_bh = NA_real_) |> relocate(any_of("note"), .after = last_col())
rq12$p_holm[eqr_tested] <- tail(rq1_holm, length(eqr_tested))
rq12$p_bh[eqr_tested] <- tail(rq1_bh, length(eqr_tested))


# ---- Report table RQ1.1 --------------------------------------------------------------------------
# Columns that are the same for every task go into the caption
cv_max <- runs |>
  group_by(task, treatment) |>
  summarise(cv = 100 * sd(energy_J) / mean(energy_J), .groups = "drop") |>
  pull(cv) |>
  max()
t11 <- rq11 |>
  mutate(centre_ai = centre_ai / 1000, centre_human = centre_human / 1000) |>
  tex_tests() |>
  select(-Variable) |>
  rename(`AI (kJ)` = AI, `Human (kJ)` = Human)
t11 <- squeeze_tests(t11, paste("RQ1.1, energy per run, AI versus human in each task.",
                                centre_sentence(rq11)))

d_note <- ""
if (any(grepl("Cohen", rq11$effect))) {
  d_note <- sprintf(paste(
    "Cohen's d is so large because the coefficient of variation within a group is at most %s%%,",
    "so even small differences between the means are many standard deviations wide."),
    fmt_sig(cv_max, 3))
}
# Which tests share the correction, said in the caption of each RQ1 table (`here` is the table's own)
holm_note <- function(here) {
  sprintf(paste("Holm and Benjamini-Hochberg adjust over the %d tests of RQ1,",
                "the %d energy tests%s and the %d EQR tests%s."),
          length(rq1_p), nrow(rq11), if (here == "energy") " here" else " of RQ1.1",
          length(eqr_tested), if (here == "eqr") " here" else "")
}
save_table(rq11, "04_rq1_1_energy_tests", paste(t11$caption, holm_note("energy"), d_note), tex = t11$tex,
           wide = TRUE)


# ---- Report table RQ1.2 --------------------------------------------------------------------------
eqr_metrics <- rq12 |> filter(!is.na(EQR)) |> pull(quality_metric) |> unique()
stopifnot("the caption below assumes lower-is-better quality scores" =
            all(filter(runs, quality_metric %in% eqr_metrics)$quality_lower_is_better))
no_quality <- rq12 |> filter(is.na(EQR)) |> pull(task)

t12 <- rq12 |>
  filter(!is.na(EQR)) |>
  transmute(Task = task, Quality = QUALITY_LABELS[quality_metric],
            `AI quality` = fmt_sig(quality_ai), `Human quality` = fmt_sig(quality_human),
            `$R_E$` = fmt_sig(R_E), `$R_Q$` = fmt_sig(R_Q), `EQR [95% CI]` = fmt_ci(EQR, EQR_lo, EQR_hi),
            `p (Holm)` = fmt_p(p_holm), `p (BH)` = fmt_p(p_bh))
no_quality_sentence <- if (length(no_quality)) {
  paste0(and_list(no_quality), " have no measured quality score and no EQR.")
} else {
  ""
}
save_table(rq12, "04_rq1_2_eqr", paste0(
  "RQ1.2, energy-to-quality ratio EQR = $R_E/R_Q$ with its 95% bootstrap interval. ",
  "$R_E$ is the AI mean energy divided by the human mean energy. ",
  "Quality is the mean measured score, and ", and_list(QUALITY_LONG[eqr_metrics]),
  " are errors where lower is better, ",
  "so $R_Q$ is the human mean score divided by the AI mean score and $R_Q > 1$ means ",
  "the AI result is better. ",
  "EQR $< 1$ means AI needs less energy for the quality it reaches. ",
  holm_note("eqr"), " ", no_quality_sentence),
  tex = t12, wide = TRUE)
