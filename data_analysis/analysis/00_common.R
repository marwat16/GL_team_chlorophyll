# 00_common.R
# Shared setup of the Section 6 analysis: packages, constants, the figure style and the helpers
# that every step uses (number formatting, saving figures and tables, the test rule of Section 4.5
# and the per-task figure). It is sourced through 01_load.R, so the test rules are defined in one
# place only.
#
# Input:  none (checks that data/processed/runs_all.csv exists)
# Output: creates output/figures and output/tables

suppressPackageStartupMessages({
  library(tidyverse)      # dplyr, ggplot2, readr, tidyr, purrr, tibble, stringr
  library(bestNormalize)  # Lab 3 step 3.2
  library(ARTool)         # Lab 3 step 4.2
  library(effsize)        # Cohen's d and Cliff's delta
  library(qqplotr)        # Q-Q plots with confidence bands (lecture 07-A)
  library(xtable)         # LaTeX tables (Lab 3 step 6.2)
  library(cowplot)        # stacks the four rows of the RQ2 figure
})
# Rscript starts in the C locale on macOS, where symbols such as the minus sign or rho print as dots.
# Some Linux systems only have C.UTF-8. R on Windows already runs in UTF-8.
for (loc in c("en_US.UTF-8", "C.UTF-8")) {
  if (nzchar(suppressWarnings(Sys.setlocale("LC_CTYPE", loc)))) break
}
if (!l10n_info()[["UTF-8"]]) {
  stop("the analysis needs a UTF-8 locale, for example LANG=C.UTF-8", call. = FALSE)
}
if (!interactive()) pdf(NULL)  # stray plot calls go nowhere instead of into Rplots.pdf
if (!file.exists("data/processed/runs_all.csv")) {
  stop("data/processed/runs_all.csv not found. Run the R scripts from the package folder (the one with ",
       "README.md), after python processing/run_all.py", call. = FALSE)
}
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)


# ---- Constants -----------------------------------------------------------------------------------
# Task names in the data files and the names the report uses
TASK_KEYS <- c("scRNA", "Covid19", "Geospatial", "GIFT-eval", "ZapBench")
TASKS <- c("scRNA-seq", "COVID-19", "Geospatial", "GIFT-Eval", "ZAPBench")

# Short names of the quality metrics for tables, and how a caption describes them
QUALITY_LABELS <- c(`wis_era_2025-02-01` = "WIS",
                    gmean_MASE_rel_seasonal_naive_2cfg = "MASE",
                    test_mae_avg8 = "test MAE",
                    test_mIoU_420 = "test mIoU")
QUALITY_LONG <- c(`wis_era_2025-02-01` = "WIS",
                  gmean_MASE_rel_seasonal_naive_2cfg = "MASE relative to a seasonal naive forecast",
                  test_mae_avg8 = "test MAE",
                  test_mIoU_420 = "test mIoU")

ALPHA <- 0.05
B <- 10000      # bootstrap resamples
SEED <- 2026
# Memory column: peak minus start of the run, whole machine (memory already in use before the run
# is left out)
MEM_COLUMN <- "mem_delta_MiB"
SAMPLE_S <- 0.2  # EnergiBridge sampling interval in seconds
THREADS <- 16    # hardware threads of the Intel Core i9-11900KF (lscpu), CPU usage is their mean


# ---- Figure style --------------------------------------------------------------------------------
# Figures are drawn at the size they are printed in the ACM two-column layout (full width 7 in,
# one column 3.33 in), so 8 pt text in R is 8 pt in the PDF. One colour per treatment in every
# figure, a colour-blind safe pair (checked for protanopia, deuteranopia and tritanopia).
COLORS <- c(AI = "#2a78d6", Human = "#eb6834")
INK <- "#0b0b0b"
INK2 <- "#52514e"
MUTED <- "#898781"
GRID <- "#e1e0d9"
AXIS <- "#c3c2b7"
FULL_W <- 7
TIMES <- "\u00d7"
MINUS_SIGN <- "\u2212"
minus_labels <- scales::label_number(style_negative = "minus", big.mark = ",")  # real minus on axes

theme_report <- function(base_size = 8) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(text = element_text(colour = INK),
          axis.text = element_text(colour = INK2, size = rel(0.9)),
          axis.title = element_text(colour = INK2, size = rel(0.95)),
          axis.title.y = element_text(colour = INK2, size = rel(0.95), angle = 90, margin = margin(r = 4)),
          axis.title.x = element_text(colour = INK2, size = rel(0.95), margin = margin(t = 4)),
          axis.line.x = element_line(colour = AXIS, linewidth = 0.3),
          axis.ticks = element_blank(),
          panel.grid.major = element_line(colour = GRID, linewidth = 0.25),
          panel.grid.minor = element_blank(),
          panel.spacing = unit(9, "pt"),
          strip.text = element_text(colour = INK, face = "bold", hjust = 0, size = rel(0.95),
                                    lineheight = 1.1, margin = margin(b = 4)),
          legend.position = "top",
          legend.justification = "left",
          legend.title = element_blank(),
          legend.margin = margin(0, 0, 0, 0),
          legend.box.spacing = unit(4, "pt"),
          legend.key.size = unit(9, "pt"),
          legend.text = element_text(colour = INK2, size = rel(0.9)),
          plot.background = element_rect(fill = "white", colour = NA),
          plot.margin = margin(4, 8, 4, 4))
}
# The ten-panel figures (one panel per task and treatment) use a slightly smaller panel title
THEME_GROUPS <- theme(strip.text = element_text(size = rel(0.85)))

# PDF device with embedded fonts. The default pdf() device does not embed Helvetica (ACM checks for
# that) and prints the hyphen as a minus sign. Quartz exists on macOS, cairo on most Linux installs.
pdf_device <- if (capabilities("aqua")) {
  function(filename, ...) {
    grDevices::quartz(type = "pdf", file = filename, title = tools::file_path_sans_ext(basename(filename)),
                      ...)
  }
} else if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

# Every figure goes out as a vector PDF for Overleaf and a 300 dpi PNG for slides and quick viewing.
# output/figures/captions.csv collects a draft caption per figure and where it is meant to go
# (report, threats, package = replication package only).
save_plot <- function(p, name, w = FULL_W, h = 2.6, caption = NA_character_, use = "package") {
  ggsave(file.path("output/figures", paste0(name, ".pdf")), p, width = w, height = h,
         device = pdf_device)
  ggsave(file.path("output/figures", paste0(name, ".png")), p, width = w, height = h, dpi = 300,
         device = ragg::agg_png)

  # Replace the row of this figure in captions.csv, sorted by figure name
  captions_file <- "output/figures/captions.csv"
  captions <- if (file.exists(captions_file)) {
    read_csv(captions_file, show_col_types = FALSE, col_types = cols(.default = "c"))
  } else {
    tibble(figure = character(), use = character(), caption = character())
  }
  captions |>
    filter(figure != name) |>
    bind_rows(tibble(figure = name, use = use, caption = str_squish(caption))) |>
    arrange(figure) |>
    write_csv(captions_file)
}


# ---- Number formatting for labels, captions and LaTeX tables -------------------------------------
# d significant digits with a thousands separator, "" for NA
fmt_sig <- function(x, d = 3) {
  out <- formatC(signif(x, d), format = "fg", digits = d, big.mark = ",", flag = "#")
  ifelse(is.na(x), "", sub("\\.$", "", trimws(out)))
}

# d decimals (one value of d per element, or one for all), "" for NA
fmt_dec <- function(x, d) {
  out <- map2_chr(x, rep_len(d, length(x)),
                  ~ formatC(.x, format = "f", digits = .y, big.mark = ","))
  ifelse(is.na(x), "", out)
}

fmt_p <- function(p) ifelse(is.na(p), "", ifelse(p < 0.001, "< 0.001", sprintf("%.3f", p)))

# "estimate [lower, upper]"
fmt_ci <- function(est, lo, hi, d = 3) {
  ifelse(is.na(est), "", sprintf("%s [%s, %s]", fmt_sig(est, d), fmt_sig(lo, d), fmt_sig(hi, d)))
}
fmt_r <- function(r, lo, hi) ifelse(is.na(r), "", sprintf("%.2f [%.2f, %.2f]", r, lo, hi))

# Treatment as it is written in running text, "AI" and "human"
treatment_word <- function(x) ifelse(x == "Human", "human", as.character(x))

# "a", "a and b", "a, b and c"
and_list <- function(x) {
  if (length(x) <= 1) return(x)
  paste(paste(head(x, -1), collapse = ", "), "and", tail(x, 1))
}

# Axis labels with the fewest decimals that still show every break exactly, the same for all breaks
num_lab <- function(x) {
  ok <- x[is.finite(x)]
  d <- 0
  while (d < 4 && any(abs(round(ok, d) - ok) > 1e-9 * pmax(1, abs(ok)))) d <- d + 1
  out <- formatC(x, format = "f", digits = d, big.mark = ",")
  sub("^-", MINUS_SIGN, out)
}

# At most three round breaks inside the axis range, for narrow panels with free scales
few_breaks <- function(n = 2) function(lims) {
  for (k in n:(n + 4)) {
    b <- pretty(lims, n = k)
    b <- b[b >= lims[1] & b <= lims[2]]
    if (length(b) >= 2) break
  }
  if (length(b) > 3) b <- b[seq(1, length(b), by = 2)]
  b
}


# ---- Saving tables -------------------------------------------------------------------------------
# Writes a table as CSV with full precision (for the text and for other scripts) and as LaTeX.
# `tex` is the readable version a script builds itself, otherwise an automatic one is made with
# p-values as "< 0.001", whole numbers without decimals and other numbers to three significant digits.
# The tables use booktabs, so the Overleaf preamble needs \usepackage{booktabs}.
save_table <- function(df, name, caption, tex = NULL, wide = FALSE) {
  write_csv(df, file.path("output/tables", paste0(name, ".csv")))

  auto <- is.null(tex)
  if (auto) {
    whole <- function(x) is.numeric(x) && all(x == round(x), na.rm = TRUE)
    tex <- df |>
      mutate(across(where(is.numeric) & matches("^p$|^p_|_p$|_p_|Pr\\("), fmt_p),
             across(where(whole), ~ ifelse(is.na(.x), "", formatC(.x, format = "d", big.mark = ","))),
             across(where(is.numeric), fmt_sig))
  }
  # Columns of numbers are right-aligned
  numeric_like <- function(x) all(grepl("^[][0-9.,<%() +\u2212\u2013\u2020-]*$", x))
  align <- c("l", ifelse(map_lgl(tex, numeric_like), "r", "l"))

  # Cells are escaped for LaTeX, a minus in front of a number becomes a real minus and a dagger is kept
  text_san <- function(x) {
    x <- sanitize(x, type = "latex")
    x <- gsub("(^|[ \\[(])(-|\u2212)(?=[0-9])", "\\1$-$", x, perl = TRUE)
    gsub("\u2020", "$^\\\\dagger$", x)
  }
  # Hand-made headers may contain LaTeX such as $R_E$, so only the percent sign is escaped there
  head_san <- if (auto) {
    function(x) sanitize(x, type = "latex")
  } else {
    function(x) gsub("(?<!\\\\)%", "\\\\%", x, perl = TRUE)
  }
  # In the caption a percent sign is escaped and a "<" outside math mode is set in math mode
  caption <- gsub("(?<!\\\\)%", "\\\\%", caption, perl = TRUE)
  caption <- gsub("(?<![$])<(?= )", "$<$", caption, perl = TRUE)

  # write_file keeps LF line ends on Windows, where print(file = ) would write CRLF
  latex <- print(xtable(tex, caption = caption, label = paste0("tab:", name), align = align),
                print.results = FALSE, include.rownames = FALSE, booktabs = TRUE, comment = FALSE,
                NA.string = "", caption.placement = "top", size = "\\small",
                floating.environment = if (wide) "table*" else "table",
                sanitize.text.function = text_san, sanitize.colnames.function = head_san)
  write_file(latex, file.path("output/tables", paste0(name, ".tex")))
}

# Columns of a report table that hold the same value in every row are moved into the caption as a
# sentence, which keeps the tables narrow enough for the two-column layout
squeeze <- function(tex, caption, sentences) {
  for (col in intersect(names(sentences), names(tex))) {
    v <- unique(tex[[col]])
    if (length(v) == 1) {
      caption <- paste(caption, sprintf(sentences[[col]], v))
      tex[[col]] <- NULL
    }
  }
  list(tex = tex, caption = caption)
}


# ---- The test rule of Section 4.5 ----------------------------------------------------------------
# Shapiro-Wilk p-value, NA when a group has fewer than three distinct values (for example the GPU
# usage of a CPU-only task, which is 0 in every run)
sw_p <- function(x) {
  x <- x[is.finite(x)]
  if (length(unique(x)) < 3) return(NA_real_)
  shapiro.test(x)$p.value
}

# AI vs Human for one variable of one task, independent samples. Decision rule of Section 4.5:
#   both groups normal on the raw scale -> Welch t-test, Cohen's d, ratio of medians
#   else both normal on the log scale   -> Welch t-test on log values, Cohen's d on log,
#                                          ratio of geometric means
#   otherwise                           -> Mann-Whitney U (wilcox.test), Cliff's delta, ratio of medians
# The AI/Human ratio gets a bootstrap 95 % interval (B resamples) so that every comparison can go in
# the forest plot. centre_ai and centre_human are the two values the ratio is built from.
compare <- function(ai, human) {
  if (sd(ai) == 0 && sd(human) == 0) return(tibble(test = "constant, not tested"))

  raw_normal <- isTRUE(sw_p(ai) > ALPHA && sw_p(human) > ALPHA)
  log_normal <- all(ai > 0, human > 0) && isTRUE(sw_p(log(ai)) > ALPHA && sw_p(log(human)) > ALPHA)
  if (raw_normal) {
    tt <- t.test(ai, human)
    eff <- cohen.d(ai, human)
    res <- tibble(test = "Welch's t-test", p = tt$p.value, effect = "Cohen's d", ratio_of = "medians")
  } else if (log_normal) {
    tt <- t.test(log(ai), log(human))
    eff <- cohen.d(log(ai), log(human))
    res <- tibble(test = "Welch's t-test on log values", p = tt$p.value, effect = "Cohen's d (log)",
                  ratio_of = "geometric means")
  } else {
    wt <- suppressWarnings(wilcox.test(ai, human))
    eff <- cliff.delta(ai, human)
    res <- tibble(test = "Mann-Whitney U test", p = wt$p.value, effect = "Cliff's delta",
                  ratio_of = "medians")
  }

  # Each bootstrap resample draws the AI runs first and then the human runs
  centre <- if (res$ratio_of == "geometric means") function(x) exp(mean(log(x))) else median
  boot <- replicate(B, centre(sample(ai, replace = TRUE)) / centre(sample(human, replace = TRUE)))
  res |>
    mutate(es = unname(eff$estimate), es_lo = eff$conf.int[[1]], es_hi = eff$conf.int[[2]],
           magnitude = as.character(eff$magnitude),
           centre_ai = centre(ai), centre_human = centre(human), ratio = centre(ai) / centre(human),
           ratio_lo = quantile(boot, 0.025), ratio_hi = quantile(boot, 0.975))
}

# compare() for every named variable (name = label, value = column) in every task that tasks_for
# gives. Variables are the outer loop, so the bootstrap draws always come in the same order.
run_tests <- function(runs, vars, tasks_for = function(v) TASKS) {
  test_one <- function(v, t) {
    d <- filter(runs, task == t)
    ai <- d[[vars[[v]]]][d$treatment == "AI"]
    human <- d[[vars[[v]]]][d$treatment == "Human"]
    tibble(task = t, variable = v, n_ai = length(ai), n_human = length(human),
           mean_ai = mean(ai), mean_human = mean(human),
           median_ai = median(ai), median_human = median(human)) |>
      bind_cols(compare(ai, human))
  }
  map_dfr(names(vars), function(v) map_dfr(tasks_for(v), function(t) test_one(v, t)))
}

# The report version of a test table. AI and Human are the two values the ratio is built from, so
# the columns divide to the ratio and match the labels of the figures.
tex_tests <- function(df, digits = 3) {
  df |>
    transmute(Task = task, Variable = variable,
              AI = fmt_sig(centre_ai, digits), Human = fmt_sig(centre_human, digits),
              `AI/Human [95% CI]` = fmt_ci(ratio, ratio_lo, ratio_hi), Test = test,
              `p (Holm)` = fmt_p(p_holm), `p (BH)` = fmt_p(p_bh), `Effect size` = effect,
              `Estimate [95% CI]` = fmt_ci(es, es_lo, es_hi), Magnitude = magnitude)
}

# Caption sentences for the columns of tex_tests() that are the same in every row
TEST_SENTENCES <- c(Test = "Every comparison uses %s.",
                    `p (Holm)` = "Every Holm-adjusted p-value is %s.",
                    `p (BH)` = "Every Benjamini-Hochberg adjusted p-value is %s.",
                    `Effect size` = "The effect size is %s.",
                    Magnitude = "Every effect is %s.")

# squeeze() for a tex_tests() table. When every row has the same effect size, its name goes in the
# header.
squeeze_tests <- function(tex, caption) {
  eff <- unique(tex$`Effect size`)
  out <- squeeze(tex, caption, TEST_SENTENCES)
  if (length(eff) == 1) {
    names(out$tex)[names(out$tex) == "Estimate [95% CI]"] <- paste(eff, "[95% CI]")
  }
  out
}

# What the AI and Human columns hold, depending on which tests were used
centre_sentence <- function(df) {
  if (any(df$ratio_of == "geometric means", na.rm = TRUE)) {
    paste("AI and Human give the median per run, or the geometric mean where the test ran on log",
          "values, and AI/Human is their ratio with a 95% bootstrap interval.")
  } else {
    "AI and Human give the median per run and AI/Human is their ratio with a 95% bootstrap interval."
  }
}


# ---- Shared figure: one variable per task, AI vs Human -------------------------------------------
# One panel per task with its own y-axis from 0, so heights are honest within a task and tasks of
# very different size stay readable. Tasks are blocks and are not compared with each other. The bar
# is the median and the points are the single runs. The runs of a group lie close together
# (coefficient of variation at most 2.5 %, 9 % for memory), so a box would hardly show and the spread
# is given in the tables and in the Q-Q and density figures. The panel title says in words how the two
# medians relate. `phrase` gets the treatment with the higher median and the factor, for example
# "%s %s more" gives "AI 2.73x more".
# show_task = FALSE leaves the task name out of the panel title (the lower rows of a stacked figure).
# Tasks without runs in `data` keep an empty panel titled `empty_title` with `empty_note` inside, and
# `ref_line` draws a dashed horizontal reference line.
plot_by_task <- function(data, var, ylab, unit = "", percent = FALSE, phrase = "%s %s more",
                         zero = "AI uses none", show_task = TRUE, empty_title = "", empty_note = NULL,
                         ref_line = NULL) {
  d <- mutate(data, y = .data[[var]])
  med <- d |>
    group_by(task, treatment) |>
    summarise(m = median(y), .groups = "drop")

  # Panel titles: the task, then which treatment has the higher median and by what factor
  wide <- pivot_wider(med, names_from = treatment, values_from = m)
  ratio <- wide$AI / wide$Human
  phrases <- case_when(wide$AI == 0 ~ zero,
                       ratio >= 1 ~ sprintf(phrase, "AI", paste0(fmt_sig(ratio, 3), TIMES)),
                       TRUE ~ sprintf(phrase, "Human", paste0(fmt_sig(1 / ratio, 3), TIMES)))
  phrases <- setNames(phrases, wide$task)
  empty <- setdiff(levels(d$task), names(phrases))
  phrases[empty] <- empty_title
  titles <- if (show_task) setNames(paste0(names(phrases), "\n", phrases), names(phrases)) else phrases

  # The median label sits above the highest run of its group
  top <- d |>
    group_by(task, treatment) |>
    summarise(top = max(y), .groups = "drop")
  med_labels <- med |>
    left_join(top, by = c("task", "treatment")) |>
    mutate(label = if (unit == "%") paste0(fmt_sig(m), "%") else trimws(paste(fmt_sig(m), unit)))

  notes <- tibble(task = factor(empty, levels = levels(d$task)), label = empty_note %||% "")

  ggplot(d, aes(treatment, y)) +
    geom_col(data = med, aes(y = m, fill = treatment), width = 0.6, alpha = 0.22) +
    (if (!is.null(ref_line)) geom_hline(yintercept = ref_line, colour = MUTED, linewidth = 0.35,
                                        linetype = "22")) +
    geom_point(aes(fill = treatment), shape = 21, colour = "white", stroke = 0.3, size = 1.6,
               position = position_jitter(width = 0.14, height = 0, seed = SEED)) +
    geom_label(data = med_labels, aes(y = top, label = label), vjust = -0.45, size = 2.5,
               colour = INK2, fill = "white", border.colour = NA, label.padding = unit(0.8, "pt"),
               label.r = unit(0, "pt")) +
    geom_text(data = notes, aes(x = 1.5, y = if (percent) 50 else 0, label = label), size = 2.2,
              colour = MUTED, fontface = "italic", inherit.aes = FALSE) +
    facet_wrap(~task, nrow = 1, scales = if (percent) "fixed" else "free_y", drop = FALSE,
               labeller = as_labeller(titles)) +
    scale_y_continuous(expand = expansion(mult = c(0, if (percent) 0.12 else 0.16)),
                       labels = scales::label_comma(),
                       breaks = if (percent) seq(0, 100, 25) else scales::breaks_extended(n = 4)) +
    expand_limits(y = if (percent) c(0, 100) else 0) +
    coord_cartesian(clip = "off") +
    scale_fill_manual(values = COLORS, guide = "none") +
    labs(x = NULL, y = ylab) +
    theme_report() +
    theme(panel.grid.major.x = element_blank(), strip.text = element_text(size = rel(0.9)))
}

# Shared caption sentence for these figures
BY_TASK_NOTE <- paste(
  "The bar and its label give the median and each point is one run. Every task has its own y-axis",
  "starting at 0, and the panel title says which treatment has the higher median and by what factor.")

# Panel label "task, treatment" with AI panels first, used by the ten-panel figures (density and
# scatter). All ten levels are kept so that AI and Human of a task stay in the same column.
GROUPS <- c(paste0(TASKS, ", AI"), paste0(TASKS, ", Human"))
group_panels <- function(data) {
  mutate(data, panel = factor(paste0(task, ", ", treatment), levels = GROUPS))
}
