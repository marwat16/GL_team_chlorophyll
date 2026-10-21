# 07_models.R
# Lab 3 steps 4.2-4.4: one model over all tasks, as an alternative to the per-task tests.
# The runs are independent executions, so there is no repeated-measures 'trial' random effect as in
# the lab, and lm is used instead of lmer. Replication package only. The per-task tests of 04 stay
# the main analysis, because both checks below fail on this data.
#
# Input:  runs (01_load.R)
# Output: tables 07_art_anova_energy and 07_lm_anova_log_energy

if (!exists("runs")) source("analysis/01_load.R")


# ---- Aligned rank transform ANOVA (Lab 3 step 4.2) -----------------------------------------------
# On treatment, task and their interaction. The check of the alignment: the ANOVA of each aligned
# response on the effects it was not aligned for should give F of about 0.
art_model <- art(energy_J ~ treatment * task, data = runs)
art_check <- summary(art_model)$aligned.anova
max_f <- max(art_check[["F value"]], na.rm = TRUE)
cat("ART alignment check: each aligned response against the effects it was not aligned for,",
    "F should be about 0\n")
print(art_check)

save_table(as_tibble(anova(art_model)), "07_art_anova_energy", paste(
  "Aligned rank transform ANOVA of energy on treatment, task and their interaction.",
  sprintf(paste("The alignment check fails, the F values of the effects that should be removed by the",
                "alignment reach %s instead of about 0,"),
          fmt_sig(max_f, 3)),
  "so this model is not used for the conclusions."))


# ---- Linear model of log energy (Lab 3 step 4.3) with a residual check (step 4.4) ----------------
lm_model <- lm(log(energy_J) ~ treatment * task, data = runs)
lm_anova <- anova(lm_model)
cat("ANOVA of log energy on treatment, task and their interaction\n")
print(lm_anova)

group_id <- interaction(runs$task, runs$treatment)
sw_res <- shapiro.test(resid(lm_model))$p.value
fligner <- fligner.test(resid(lm_model) ~ group_id)
res_sd <- tapply(resid(lm_model), group_id, sd)
fligner_p <- ifelse(fligner$p.value < 0.001, "< 0.001", paste("=", fmt_p(fligner$p.value)))
save_table(as_tibble(lm_anova, rownames = "Term"), "07_lm_anova_log_energy", paste(
  "ANOVA of log energy on treatment, task and their interaction.",
  sprintf(paste("The residuals are close to normal (Shapiro-Wilk p = %s) but their spread differs",
                "between the groups"),
          fmt_p(sw_res)),
  sprintf(paste("(Fligner-Killeen p %s, residual SD from %s to %s on the log scale), so the equal-variance",
                "assumption does not hold."),
          fligner_p, fmt_sig(min(res_sd), 2), fmt_sig(max(res_sd), 2))))
cat("Residuals of the log-energy model: Shapiro-Wilk p =", signif(sw_res, 3),
    "(normal), Fligner-Killeen p =", signif(fligner$p.value, 3), "(unequal variances)\n")
