# Data analysis: Section 6 results

Turns the raw measurements of the 140 runs into one table, `data/processed/runs_all.csv`, and runs the R
analysis that makes every figure and table of Section 6. The energy is recomputed from the EnergiBridge
samples, because the energy column of the Experiment Runner counts the CPU cores twice and leaves out the
GPU.

Target Python version: **3.12**. R version: **4.6.1**.

## Setup
Python, from the repository root. On macOS and Linux:
```bash
cd data_analysis
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
On Windows (64-bit Python 3.12, in cmd or PowerShell):
```bat
cd data_analysis
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
R, once, in the R console or RStudio:
```r
install.packages(c("tidyverse", "bestNormalize", "ARTool", "effsize", "qqplotr", "xtable", "cowplot", "ragg"), repos = "https://cloud.r-project.org")
```

## Run
Two steps, both from `data_analysis/`. Every output is committed, so you only need them to reproduce the
results. The R step reads only the committed `data/processed/runs_all.csv`, so it also runs on its own.

1. Python, steps 1 to 3 (about 10 s). On macOS and Linux:
   ```bash
   .venv/bin/python processing/run_all.py
   ```
   On Windows:
   ```bat
   .venv\Scripts\python processing\run_all.py
   ```
   The runner data is read from `../experiment-runner/greenlab/experiments/full_experiment`. Add
   `--runner DIR` to use another copy.
2. R, the analysis (about 15 s). In a terminal:
   ```bash
   Rscript analysis/run_all.R
   ```
   Or in the R console or RStudio, with `data_analysis/` as the working directory:
   ```r
   source("analysis/run_all.R")
   ```

## Files
| Path | What it does |
| --- | --- |
| `processing/run_all.py` | Runs the three Python steps in order |
| `processing/1_energy_from_runner.py` | Energy, time, CPU, GPU and memory of every run from its `energibridge.csv`, into `data/processed/energy_runs.csv` |
| `processing/2_quality_<task>.py` | The quality score of every run, matched to its run, into `data/processed/quality_<task>.csv` |
| `processing/3_build_runs_all.py` | Joins the two into `data/processed/runs_all.csv`, the only file R reads |
| `processing/common.py` | Paths and small helpers of the Python steps |
| `data/raw/` | The per-run outputs the task scripts wrote on glg4 (JSON lines for Covid19 and scRNA, `all_results.csv` for GIFT-eval, `result.json` for ZapBench) |
| `analysis/run_all.R` | Runs the R scripts 01 to 08 in order |
| `analysis/00_common.R` | Packages, plot style and the helpers every step uses |
| `analysis/01_load.R` | Loads `runs_all.csv` and prints an overview |
| `analysis/02_explore.R` | Energy per task, and checks for drift and for heat left from the previous run |
| `analysis/03_normality.R` | Shapiro-Wilk tests, Q-Q plots and energy density plots |
| `analysis/04_rq1_energy.R` | RQ1, energy tests and the energy-to-quality ratio (EQR) |
| `analysis/05_rq2_performance.R` | RQ2, tests on time, CPU, GPU and memory |
| `analysis/06_rq3_correlations.R` | RQ3, correlation of energy with each performance metric |
| `analysis/07_models.R` | One model over all tasks, as an alternative to the per-task tests |
| `analysis/08_summary.R` | Forest plot of the AI/human ratios and the descriptive tables |
| `output/figures/` | Figures as PDF and PNG, a name ending in `_col` is the one-column version of a figure. `captions.csv` gives each caption and where it is used |
| `output/tables/` | Tables as CSV and LaTeX |

## Playing with the data
Everything is CSV, so Python works fine for looking around:
```python
import pandas as pd

runs = pd.read_csv("data/processed/runs_all.csv")
print(runs.groupby(["task", "treatment"]).energy_J.median().unstack())
```
The statistical tests themselves are only in R.

## Notes
- `quality` is empty for scRNA, which logs no score, and for Geospatial, whose scores were printed to the
  terminal and not kept.
- `runs_all.csv` uses the task keys (`scRNA`, `Covid19`, ...) and `ai`/`human`. The tables in `output/`
  use the report names (`scRNA-seq`, `COVID-19`, ...) and `AI`/`Human`.
- The committed figures were made on macOS with Helvetica. On Windows and Linux R uses a similar font, so
  rebuilt figures look slightly different, but every number is the same.
- On Windows the R installer does not put `Rscript` on the PATH. Use the R console way above, or the full
  path `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`.
- On Debian and Ubuntu, `python3.12 -m venv` needs `sudo apt install python3.12-venv` first.
- On Linux, R builds the packages from source, which needs system libraries. On Debian and Ubuntu:
  `sudo apt install r-base-dev cmake libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev
  libfreetype-dev libharfbuzz-dev libfribidi-dev libpng-dev libtiff-dev libjpeg-dev libwebp-dev libfftw3-dev`.
