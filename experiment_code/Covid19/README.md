# COVID-19 forecasting, local test of the two treatments

This folder runs the AI treatment (ERA node 1937) and the human treatment
(UMass-gbqr) on the same input, on your own machine. It only checks that both
codes run and gives rough timings. Energy measurement (Experiment Runner +
EnergiBridge) comes later and will call `run_one.py` once per run.

## Files

| File | Purpose |
| --- | --- |
| `ai_node1937.py` | AI treatment, copied verbatim from ERA study 172492931, node 1937 (nodes 1960 and 1975 have identical code and the same score) |
| `human_gbqr.py` | Human treatment, `GBQRModel` from reichlab/idmodels with the COVID settings of reichlab/operational-models. Only the data download is replaced, so it receives the same data as the AI |
| `harness.py` | Fold construction and WIS, copied from the ERA forecasting notebook |
| `prepare_input.py` | Builds `dataset.csv` and `locations.csv` from CovidHub data (as of 30 April 2025), or synthetic data |
| `run_one.py` | Runs one treatment once for one reference date and prints timings and checks |
| `smoke_test.sh` | Runs both treatments one after the other |
| `requirements.txt` | Pinned library versions (fixed factor of the experiment) |

## 1. Environment

Linux or macOS (on Windows use WSL), Python 3.12, git, about 2 GB of free RAM.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Quick check without downloading anything:

```bash
python prepare_input.py --synthetic
```

Real data. Clone only the data folders of the CovidHub repository, then build the input:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/CDCgov/covid19-forecast-hub.git
git -C covid19-forecast-hub sparse-checkout set target-data auxiliary-data
python prepare_input.py --hub-dir covid19-forecast-hub
```

This takes the NHSN COVID-19 hospital admissions as they were available on
30 April 2025 (the last snapshot before the ERA cut-off of 1 May 2025), keeps
the 52 jurisdictions ERA forecasts, replaces missing values by zero and adds the
CovidHub populations. Expected output is 12,844 rows, 52 locations and 247 weeks.

## 3. Run

```bash
./smoke_test.sh                                  # both treatments, human with 2 bags (quick)
NUM_BAGS=100 ./smoke_test.sh                     # real configuration (slow)
python run_one.py --treatment ai                 # one treatment
python run_one.py --treatment human --num-bags 2 # quick check (the real value is 100)
python run_one.py --treatment human --load-only  # loading + imports only
```

Each run prints one JSON line. Example on real data, one CPU core:

```
{"treatment": "ai", "load_s": 0.85, "method_s": 22.2, "peak_rss_mb": 226, "n_rows": 208, "n_quantiles": 23, "any_nan": false, "monotone": true, "wis_check": 38.21}
```

The human treatment with all 100 bags took an estimated 31 minutes on one core
(about 19 seconds per bag). With several cores it is much faster, because
LightGBM uses all available threads. Fix the number of threads with
`OMP_NUM_THREADS` so that every run uses the same value.

## Notes

- One run is one forecast for one reference date, 52 locations x 4 horizons x 23 quantiles.
- `--ai-config` selects the entry of `config_list` in node 1937 (default 0, the first).
  The COVID notebook that decides how ERA used the two configurations is not released.
- `wis_check` is computed after the measured section, only to confirm the forecast is
  sensible. Quality scores in the report come from the released ERA data.
- Do not change `ai_node1937.py` or `human_gbqr.py` between runs.
- If you see `allow_write_nullable_strings` or similar errors, your pandas is 3.x.
  idmodels requires pandas 2.
