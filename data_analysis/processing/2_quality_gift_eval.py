"""Step 2, GIFT-Eval: the test MASE of every GIFT-eval run.

Each run writes results/<era|ttm>_<YYYYmmdd_HHMMSS>/all_results.csv, named with the time on glg4 when it
started. A result folder belongs to the run whose EnergiBridge window holds that time.

Quality is the geometric mean over the two configurations of the test MASE divided by the MASE of
Seasonal Naive, as on the GIFT-Eval leaderboard (lower is better). Every AI run gives exactly the MASE of
the last-value naive forecaster, which equals_naive_baseline shows.

Input   data/processed/energy_runs.csv, data/raw/GIFT-eval/results/<folder>/all_results.csv
Output  data/processed/quality_GIFT-eval.csv
"""
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from common import RAW_DIR, save_quality, task_runs

TASK = "GIFT-eval"
TREATMENT_OF_PREFIX = {"era": "ai", "ttm": "human"}
CONFIGS = ["bitbrains_fast_storage/5T/short", "m4_quarterly/Q/short"]
# MASE of the GIFT-Eval leaderboard (corrected results, commit b24c41dd) on the two configurations
SEASONAL_NAIVE_MASE = {"bitbrains_fast_storage/5T/short": 1.1360241737534569,
                       "m4_quarterly/Q/short": 1.602247174616002}
NAIVE_MASE = {"bitbrains_fast_storage/5T/short": 1.0948146284753992,
              "m4_quarterly/Q/short": 1.4770341883914722}
GLG4_TZ = ZoneInfo("Europe/Amsterdam")
MARGIN_S = 2  # the folder name drops the fraction of a second


def quality_row(runs, result_csv):
    prefix, stamp = result_csv.parent.name.split("_", 1)
    started = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=GLG4_TZ).timestamp()
    match = runs[(runs.start_unix - MARGIN_S <= started) & (started <= runs.end_unix)]
    assert len(match) == 1, f"{result_csv.parent.name}: {len(match)} matching runs"
    run = match.iloc[0]
    assert run.treatment == TREATMENT_OF_PREFIX[prefix], f"{result_csv.parent.name}: run is {run.treatment}"

    mase = pd.read_csv(result_csv).set_index("dataset")["eval_metrics/MASE[0.5]"].astype(float)
    rel = {c: mase[c] / SEASONAL_NAIVE_MASE[c] for c in CONFIGS}
    return {
        "run_id": run.run_id, "task": TASK, "treatment": run.treatment, "rep": run.rep, "order": run.order,
        "quality_metric": "gmean_MASE_rel_seasonal_naive_2cfg",
        "quality": math.sqrt(rel[CONFIGS[0]] * rel[CONFIGS[1]]),
        "MASE_bitbrains_fast_storage_5T_short": mase[CONFIGS[0]],
        "MASE_m4_quarterly_Q_short": mase[CONFIGS[1]],
        "equals_naive_baseline": all(abs(mase[c] - NAIVE_MASE[c]) < 1e-9 for c in CONFIGS),
        "result_folder": result_csv.parent.name,
    }


def main():
    runs = task_runs(TASK)
    results = sorted((RAW_DIR / TASK / "results").glob("*/all_results.csv"))
    save_quality(pd.DataFrame([quality_row(runs, result_csv) for result_csv in results]), runs, TASK)


if __name__ == "__main__":
    main()
