"""Step 2, ZAPBench: the test MAE of every ZapBench run.

The ZAPBench harness (run_era.py / run_tsmixer.py) writes runs/<era|tsmixer>/run_<k>/result.json with the
time the script started (started_at), its status, seed and test MAE. A result.json belongs to the run
whose EnergiBridge window holds its started_at.

Quality is the test MAE averaged over the 8 stimulus conditions that ZAPBench averages officially (all
except taxis). Lower is better.

Input   data/processed/energy_runs.csv, data/raw/ZapBench/runs/<impl>/run_<k>/result.json
Output  data/processed/quality_ZapBench.csv
"""
import json

import pandas as pd

from common import RAW_DIR, save_quality, task_runs

TASK = "ZapBench"
# the script can stamp its start a little before the first EnergiBridge sample. Runs are more than 60 s
# apart, so this margin cannot pick a neighbouring run.
MARGIN_S = 5


def quality_row(runs, result_json, runs_dir):
    res = json.loads(result_json.read_text())
    started = res["started_at"]
    match = runs[(runs.start_unix - MARGIN_S <= started) & (started <= runs.end_unix)]
    assert len(match) == 1, f"{result_json}: {len(match)} matching runs"
    run = match.iloc[0]
    treatment = "ai" if res["impl"] == "era" else "human"
    assert treatment == run.treatment, f"{result_json}: run is {run.treatment}"
    return {"run_id": run.run_id, "task": TASK, "treatment": treatment, "rep": run.rep, "order": run.order,
            "status": res["status"], "seed": res["seed"], "quality_metric": "test_mae_avg8",
            "quality": res["test_mae"]["avg_8"], "result_json": result_json.relative_to(runs_dir).as_posix()}


def main():
    runs = task_runs(TASK)
    runs_dir = RAW_DIR / TASK / "runs"
    results = sorted(runs_dir.glob("*/run_*/result.json"))
    rows = [quality_row(runs, result_json, runs_dir) for result_json in results]
    save_quality(pd.DataFrame(rows), runs, TASK)


if __name__ == "__main__":
    main()
