"""Step 2, Covid19: the WIS of every Covid19 run.

AI.sh and Human.sh append one JSON line per run to runs_ai.jsonl and runs_human.jsonl. The lines have no
time stamp, so the last n lines of each file are paired with the n runs in execution order. The lines
above them come from test runs before the experiment.

Quality is the WIS of the forecast for the reference date 2025-02-01 (lower is better). status is
check_failed when the forecast table is incomplete or broken.

Input   data/processed/energy_runs.csv, data/raw/Covid19/runs_ai.jsonl and runs_human.jsonl
Output  data/processed/quality_Covid19.csv
"""
import pandas as pd

from common import RAW_DIR, last_lines, save_quality, task_runs

TASK = "Covid19"
REF_DATE = "2025-02-01"


def forecast_ok(line, treatment):
    # 52 locations x 4 horizons = 208 rows, each with the 23 quantiles the forecast hub asks for
    return (line.treatment == treatment and line.ref_date == REF_DATE and line.n_rows == 208
            and line.n_quantiles == 23 and not line.any_nan and line.monotone)


def main():
    runs = task_runs(TASK)
    parts = []
    for treatment in ["ai", "human"]:
        t_runs = runs[runs.treatment == treatment].reset_index(drop=True)
        lines = last_lines(RAW_DIR / TASK / f"runs_{treatment}.jsonl", len(t_runs))
        parts.append(pd.DataFrame({
            "run_id": t_runs.run_id, "task": TASK, "treatment": treatment, "rep": t_runs.rep,
            "order": t_runs.order,
            "status": ["ok" if forecast_ok(line, treatment) else "check_failed"
                       for line in lines.itertuples()],
            "quality_metric": f"wis_era_{REF_DATE}", "quality": lines.wis_check,
            "jsonl_line": lines.jsonl_line,
            "proc_rss_MiB": lines.peak_rss_mb,  # ru_maxrss / 1024, which is MiB on Linux
        }))
    save_quality(pd.concat(parts), runs, TASK)


if __name__ == "__main__":
    main()
