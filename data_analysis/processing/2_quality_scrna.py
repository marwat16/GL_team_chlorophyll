"""Step 2, scRNA-seq: the process memory of every scRNA run.

The scRNA workloads print one JSON line per run with its timings and peak memory, but no quality score,
and the integrated output is not saved. So quality stays empty for this task.

The lines have no time stamp, so the last n lines of each file are paired with the n runs in execution
order. The lines above them come from test runs before the experiment.

Input   data/processed/energy_runs.csv, data/raw/scRNA/runs_ai.jsonl and runs_human.jsonl
Output  data/processed/quality_scRNA.csv
"""
import pandas as pd

from common import RAW_DIR, last_lines, save_quality, task_runs

TASK = "scRNA"
METRIC = "OpenProblems v2.0.0 overall batch-integration score (higher is better)"


def main():
    runs = task_runs(TASK)
    parts = []
    for treatment in ["ai", "human"]:
        t_runs = runs[runs.treatment == treatment].reset_index(drop=True)
        lines = last_lines(RAW_DIR / TASK / f"runs_{treatment}.jsonl", len(t_runs))
        parts.append(pd.DataFrame({
            "run_id": t_runs.run_id, "task": TASK, "treatment": treatment, "rep": t_runs.rep,
            "order": t_runs.order, "quality_metric": METRIC, "quality": float("nan"),
            "jsonl_line": lines.jsonl_line,
            "proc_rss_MiB": lines.peak_rss_mb,  # ru_maxrss / 1024, which is MiB on Linux
        }))
    save_quality(pd.concat(parts), runs, TASK)


if __name__ == "__main__":
    main()
