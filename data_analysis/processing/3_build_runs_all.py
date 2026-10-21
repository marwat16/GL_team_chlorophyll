"""Step 3: one table for the R analysis, with the energy, performance and quality of every run.

Geospatial has no quality file. Its scripts printed the test mIoU only to the terminal, and that output
was not kept, so its quality stays empty, as for scRNA.

mem_delta_MiB is the memory peak of the whole machine minus its value at the start of the run.
proc_rss_MiB is the peak of the workload process, which only Covid19 and scRNA logged.

Input   data/processed/energy_runs.csv and data/processed/quality_<task>.csv
Output  data/processed/runs_all.csv, one row per run, sorted by task, treatment and rep
"""
import pandas as pd

from common import ENERGY_CSV, PROCESSED_DIR, write_csv

TASKS = ["scRNA", "Covid19", "Geospatial", "GIFT-eval", "ZapBench"]
LOWER_IS_BETTER = {"Covid19": True, "GIFT-eval": True, "ZapBench": True, "scRNA": False, "Geospatial": False}
QUALITY_COLUMNS = ["run_id", "quality_metric", "quality", "quality_lower_is_better", "proc_rss_MiB"]
ENERGY_COLUMNS = ["task", "treatment", "rep", "run_id", "order", "start_unix", "energy_J", "cpu_J", "dram_J",
                  "gpu_J", "gpu_idle_W", "eb_seconds", "cpu_pct_mean", "gpu_pct_mean", "mem_peak_MiB",
                  "mem_delta_MiB", "gpu_mem_peak_MiB", "gpu_temp_start_C"]


def read_quality(energy):
    parts = []
    for task in TASKS:
        if task == "Geospatial":
            # the planned score, the mIoU on the 420 test images
            q = energy.loc[energy.task == task, ["run_id"]].assign(quality_metric="test_mIoU_420",
                                                                    quality=float("nan"))
        else:
            q = pd.read_csv(PROCESSED_DIR / f"quality_{task}.csv")
        q = q.assign(proc_rss_MiB=q.get("proc_rss_MiB", pd.NA),  # only Covid19 and scRNA have it
                     quality_lower_is_better=LOWER_IS_BETTER[task])
        parts.append(q[QUALITY_COLUMNS])
    return pd.concat(parts)


def main():
    energy = pd.read_csv(ENERGY_CSV)
    energy["mem_delta_MiB"] = (energy.mem_peak_MiB - energy.mem_start_MiB).round(1)
    quality = read_quality(energy)
    one_each = quality.run_id.is_unique and set(quality.run_id) == set(energy.run_id)
    assert one_each, "not one quality row per run"

    runs = energy[ENERGY_COLUMNS].rename(columns={"eb_seconds": "time_s"}).merge(quality, on="run_id")
    runs["task"] = pd.Categorical(runs.task, TASKS, ordered=True)
    runs = runs.sort_values(["task", "treatment", "rep"]).reset_index(drop=True)
    write_csv(runs, PROCESSED_DIR / "runs_all.csv")


if __name__ == "__main__":
    main()
