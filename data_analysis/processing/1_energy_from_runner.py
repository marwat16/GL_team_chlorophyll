"""Step 1: energy, time and resource use of every run, from its EnergiBridge samples.

The energy column of the Experiment Runner cannot be used. It adds the CPU cores (PP0) on top of the CPU
package that already contains them, it leaves out the GPU, and it does not correct the 32-bit RAPL counter
when it wraps around. So the energy of every run is recomputed here as

    energy_J = cpu_J (CPU package) + dram_J + gpu_J

Input   <runner>/run_table.csv and <runner>/<run_id>/energibridge.csv
Output  data/processed/energy_runs.csv, one row per run in execution order (column order)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ENERGY_CSV, RUNNER_IN_REPO, write_csv

PACKAGE_J, DRAM_J, GPU_MW = "PACKAGE_ENERGY (J)", "DRAM_ENERGY (J)", "GPU0_POWER (mWatts)"
# RAPL counts in units of 2^-14 J in a 32-bit register, so the counter wraps around every 262,144 J
WRAP_J = 2**32 * 2**-14
MIB = 2**20  # EnergiBridge reports memory in bytes


def counter_energy(counter):
    """Energy (J) of a cumulative RAPL counter. The counter only grows, so a negative step is a wrap."""
    steps = counter.astype(float).diff().iloc[1:]
    return float(steps.where(steps >= 0, steps + WRAP_J).sum())


def measure_run(eb_csv):
    """Energy, time and resource use of one run from its energibridge.csv."""
    samples = pd.read_csv(eb_csv).dropna(subset=["Time", PACKAGE_J, DRAM_J, GPU_MW])
    t = samples["Time"].to_numpy(dtype=float) / 1000  # Unix ms to s
    cpu_J = counter_energy(samples[PACKAGE_J])
    dram_J = counter_energy(samples[DRAM_J])
    # each GPU power sample holds until the next one
    gpu_J = float((samples[GPU_MW].to_numpy(dtype=float)[:-1] / 1000 * np.diff(t)).sum())
    cpu_pct = samples.filter(regex=r"^CPU_USAGE_\d+$").mean(axis=1)  # mean over the logical cores
    mem_MiB = samples["USED_MEMORY"].astype(float) / MIB
    return {
        "start_unix": round(t[0], 3), "end_unix": round(t[-1], 3), "eb_seconds": round(t[-1] - t[0], 3),
        "energy_J": round(cpu_J + dram_J + gpu_J, 3), "cpu_J": round(cpu_J, 3), "dram_J": round(dram_J, 3),
        "gpu_J": round(gpu_J, 3),
        # the GPU is idle before every run, so the first sample is its idle draw
        "gpu_idle_W": float(samples[GPU_MW].iloc[0]) / 1000,
        "cpu_pct_mean": round(float(cpu_pct.mean()), 4),
        "gpu_pct_mean": round(float(samples["GPU0_USAGE"].mean()), 4),
        "mem_start_MiB": round(float(mem_MiB.iloc[0]), 1), "mem_peak_MiB": round(float(mem_MiB.max()), 1),
        "gpu_mem_peak_MiB": float(samples["GPU0_MEMORY_USED"].max()),
        "gpu_temp_start_C": float(samples["GPU0_TEMPERATURE"].iloc[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1: energy, time and resource use of every run.")
    parser.add_argument("--runner", metavar="DIR", type=Path, default=RUNNER_IN_REPO,
                        help="full_experiment folder of the Experiment Runner "
                             "(default: ../experiment-runner/greenlab/experiments/full_experiment)")
    runner = parser.parse_args().runner
    if not (runner / "run_table.csv").is_file():
        sys.exit(f"No run_table.csv in {runner}. Pass the full_experiment folder with --runner.")

    table = pd.read_csv(runner / "run_table.csv").rename(columns={"__run_id": "run_id", "__done": "done"})
    assert (table.done == "DONE").all(), "some runs in run_table.csv did not finish"
    rows = [{"run_id": run.run_id, "task": run.benchmark, "treatment": run.code.lower(),
             **measure_run(runner / run.run_id / "energibridge.csv")}
            for run in table.itertuples()]
    runs = pd.DataFrame(rows).sort_values("start_unix").reset_index(drop=True)
    runs.insert(1, "order", range(1, len(runs) + 1))
    runs.insert(4, "rep", runs.groupby(["task", "treatment"]).cumcount() + 1)  # 1, 2, ... per group
    write_csv(runs, ENERGY_CSV)


if __name__ == "__main__":
    main()
