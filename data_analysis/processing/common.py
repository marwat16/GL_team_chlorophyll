"""Paths and small helpers shared by the processing scripts."""
import json
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
RAW_DIR = PACKAGE / "data" / "raw"
PROCESSED_DIR = PACKAGE / "data" / "processed"
ENERGY_CSV = PROCESSED_DIR / "energy_runs.csv"
# The runner data is not copied into this folder. It is read from the team repository next to it.
RUNNER_IN_REPO = PACKAGE.parent / "experiment-runner" / "greenlab" / "experiments" / "full_experiment"


def task_runs(task):
    """The runs of one task from step 1, in execution order."""
    energy = pd.read_csv(ENERGY_CSV)
    return energy[energy.task == task].sort_values("order").reset_index(drop=True)


def last_lines(path, n):
    """The last n lines of a JSON-lines log, numbered from the top of the file in jsonl_line."""
    lines = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    assert len(lines) >= n, f"{path.name}: {len(lines)} lines for {n} runs"
    lines.insert(0, "jsonl_line", range(1, len(lines) + 1))
    return lines.tail(n).reset_index(drop=True)


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")  # the same line ends on Windows
    print(f"{len(df)} runs written to {path.relative_to(PACKAGE)}")


def save_quality(quality, runs, task):
    """Write data/processed/quality_<task>.csv after checking that it has one row per run."""
    assert quality.run_id.is_unique and set(quality.run_id) == set(runs.run_id), f"not one row per {task} run"
    write_csv(quality.sort_values("order"), PROCESSED_DIR / f"quality_{task}.csv")
