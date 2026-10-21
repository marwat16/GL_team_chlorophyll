"""Runs the processing steps 1 to 3 in order. The R analysis runs after it with Rscript analysis/run_all.R.

Start it with the Python of the .venv (README.md, Setup). Every step runs with that same Python.

Usage   python processing/run_all.py [--runner DIR]
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["1_energy_from_runner.py", "2_quality_covid19.py", "2_quality_scrna.py", "2_quality_gift_eval.py",
         "2_quality_zapbench.py", "3_build_runs_all.py"]


def main():
    parser = argparse.ArgumentParser(description="Runs the processing steps 1 to 3 in order.")
    parser.add_argument("--runner", metavar="DIR",
                        help="full_experiment folder of the Experiment Runner, passed to step 1 "
                             "(default: ../experiment-runner/greenlab/experiments/full_experiment)")
    runner = parser.parse_args().runner
    if not all(importlib.util.find_spec(name) for name in ["numpy", "pandas"]):
        sys.exit(f"numpy or pandas is missing in {sys.executable}. Start this with the .venv Python of "
                 "README.md (Setup).")
    for step in STEPS:
        print(f"==> {step}", flush=True)
        args = ["--runner", runner] if runner and step.startswith("1_") else []  # only step 1 reads it
        if subprocess.run([sys.executable, str(HERE / step), *args]).returncode != 0:
            sys.exit(f"{step} failed, see the error above")
    print("Done. Now run the R analysis with: Rscript analysis/run_all.R")


if __name__ == "__main__":
    main()
