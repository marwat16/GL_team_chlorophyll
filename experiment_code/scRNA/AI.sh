#!/bin/bash
set -e
set -o pipefail
cd "$(dirname "$0")"
# One run of the ai treatment. The JSON line with timings and checks is also kept in runs_ai.jsonl.
.venv/bin/python run_one.py --treatment ai --input input_20k.h5ad | tee -a runs_ai.jsonl
