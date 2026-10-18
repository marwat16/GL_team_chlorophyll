#!/bin/bash
set -e
set -o pipefail
cd "$(dirname "$0")"
# One run of the human treatment. The JSON line with timings and checks is also kept in runs_human.jsonl.
.venv/bin/python run_one.py --treatment human --ref-date 2025-02-01 --num-bags 100 | tee -a runs_human.jsonl
