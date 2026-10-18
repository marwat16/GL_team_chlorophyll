#!/bin/bash
set -e
set -o pipefail
cd "$(dirname "$0")"
# One run of the ai treatment. The JSON line with timings and checks is also kept in runs_ai.jsonl.
.venv/bin/python run_one.py --treatment ai --ref-date 2025-02-01 | tee -a runs_ai.jsonl
