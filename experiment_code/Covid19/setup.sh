#!/bin/bash
# One-time setup on the server. Creates .venv with the pinned libraries.
# dataset.csv and locations.csv are already in this folder (real data, as of 30 April 2025).
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_one.py --treatment human --load-only
echo "Setup done."
