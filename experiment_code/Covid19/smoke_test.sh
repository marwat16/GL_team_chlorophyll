#!/usr/bin/env bash
# Checks that both treatments run end to end on the prepared input.
# The human treatment uses 2 bags here so the check takes about a minute.
# Use NUM_BAGS=100 ./smoke_test.sh for the real configuration (much slower).
set -e
REF=${1:-2025-02-01}
NUM_BAGS=${NUM_BAGS:-2}
echo "== ai"
python run_one.py --treatment ai --ref-date "$REF"
echo "== human ($NUM_BAGS bags)"
python run_one.py --treatment human --ref-date "$REF" --num-bags "$NUM_BAGS"
