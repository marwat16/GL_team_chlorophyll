#!/usr/bin/env bash
# Checks that both treatments run end to end on the prepared input.
set -e
INPUT=${1:-input_20k.h5ad}
for t in ai human; do
  echo "== $t"
  python run_one.py --treatment $t --input "$INPUT"
done
