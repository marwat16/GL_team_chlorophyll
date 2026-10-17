#!/bin/bash
set -e
cd "$(dirname "$0")"
# ERA's node 775 for 5 epochs. The runner passes no seed, so the N-th run gets seed N and the folder runs/era/run_N.
OUT=runs/era
mkdir -p $OUT
N=$(ls -d $OUT/run_* 2>/dev/null | wc -l | tr -d ' ')
TF_FORCE_GPU_ALLOW_GROWTH=true .venv-era/bin/python run/run_era.py --data data/zap36k --epochs 5 --seed $N --out $OUT/run_$N
