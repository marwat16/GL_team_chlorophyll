#!/bin/bash
set -e
cd "$(dirname "$0")"
# TSMixer for 2,980 steps. The runner passes no seed, so the N-th run gets seed N and the folder runs/tsmixer/run_N.
OUT=runs/tsmixer
mkdir -p $OUT
N=$(ls -d $OUT/run_* 2>/dev/null | wc -l | tr -d ' ')
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-human/bin/python run/run_tsmixer.py --data data/zap36k --steps 2980 --val-every 250 --seed $N --out $OUT/run_$N
rm -rf $OUT/run_$N/train/checkpoints
