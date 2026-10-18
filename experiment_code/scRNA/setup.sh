#!/bin/bash
# One-time setup on the server. Creates .venv with the pinned libraries and,
# if the ERA Kaggle train file is in this folder, builds input_20k.h5ad from it.
set -e
cd "$(dirname "$0")"
TRAIN=ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [ ! -f input_20k.h5ad ]; then
  if [ -f "$TRAIN" ]; then
    .venv/bin/python prepare_input.py --train "$TRAIN"
  else
    echo "Copy $TRAIN into this folder and run setup.sh again." >&2
    exit 1
  fi
fi
echo "Setup done."
