#!/bin/bash
# One-time setup from README.md
set -e

cd "$(dirname "$0")"

PY=""
for cand in python3.12 "py -3.12" python3 python; do
    if $cand -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.12 is required but was not found." >&2
    exit 1
fi

# Recreate the venv if it is missing 
if [ -d .venv ]; then
    if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi
    if ! "$VPY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
        rm -rf .venv
    fi
fi
[ -d .venv ] || $PY -m venv .venv
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -U huggingface_hub
for attempt in 1 2 3 4 5; do
    hf download Salesforce/GiftEval --repo-type=dataset --local-dir ./data/gift-eval \
        --include "bitbrains_fast_storage/5T/*" --include "m4_quarterly/*" && break
    [ "$attempt" -lt 5 ] || { echo "Dataset download failed." >&2; exit 1; }
    echo "Download failed, retrying ($attempt/5)..." >&2
    sleep 10
done
# .env 
# absolute path; pwd -W gives a Windows path in Git Bash (plain pwd gives /c/..., unreadable by Python)
echo "GIFT_EVAL=$(pwd -W 2>/dev/null || pwd)/data/gift-eval" > .env

# granite-tsfm for ttm.py 
cd notebooks
if [ ! -d granite-tsfm ]; then
    git clone https://github.com/ibm-granite/granite-tsfm.git
    (cd granite-tsfm && git switch gift && python -m pip install ".[notebooks]" "transformers>=4.38,<5")
fi

echo "Setup done."
