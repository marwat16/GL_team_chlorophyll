# GIFT-Eval: era_run.py and ttm.py

Minimal copy of [gift-eval](https://github.com/SalesforceAIResearch/gift-eval) to run
`notebooks/era_run.py` and `notebooks/ttm.py`
(TinyTimeMixer, 20% few-shot) on the same GIFT-Eval datasets.

Target Python version: **3.12**.

## Setup
Run `bash setup.sh` once; it performs these steps (plus the one-time granite-tsfm install for `ttm.py`).
`AI.sh` (runs `era_run.py`) and `Human.sh` (runs `ttm.py`) only run the code, so setup is not measured.
To do it manually, from this folder:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# dataset (not stored in git)
hf download Salesforce/GiftEval --repo-type=dataset --local-dir ./data/gift-eval
# create a .env file in this folder containing: GIFT_EVAL=<absolute path to ./data/gift-eval>
```
