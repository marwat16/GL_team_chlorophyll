# GIFT-Eval: era_run.py and ttm.py

Minimal copy of [gift-eval](https://github.com/SalesforceAIResearch/gift-eval) to run
`notebooks/era_run.py` and `notebooks/ttm.py`
(TinyTimeMixer, 20% few-shot) on the same GIFT-Eval datasets.

## Setup
```bash
cd gift-eval
python -m venv .venv                  
source .venv/bin/activate                
pip install -e .

# dataset (not stored in git)
hf download Salesforce/GiftEval --repo-type=dataset --local-dir ./data/gift-eval
# create a .env file in this folder containing: GIFT_EVAL=./data/gift-eval
```
