# scRNA-seq batch integration, local test of the two treatments

This folder lets you run the AI treatment (ERA node 429) and the human
treatment (OpenProblems BBKNN) on the same input, on your own machine.
It only checks that both codes run and gives rough timings.
Energy measurement (Experiment Runner + EnergiBridge) comes later and
will call `run_one.py` once per run.

## Files

| File | Purpose |
| --- | --- |
| `ai_node429.py` | AI treatment, copied verbatim from ERA study 173582811, node 429 |
| `human_bbknn.py` | Human treatment, OpenProblems BBKNN script without VIASH/I-O, plus the OpenProblems normalization step |
| `prepare_input.py` | Builds the fixed input (20,000 cells, seed 42) from the ERA data, or synthetic data |
| `run_one.py` | Runs one treatment once and prints timings and a check of the output |
| `smoke_test.sh` | Runs both treatments one after the other |
| `requirements.txt` | Pinned library versions (fixed factor of the experiment) |

## 1. Requirements

- Linux or macOS (on Windows use WSL). Linux is what the Green Lab server uses.
- Python 3.12 (3.11 should also work).
- A C++ compiler for the `annoy` package (`build-essential` on Ubuntu,
  Xcode command line tools on macOS).
- About 4 GB of free RAM.

## 2. Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Data

Quick check without downloading anything:

```bash
python prepare_input.py --synthetic
```

Real data. Download the ERA single-cell dataset from Kaggle
(https://kaggle.com/datasets/02bdd1a079f253e04766213cb09e71d79a912200b901cac83fe7ab40bcd7cd48)
and put the files in `datasets/single-cell-batch-integration/`. Then:

```bash
python prepare_input.py --train datasets/single-cell-batch-integration/ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad
```

This reads the raw counts (`layers/counts`) and applies the notebook's own
subsampling function with 20,000 cells and seed 42. It prints the number of
batches and cell types, which you need for the subject table.

## 4. Run

```bash
./smoke_test.sh                      # both treatments
python run_one.py --treatment ai     # one treatment
python run_one.py --treatment human --load-only   # loading + imports only
```

Each run prints one JSON line, for example (synthetic data, 1 CPU core):

```
{"treatment": "ai",    "load_s": 1.97,  "method_s": 58.9, "peak_rss_mb": 1894, "has_embedding": true,  "has_graph": true, ...}
{"treatment": "human", "load_s": 13.9,  "method_s": 26.5, "peak_rss_mb": 891,  "has_embedding": false, "has_graph": true, ...}
```

`load_s` includes the imports. The human side is slower to load because
numba compiles BBKNN's dependencies on first use. `--load-only` measures this
part alone, so it can be subtracted from a whole-process energy measurement.

## Notes

- The cell type column is removed before either method runs, as in the ERA notebook.
- Do not change `ai_node429.py`. The point is to measure the code ERA produced.
- Quality scores are not computed here. In the report they are read from the
  released ERA data (Section 3.3), so the scoring stack (scib, R, kBET) is not needed.
- If you see an error about `allow_write_nullable_strings`, your pandas is 3.x.
  Install the pinned version from `requirements.txt`.
