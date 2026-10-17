# ZapBench: ERA node 775 and TSMixer

Runs ERA's node 775 (`run/node775.py`) and TSMixer from
[ZAPBench](https://github.com/google-research/zapbench) (commit `9a26d3f`), both unchanged, on the
first 36,000 neurons. Quality is the test MAE (lower is better).

Target Python version: **3.12**.

## Setup
Run `bash setup/setup.sh` once on the server. It builds `.venv-era` (TensorFlow) and `.venv-human`
(JAX) and downloads the data to `data/zap36k`.
`AI.sh` (ERA, 5 epochs) and `Human.sh` (TSMixer, 2,980 steps) only run the code, so setup is not measured.

## Output
Each call takes the next seed (0, 1, 2, ...) and writes `runs/era/run_N/result.json` or
`runs/tsmixer/run_N/result.json` with the status, seed and test MAE (`test_mae.avg_8`).
Move `runs/` away before a new experiment so the seeds start at 0.
