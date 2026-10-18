# Geospatial: ERA node 771 and MA-UNet

Runs ERA's node 771 (`ai_generated.py`, study 174509051) and MA-UNet from [Multiattention-UNet](https://github.com/1343744768/Multiattention-UNet) (commit `726f7f6`, in `ma_unet_repo/`) on DLRSD with a stratified 80/20 split (`train_split.json`, `test_split.json`). Quality is the test mIoU (higher is better).

Target Python version: 3.12.

## Setup

Run these steps once on the server. `AI.sh` (ERA, 90 epochs) and `Human.sh` (MA-UNet, 80 epochs, batch size 8) only run the code, so setup is not measured. From this folder:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# dataset (not stored in git): place DLRSD in data/DLRSD with Images/ and Labels/
```
## Output

Each run prints the test mIoU at the end. MA-UNet's checkpoints in `ma_unet_repo/logs/` are deleted after each run.