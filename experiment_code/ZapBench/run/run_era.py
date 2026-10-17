import time

T_START = time.time()  # the run's clock starts here, before the slow imports

import argparse
import hashlib
import pathlib

from common import RunRecord, use_local_data

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", default="data/zap36k", help="data folder made by setup/get_data.py")
parser.add_argument("--code", default="run/node775.py", help="ERA's program, run unchanged")
parser.add_argument("--epochs", type=int, default=5,
                    help="cap on the training epochs (596 steps each); 0 = ERA's own early stopping")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/test/era", help="folder for result.json")
args = parser.parse_args()

record = RunRecord("era", args.out, T_START, seed=args.seed, budget={"epochs": args.epochs},
                   data={"dir": args.data}, code={"file": args.code})

# Imported after the options so that --help answers quickly (keep this order)
import keras
import numpy as np
from connectomics.jax import metrics
from zapbench import constants, data_utils
from zapbench.ts_forecasting import data_source

n_t, n_neurons = use_local_data(args.data)
record.set("data.frames", n_t)
record.set("data.neurons", n_neurons)
keras.utils.set_random_seed(args.seed)
C = 4  # the model sees 4 frames and predicts the next 32

# Cap the epochs without editing ERA's file: its Model.fit ends in keras.Model.fit
if args.epochs > 0:
    keras_fit = keras.Model.fit

    def capped_fit(self, *a, **kw):
        kw["epochs"] = min(kw.get("epochs", 1), args.epochs)
        return keras_fit(self, *a, **kw)

    keras.Model.fit = capped_fit

# ERA's file expects six variables from its unreleased notebook. They are built as in ZAPBench's
# tutorial: the train split of the 8 training conditions, context 4, batch size 8.
record.phase("load_data")
sources = []
for cond in constants.CONDITIONS_TRAIN:
    spec = data_utils.adjust_spec_for_condition_and_split(
        condition=cond, split="train", spec=data_utils.get_spec(constants.TIMESERIES_NAME),
        num_timesteps_context=C)
    cfg = data_source.TensorStoreTimeSeriesConfig(
        input_spec=spec, timesteps_input=C, timesteps_output=constants.PREDICTION_WINDOW_LENGTH)
    sources.append(data_source.TensorStoreTimeSeries(cfg, prefetch=True))
source = data_source.ConcatenatedTensorStoreTimeSeries(*sources)
era_globals = {"__name__": "__era__", "source": source, "num_timesteps_context": C,
               "batch_size": 8, "num_epochs": 1, "shuffle": True, "data_utils": data_utils}
print(f"training windows: {len(source)}")

# Run ERA's file unchanged. It builds its model and trains it (model.fit at the end of the file).
code = pathlib.Path(args.code).read_text()
md5 = hashlib.md5(code.encode()).hexdigest()
record.set("code.md5", md5)
print(f"ERA code {args.code}, md5 {md5}")
record.phase("train")
t0 = time.time()
exec(compile(code, args.code, "exec"), era_globals)
t1 = time.time()
model = era_globals["model"]

history = model.history.history
steps_per_epoch = len(source) // 8  # ERA's loader drops the last partial batch
epochs_run = len(model.history.epoch)
record.set("epochs_run", epochs_run)
record.set("steps_per_epoch", steps_per_epoch)
record.set("steps_run", epochs_run * steps_per_epoch)
val_loss = [float(v) for v in history.get("val_loss", [])]
if val_loss:
    record.set("best_epoch", int(np.argmin(val_loss)) + 1)
record.set("val_history", [[epoch, v] for epoch, v in enumerate(val_loss, start=1)])

# Test MAE on the 1,557 test windows of all 9 conditions (ZAPBench's evaluation loop)
record.phase("test")
infer_source = data_source.TensorStoreTimeSeries(
    data_source.TensorStoreTimeSeriesConfig(
        input_spec=data_utils.get_spec(constants.TIMESERIES_NAME),
        timesteps_input=C, timesteps_output=constants.PREDICTION_WINDOW_LENGTH),
    prefetch=True)
per_cond, n_windows = [], 0
print()
for cond_id, name in enumerate(constants.CONDITION_NAMES):
    split = "test_holdout" if cond_id in constants.CONDITIONS_HOLDOUT else "test"
    lo, hi = data_utils.adjust_condition_bounds_for_split(
        split, *data_utils.get_condition_bounds(cond_id), num_timesteps_context=C)
    maes = []
    for w in range(data_utils.get_num_windows(lo, hi, C)):
        el = infer_source[lo + w]
        prediction = model.predict_non_batched(el["series_input"])
        maes.append(np.array(metrics.mae(predictions=prediction, targets=el["series_output"])))
    per_cond.append(np.stack(maes).mean(axis=0))  # one MAE per step ahead (32)
    n_windows += len(maes)
    print(f"  {name:<10} test MAE {per_cond[-1].mean():.6f}")
t2 = time.time()

# ERA's own log always says "Training completed after 120 epochs", the line below is the real count
print(f"test MAE averaged over 9 conditions x 32 steps ({n_windows} windows): "
      f"{np.mean(per_cond):.6f}")
print(f"epochs run: {epochs_run}; time: train {t1 - t0:.1f} s, test {t2 - t1:.1f} s")

# avg_8 leaves out the held-out condition (taxis), like ZAPBench's official average
names = constants.CONDITION_NAMES
kept = [i for i in range(len(names)) if i not in constants.CONDITIONS_HOLDOUT]
record.set("test_windows", n_windows)
record.set("test_mae.avg_9", float(np.mean(per_cond)))
record.set("test_mae.avg_8", float(np.mean([per_cond[i] for i in kept])))
record.set("test_mae.per_condition", {names[i]: float(m.mean()) for i, m in enumerate(per_cond)})
record.set("test_mae.per_step", {names[i]: m for i, m in enumerate(per_cond)})
record.finish()
print(f"result: {record.path}")
