import time

T_START = time.time()  # the run's clock starts here, before the slow imports

import argparse
import json
import pathlib
import shutil

from common import RunRecord, git_commit, use_local_data

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", default="data/zap36k", help="data folder made by setup/get_data.py")
parser.add_argument("--out", default="runs/test/tsmixer",
                    help="run folder, DELETED at the start of the run (checkpoints, logs, result.json)")
parser.add_argument("--steps", type=int, default=2980,
                    help="training steps; -1 = ZAPBench's own early stopping")
parser.add_argument("--val-every", type=int, default=250,
                    help="validate and save a checkpoint every N steps")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Start from an empty folder, ZAPBench would otherwise resume from old checkpoints
out = pathlib.Path(args.out).resolve()
shutil.rmtree(out, ignore_errors=True)

record = RunRecord("tsmixer", out, T_START, seed=args.seed,
                   budget={"steps": args.steps, "val_every": args.val_every},
                   data={"dir": args.data})

# The data must point to the local copy before ZAPBench is imported (keep this order)
n_t, n_neurons = use_local_data(args.data)
record.set("data.frames", n_t)
record.set("data.neurons", n_neurons)

import zapbench
from zapbench.ts_forecasting import infer, train
from zapbench.ts_forecasting.configs import infer as infer_config
from zapbench.ts_forecasting.configs import tsmixer

record.set("code.zapbench_commit", git_commit(pathlib.Path(zapbench.__file__).resolve().parents[1]))

# ZAPBench's TSMixer config with 4 input frames. Only the run length, how often to validate and
# checkpoint and the loss logging change, and the profiler is off (it would add work to the run).
config = tsmixer.get_config(f"timesteps_input=4,seed={args.seed}")
config.num_train_steps = args.steps
config.val_every_steps = config.checkpoint_every_steps = args.val_every
config.log_loss_every_steps = min(config.log_loss_every_steps, args.val_every)
config.periodic_profiling = False

# Train, then predict the test windows with the checkpoint of the best validation loss, with the
# functions ZAPBench's main_train.py and main_infer.py call. TSMixer loads its data inside train.
record.phase("train")
t0 = time.time()
train.train_and_evaluate(config, out / "train")
t1 = time.time()
record.phase("test")
infer.inference(infer_config.get_config(f"exp_workdir={out / 'train'}"), out / "infer")
t2 = time.time()
record.phase("report")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import tensorflow as tf
from zapbench import constants

# Validation MAE at every validation step, from the TensorBoard log of training
event_file = next((out / "train").glob("events.out.tfevents*"))
ea = EventAccumulator(str(event_file), size_guidance={"tensors": 0})  # 0 = keep all values
ea.Reload()
val = [(e.step, float(tf.make_ndarray(e.tensor_proto))) for e in ea.Tensors("val_loss")]
print("\nvalidation MAE by step:", ", ".join(f"{s}: {v:.4f}" for s, v in val))

# Test MAE per condition, from the JSON files of inference. The folder under step/ is named
# after the checkpoint that was used.
step_dir = next((out / "infer" / "inference" / "step").iterdir())
print(f"inference used the checkpoint from step {step_dir.name}")
maes = []
by_condition = {}  # condition number -> (mean test MAE, the 32 per-step values)
for f in sorted(step_dir.glob("*.json")):
    name = f.stem  # e.g. test_condition_4, test_holdout_condition_3
    scores = json.loads(f.read_text())
    mae = scores[f"infer_{name}_mae"]
    maes.append(mae)
    cond_id = int(name.rsplit("_", 1)[1])
    by_condition[cond_id] = (mae, [scores[f"infer_{name}_mae_step/{k}"] for k in range(1, 33)])
    print(f"  {constants.CONDITION_NAMES[cond_id]:<10} test MAE {mae:.6f}")
print(f"test MAE averaged over {len(maes)} conditions x 32 steps: {np.mean(maes):.6f}")
print(f"time: train {t1 - t0:.1f} s, inference {t2 - t1:.1f} s")

# Training always validates on its last step, so that step is the number of steps run.
# avg_8 leaves out the held-out condition (taxis), like ZAPBench's official average.
names = constants.CONDITION_NAMES
ids = sorted(by_condition)
kept = [i for i in ids if i not in constants.CONDITIONS_HOLDOUT]
record.set("steps_run", max(s for s, _ in val))
record.set("best_step", int(step_dir.name))
record.set("val_history", [[s, v] for s, v in val])
record.set("test_windows", sum(s["num_windows"] for s in config.infer_sets))
record.set("test_mae.avg_9", float(np.mean([by_condition[i][0] for i in ids])))
record.set("test_mae.avg_8", float(np.mean([by_condition[i][0] for i in kept])))
record.set("test_mae.per_condition", {names[i]: by_condition[i][0] for i in ids})
record.set("test_mae.per_step", {names[i]: by_condition[i][1] for i in ids})
record.finish()
print(f"result: {record.path}")
