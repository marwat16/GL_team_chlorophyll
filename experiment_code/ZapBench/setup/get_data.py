import argparse
import hashlib
import pathlib

import numpy as np
import tensorstore as ts

SOURCE = "gs://zapbench-release/volumes/20240930/"  # public bucket, no login needed


def sha256(a):
    # Hash of the numbers only, so it does not depend on how they are stored on disk
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


# Copies all 7,879 frames of the first N of the 71,721 neurons, plus the stimulus features
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out", default="data/zap36k", help="folder to write to (overwritten)")
parser.add_argument("--neurons", type=int, default=36000)
args = parser.parse_args()
out = pathlib.Path(args.out).resolve()

# Neural activity: zarr v3, float32 [frames, neurons], kept in 512 x 512 chunks
src = ts.open({"driver": "zarr3", "kvstore": SOURCE + "traces/"}).result()
print("source traces", src.shape)
traces = src[:, :args.neurons].read().result()
dst = ts.open({
    "driver": "zarr3", "kvstore": f"file://{out}/traces/", "create": True, "delete_existing": True,
    "metadata": {"shape": list(traces.shape), "data_type": "float32",
                 "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [512, 512]}}},
}).result()
dst.write(traces).result()
print("traces", traces.shape, sha256(traces))

# Stimulus features: zarr v2, float32 [frames, 26], which ZAPBench's training also loads
src = ts.open({"driver": "zarr", "kvstore": SOURCE + "stimuli_features/"}).result()
stim = src.read().result()
dst = ts.open({
    "driver": "zarr", "kvstore": f"file://{out}/stimuli_features/",
    "create": True, "delete_existing": True,
    "metadata": {"shape": list(stim.shape), "chunks": list(stim.shape), "dtype": "<f4",
                 "compressor": None},
}).result()
dst.write(stim).result()
print("stimuli_features", stim.shape, sha256(stim))
