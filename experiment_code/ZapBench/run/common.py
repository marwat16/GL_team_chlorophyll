import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from importlib import metadata

import tensorstore as ts
from zapbench import constants

# Libraries whose installed version goes into result.json
LIBRARIES = ["tensorflow", "tensorflow-cpu", "keras", "jax", "jaxlib", "numpy", "grain",
             "tensorstore", "array-record", "zapbench"]


def use_local_data(data_dir):
    """Point ZAPBench at the local copy made by setup/get_data.py, so no run downloads data.
    Call it before any ZAPBench config is built. Returns (frames, neurons)."""
    d = pathlib.Path(data_dir).resolve()
    traces = {"driver": "zarr3", "kvstore": f"file://{d}/traces/"}
    n_t, n_neurons = ts.open(traces).result().shape  # a missing folder fails here
    constants.SPECS[constants.TIMESERIES_NAME] = {**traces, "transform": {
        "input_inclusive_min": [0, 0],
        "input_exclusive_max": [[n_t], n_neurons],
        "input_labels": ["t", "f"],
    }}
    constants.COVARIATE_SPECS[constants.COVARIATE_SERIES_NAME] = {
        "driver": "zarr", "kvstore": f"file://{d}/stimuli_features/", "rank": 2,
        "transform": {
            "input_inclusive_min": [0, 0],
            "input_exclusive_max": [[n_t], [26]],  # 26 stimulus features per frame
            "input_labels": ["t", "f"],
        },
    }
    return n_t, n_neurons


class RunRecord:
    """Writes the facts of one run to <out>/result.json.

    The run is split into phases like the laps of a stopwatch: it starts in "startup" when the
    script starts, and phase(name) ends the current phase and starts the next one. The file is
    written at the start ("running"), after a crash ("failed", with the error) and by finish() ("ok").
    """

    def __init__(self, impl, out_dir, started_at, **info):
        self.path = pathlib.Path(out_dir).resolve() / "result.json"
        self.data = {"impl": impl, "status": "running", "started_at": started_at, **info}
        self.data["environment"] = environment()
        self.phases = {}
        self.current = ("startup", started_at)

        previous_hook = sys.excepthook

        def on_crash(exc_type, exc, tb):
            try:
                self._end("failed", f"{exc_type.__name__}: {exc}")
            finally:
                previous_hook(exc_type, exc, tb)

        sys.excepthook = on_crash
        self.write()

    def phase(self, name):
        now = time.time()
        self._close(now)
        self.current = (name, now)

    def set(self, key, value):
        """set("test_mae.avg_8", 0.03) stores data["test_mae"]["avg_8"] = 0.03."""
        *parents, last = key.split(".")
        node = self.data
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = plain(value)

    def finish(self):
        self._end("ok")

    def write(self):
        # Write a temporary file first, so result.json is never left half-written
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    def _close(self, now):
        if self.current is not None:
            name, start = self.current
            self.phases[name] = [start, now]
            self.current = None

    def _end(self, status, error=None):
        now = time.time()
        self._close(now)
        self.data["status"] = status
        self.data["finished_at"] = now
        if error:
            self.data["error"] = error
        durations = {name: round(end - start, 3) for name, (start, end) in self.phases.items()}
        durations["total"] = round(now - self.data["started_at"], 3)
        self.data["durations"] = durations
        self.data["phases"] = self.phases
        self.write()


def git_commit(repo_dir):
    """The commit a git checkout is on, or None."""
    if not (pathlib.Path(repo_dir) / ".git").exists():
        return None  # otherwise git would report the commit of a repository around it
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout
    except OSError:
        return None
    return out.strip() or None


def environment():
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return {"host": platform.node(), "machine": platform.machine(),
            "python": platform.python_version(), "libraries": versions}


def plain(value):
    # NumPy numbers and arrays to plain Python numbers and lists, so json can write them
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
