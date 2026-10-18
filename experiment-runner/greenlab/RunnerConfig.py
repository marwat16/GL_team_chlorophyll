from multiprocessing import context
from sys import stderr
import pandas as pd

from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import subprocess
import shutil


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "full_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 60000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        data_list = ["scRNA-data-1", "scRNA-data-2", 
                    "Covid19-data-1", "Covid19-data-2", 
                    "Geospatial-data-1","Geospatial-data-2", 
                    "ZapBench-data-1", "ZapBench-data-2",
                    "GIFT-eval-data", "GIFT-eval-data-2"]
        benchmark_factor = FactorModel("benchmark", ["scRNA", "Covid19", "Geospatial", "ZapBench", "GIFT-eval"])
        code_factor = FactorModel("code", ["AI", "Human"])

        self.run_table_model = RunTableModel(
            factors=[
                benchmark_factor,
                code_factor,
            ],
            repetitions = {
                "scRNA": 20,
                "Covid19": 10,
                "Geospatial": 10,
                "ZapBench": 20,
                "GIFT-eval": 10,
            },
            data_columns=['energy consumption (J)', 'execution time (s)', "CPU usage max (%)", 'CPU usage avg (%)', 
                          'GPU usage max (%)', 'GPU usage avg (%)', "Total memory (MB)", "Memory use mean (MB)", "Memory use max (MB)"],
            shuffle = True
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        subprocess.run(
            ["python3", "/home/gl_chlorophyll/GL_team_chlorophyll/experiment_code/warmup.py"],
            check=True,
        )

    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        pass       

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        remote_file = f"/tmp/energibridge_{context.run_dir.name}.csv"
        print(f"context.execute_run['benchmark']: {context.execute_run['benchmark']}")
        self.remote_energy_file = remote_file

        remote_code_file = (
            f"/home/gl_chlorophyll/GL_team_chlorophyll/experiment_code/"
            f"{context.execute_run['benchmark']}/"
            f"{context.execute_run['code']}.sh"
        )

        remote_cmd = (
            f"rm -f {remote_file} && "
            f"energibridge --gpu --summary "
            f"-o {remote_file} "
            f"bash {remote_code_file}"
        )

        self.profiler_process = subprocess.Popen(
            ["bash", "-c", remote_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        stdout, stderr = self.profiler_process.communicate()

        self.run_failed = self.profiler_process.returncode != 0
        if self.run_failed:
            (context.run_dir / "error.log").write_text(stderr)
            output.console_log(f"Run failed, see {context.run_dir / 'error.log'}")

    def stop_measurement(self, context: RunnerContext) -> None:
        local_file = context.run_dir / "energibridge.csv"
        if Path(self.remote_energy_file).exists():
            shutil.copy(self.remote_energy_file, local_file)

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        if self.run_failed or not (context.run_dir / "energibridge.csv").exists():
            return {column: None for column in self.run_table_model.data_columns}
        
        csv_file = context.run_dir / "energibridge.csv"

        df = pd.read_csv(csv_file)

        energy = (
            (df["PACKAGE_ENERGY (J)"].iloc[-1]
            - df["PACKAGE_ENERGY (J)"].iloc[0]) +
            (df["DRAM_ENERGY (J)"].iloc[-1]
            - df["DRAM_ENERGY (J)"].iloc[0]) +
            (df["PP0_ENERGY (J)"].iloc[-1]
            - df["PP0_ENERGY (J)"].iloc[0]) +
            (df["PP1_ENERGY (J)"].iloc[-1]
            - df["PP1_ENERGY (J)"].iloc[0])
        )

        runtime = df["Delta"].sum() / 1000.0
        cpu_usage = (df["CPU_USAGE_0"] + df["CPU_USAGE_1"] + df["CPU_USAGE_2"] + df["CPU_USAGE_3"] 
        + df["CPU_USAGE_4"] + df["CPU_USAGE_5"] + df["CPU_USAGE_6"] + df["CPU_USAGE_7"]
        + df["CPU_USAGE_8"] + df["CPU_USAGE_9"] + df["CPU_USAGE_10"] + df["CPU_USAGE_11"]
        + df["CPU_USAGE_12"] + df["CPU_USAGE_13"] + df["CPU_USAGE_14"] + df["CPU_USAGE_15"])/16

        memory = df["USED_MEMORY"]

        return {
            "energy consumption (J)": energy,
            "execution time (s)": runtime,
            "CPU usage max (%)": cpu_usage.max(),
            "CPU usage avg (%)": cpu_usage.mean(),
            "GPU usage max (%)": df["GPU0_USAGE"].max(),
            "GPU usage avg (%)": df["GPU0_USAGE"].mean(),
            "Total memory (MB)": df["TOTAL_MEMORY"].iloc[0] / (1048576),
            "Memory use mean (MB)": memory.mean() / (1048576),
            "Memory use max (MB)": df["USED_MEMORY"].max() / (1048576)
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None