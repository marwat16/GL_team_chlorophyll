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



class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "new_runner_experiment"

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
        #data_factor = FactorModel("data", data_list)

        self.run_table_model = RunTableModel(
            factors=[
                benchmark_factor,
                code_factor,
                #data_factor
            ],
            #exclude_combinations=[
            #    {
            #        benchmark_factor: ['scRNA'],
            #        data_factor: [
            #            x for x in data_list
            #            if x not in ["scRNA-data-1", "scRNA-data-2"]
            #        ]
            #    },
            #    {
            #        benchmark_factor: ['Covid19'],
            #        data_factor: [
            #            x for x in data_list
            #            if x not in ["Covid19-data-1", "Covid19-data-2"]
            #        ]
            #    },
            #    {
            #        benchmark_factor: ['Geospatial'],
            #        data_factor: [
            #            x for x in data_list
            #            if x not in ["Geospatial-data-1", "Geospatial-data-2"]
            #        ]
            #    },
            #    {
            #        benchmark_factor: ['ZapBench'],
            #        data_factor: [
            #            x for x in data_list
            #            if x not in ["ZapBench-data-1", "ZapBench-data-2"]
            #        ]
            #    },
            #    {
            #        benchmark_factor: ['GIFT-eval'],
            #        data_factor: [
            #            x for x in data_list
            #            if x not in ["GIFT-eval-data", "GIFT-eval-data-2"]
            #        ]
            #    }
            #],
            repetitions = 20,
            data_columns=['energy consumption (J)', 'execution time (s)', "CPU memory (MB)", 'CPU memory usage (MB)', 
                          'GPU memory (MB)', 'GPU memory usage (MB)', "Memory usage (MB)"],
            shuffle = True
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        pass

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        pass       

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        remote_file = f"/tmp/energibridge_{context.run_dir.name}.csv"
        self.remote_energy_file = remote_file

        remote_cmd = (
            f"rm -f {remote_file} && "
            f"energibridge --summary "
            f"-o {remote_file} "
            f"sleep 5"
        )

        self.profiler_process = subprocess.Popen(
            [
                "ssh",
                "glg4",
                remote_cmd
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        stdout, stderr = self.profiler_process.communicate()

        if self.profiler_process.returncode != 0:
            raise RuntimeError(
                f"Remote EnergiBridge failed:\n{stderr}"
            )

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        local_file = context.run_dir / "energibridge.csv"

        with open(local_file, "wb") as f:
            subprocess.run(
                [
                    "ssh",
                    "glg4",
                    "cat",
                    self.remote_energy_file
                ],
                stdout=f,
                check=True
            )

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        
        csv_file = context.run_dir / "energibridge.csv"

        df = pd.read_csv(csv_file)

        energy = (
            df["PACKAGE_ENERGY (J)"].iloc[-1]
            - df["PACKAGE_ENERGY (J)"].iloc[0]
        )

        runtime = df["Delta"].sum() / 1000.0

        memory = df["USED_MEMORY"].max()

        return {
            "energy consumption (J)": energy,
            "execution time (s)": runtime,
            "CPU memory (MB)": memory / (1024 * 1024)
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None