import argparse
import csv
import json
import os
from typing import Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.dataset.split import split
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality
from gluonts.transform.feature import LastValueImputation

import era

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--configs", type=int, nargs="*", default=None, help="indices into era.config_list")
# ADDED LINE: optional override of `runs` below, e.g. --runs m4_quarterly:short bitbrains_fast_storage/5T:short
parser.add_argument("--runs", nargs="*", default=None, help="dataset:term pairs, overrides the `runs` list")
args = parser.parse_args()

load_dotenv()

# data sets to run
runs = [
    ("bitbrains_fast_storage/5T", "short"),
    ("m4_quarterly", "short"),
]
# ADDED LINES: use the dataset:term pairs from --runs instead of the list above, if given
if args.runs:
    runs = [tuple(r.rsplit(":", 1)) for r in args.runs]

with open("dataset_properties.json") as f:
    dataset_properties_map = json.load(f)

configs = era.config_list if args.configs is None else [era.config_list[i] for i in args.configs]


class EraPredictor(RepresentablePredictor):
    """GluonTS wrapper around era.fit_and_predict_fn (point forecasts, each channel forecast separately)."""

    def __init__(self, prediction_length: int, season_length: int, config: dict):
        super().__init__(prediction_length=prediction_length)
        self.season_length = season_length
        self.config = config
        self.imputation = LastValueImputation()

    def predict(self, dataset, **kwargs) -> Iterator[Forecast]:
        for entry in dataset:
            full_target = np.asarray(entry["target"], np.float32)
            channels = full_target[None] if full_target.ndim == 1 else full_target

            start = entry["start"]
            index = pd.date_range(start=start.to_timestamp(), periods=channels.shape[-1], freq=start.freq)
            prediction_index = pd.date_range(
                start=index[-1], periods=self.prediction_length + 1, freq=index.freq
            )[1:]

            points = []
            for channel in channels:
                if np.isnan(channel).any():
                    channel = self.imputation(channel.copy())
                series = pd.Series(channel, index=index, name=str(entry["item_id"]))
                point = era.fit_and_predict_fn(series, prediction_index, self.season_length, self.config)
                points.append(point.to_numpy(dtype=np.float32))
            point = points[0] if full_target.ndim == 1 else np.stack(points, axis=-1)

            yield QuantileForecast(
                forecast_arrays=np.tile(point, (len(QUANTILES) + 1,) + (1,) * point.ndim),
                forecast_keys=["mean"] + [str(q) for q in QUANTILES],
                start_date=prediction_index[0].to_period(freq=start.freq),
                item_id=entry["item_id"],
            )


metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(forecast_type="mean"),
    MAE(forecast_type=0.5),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES),
]

OUT_DIR = f"../results/{args.out_dir}"
os.makedirs(OUT_DIR, exist_ok=True)
csv_file_path = os.path.join(OUT_DIR, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

if not os.path.exists(csv_file_path):
    with open(csv_file_path, "a", newline="") as csvfile:
        csv.writer(csvfile).writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[mean]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ]
        )

done_datasets = pd.read_csv(csv_file_path)["dataset"].values
print("Done datasets")
print(done_datasets)


def write_row(ds_config, model_name, res, ds_key):
    with open(csv_file_path, "a", newline="") as csvfile:
        csv.writer(csvfile).writerow(
            [
                ds_config,
                model_name,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[mean]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
            ]
        )


def validation_data(dataset):
    """GIFT-Eval's official validation fold: forecast the window between training_dataset and
    validation_dataset (one window per series), which ends where the test windows begin,
    so no test data is used for selection."""
    _, template = split(dataset.validation_dataset, offset=-dataset.prediction_length)
    return template.generate_instances(prediction_length=dataset.prediction_length, windows=1)


for ds_name, term in runs:
    if True:
        if "/" in ds_name:
            ds_key, ds_freq = ds_name.split("/")
        else:
            ds_key = ds_name
            ds_freq = dataset_properties_map[pretty_names.get(ds_key.lower(), ds_key.lower())]["frequency"]
        ds_key = pretty_names.get(ds_key.lower(), ds_key.lower())
        ds_config = f"{ds_key}/{ds_freq}/{term}"

        if ds_config in done_datasets:
            print(f"Done with {ds_config}. Skipping...")
            continue

        dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        season_length = get_seasonality(dataset.freq)
        print(f"Dataset: {ds_config}, Freq = {dataset.freq}, H = {dataset.prediction_length}")

        eval_kwargs = dict(
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        # Score every config on the validation fold only
        val_data = validation_data(dataset)
        val_mase = {}
        for config in configs:
            predictor = EraPredictor(dataset.prediction_length, season_length, config)
            val_res = evaluate_model(predictor, test_data=val_data, metrics=[MASE()], **eval_kwargs)
            val_mase[config["name"]] = float(val_res["MASE[0.5]"][0])

        # Pick the config with the lowest validation MASE 
        scores = {name: (v if np.isfinite(v) else np.inf) for name, v in val_mase.items()}
        best = min(scores, key=scores.get)
        print(f"Validation MASE per config: {val_mase}")
        print(f"Selected {best} for {ds_config}")

        # Run the test windows once, with the selected config only
        best_config = next(c for c in configs if c["name"] == best)
        predictor = EraPredictor(dataset.prediction_length, season_length, best_config)
        res = evaluate_model(predictor, test_data=dataset.test_data, metrics=metrics, **eval_kwargs)
        write_row(ds_config, era.MODEL_NAME, res, ds_key)
        print(f"Results for {ds_name} have been written to {csv_file_path}")

df = pd.read_csv(csv_file_path).sort_values(by="dataset")
print(df[["dataset", "eval_metrics/MASE[0.5]", "eval_metrics/NRMSE[mean]"]].to_string(index=False))
