# Human treatment. UMass-gbqr, the CovidHub model of the UMass-Amherst team,
# as implemented in reichlab/idmodels (src/idmodels/gbqr.py) and configured
# for COVID-19 in reichlab/operational-models (covid_gbqr/main.py).
#
# The original model downloads its data from the internet (iddata, S3).
# Here that single step is replaced: the model receives exactly the same
# training data as the AI treatment, converted to the data frame layout that
# iddata would have produced. Everything after loading (transform, features,
# 100 bags x 23 quantile LightGBM models, inverse transform) is the unchanged
# library code.
import datetime

import numpy as np
import pandas as pd
from iddata.constants import PANDEMIC_SEASONS
from iddata.enums import Disease
from iddata.utils import add_season_columns
from idmodels.config import GBQRModelConfig, PowerTransform, RunConfig, SourceType
from idmodels.gbqr import GBQRModel

TARGET_COL = 'Total COVID-19 Admissions'

QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
Q_LABELS = ['0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3',
            '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75',
            '0.8', '0.85', '0.9', '0.95', '0.975', '0.99']


def model_config(num_bags: int = 100) -> GBQRModelConfig:
  """Same values as reichlab/operational-models covid_gbqr/main.py."""
  return GBQRModelConfig(
      model_name='gbqr',
      incl_level_feats=True,
      num_bags=num_bags,
      bag_frac_samples=0.7,
      reporting_adj=False,
      save_feat_importance=False,
      main_source=SourceType.NHSN,
      fit_locations_separately=False,
      power_transform=PowerTransform.FOURTH_ROOT)


class GBQRFromFrame(GBQRModel):
  """GBQRModel.run() without the download and without writing a CSV."""

  def run_on_frame(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
    df = df.copy()
    df['unique_id'] = df['agg_level'] + df['location']
    transform = self._build_transform()
    df = transform.apply(df)
    pipeline = self._build_feature_pipeline(run_config)
    df, feat_names = pipeline.apply(df)
    preds_df = self._fit_and_predict(df, feat_names, run_config)
    preds_df = self._invert_and_scale(preds_df, transform, run_config)
    return self._format_output(preds_df, run_config)


def to_iddata_frame(train_x: pd.DataFrame, train_y: pd.Series) -> pd.DataFrame:
  """Builds the frame that DiseaseDataLoader + NHSNDataSource + PopulationData
  would return, from the ERA style training data."""
  d = train_x[['target_end_date', 'location', 'population']].copy()
  d['count'] = train_y.values
  d['wk_end_date'] = pd.to_datetime(d['target_end_date']).dt.strftime('%Y-%m-%d')
  d['location'] = d['location'].astype(int).map(lambda x: f'{x:02d}')
  d = add_season_columns(d)  # season and season_week, as iddata does
  d['wk_end_date'] = pd.to_datetime(d['wk_end_date'])
  d['pop'] = d['population'].astype(float)
  d['log_pop'] = np.log(d['pop'])
  d['inc'] = d['count'] / d['pop'] * 100000  # NHSN rates per 100k, as iddata
  d.loc[d['season'].isin(PANDEMIC_SEASONS), 'inc'] = np.nan  # loader default
  d['agg_level'] = 'state'
  d['source'] = SourceType.NHSN.value
  cols = ['agg_level', 'location', 'season', 'season_week', 'wk_end_date',
          'inc', 'source', 'pop', 'log_pop']
  return d[cols].sort_values(['source', 'location', 'wk_end_date'])


def fit_and_predict_human(train_x: pd.DataFrame, train_y: pd.Series,
                          test_x: pd.DataFrame, num_bags: int = 100) -> pd.DataFrame:
  """Same interface as the AI fit_and_predict_fn, so the harness is shared."""
  ref_date = pd.to_datetime(test_x['reference_date'].iloc[0]).date()
  locations = sorted({f'{int(x):02d}' for x in test_x['location']})
  run_config = RunConfig(
      disease=Disease.COVID,
      ref_date=ref_date,
      output_root=None,
      artifact_store_root=None,
      max_horizon=int(test_x['horizon'].max()) + 1,
      states=locations,
      hsas=[],
      q_levels=QUANTILES,
      q_labels=Q_LABELS)
  df = to_iddata_frame(train_x, train_y)
  out = GBQRFromFrame(model_config(num_bags)).run_on_frame(df, run_config)

  # Long hub format -> one row per test_x row, one column per quantile.
  out = out[out['horizon'].isin(test_x['horizon'].unique())].copy()
  out['location'] = out['location'].astype(int)
  wide = out.pivot_table(index=['location', 'horizon'],
                         columns='output_type_id', values='value')
  wide.columns = [f'quantile_{float(c):g}' for c in wide.columns]
  keys = pd.MultiIndex.from_frame(test_x[['location', 'horizon']].astype(int))
  result = wide.reindex(keys)
  result.index = test_x.index
  return result[[f'quantile_{q}' for q in QUANTILES]]
