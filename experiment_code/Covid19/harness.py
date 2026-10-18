"""Shared harness. The fold logic and the WIS are copied from the ERA task
notebook for CDC forecasting (implementation/notebooks/
flu-cornell-jhu-hierarchsir.ipynb), with the target renamed to COVID-19.
The COVID notebook itself is not released; this is its closest public form."""
import numpy as np
import pandas as pd

TARGET_STR = 'Total COVID-19 Admissions'
QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
HORIZONS = [0, 1, 2, 3]


def _create_fold_scaffold(reference_date, horizons, location_codes, locations_df):
  """Creates the feature scaffold for a single forecast date (notebook code)."""
  all_combinations = pd.MultiIndex.from_product(
      [[reference_date], horizons, location_codes],
      names=['reference_date', 'horizon', 'location'])
  scaffold = pd.DataFrame(index=all_combinations).reset_index()
  scaffold['target_end_date'] = scaffold.apply(
      lambda row: row['reference_date'] + pd.Timedelta(weeks=row['horizon']),
      axis=1)
  return scaffold.merge(locations_df, on='location', how='left')


def make_fold(dataset, locations, reference_date):
  """Returns train_x, train_y, test_x for one reference date (notebook logic)."""
  reference_date = pd.Timestamp(reference_date)
  train_end_date = reference_date - pd.Timedelta(weeks=1)
  dataset = dataset.copy()
  dataset['target_end_date'] = pd.to_datetime(dataset['target_end_date'])
  train = dataset[dataset['target_end_date'] <= train_end_date]
  train_x = train.drop(columns=[TARGET_STR])
  train_y = train[TARGET_STR]
  test_x = _create_fold_scaffold(reference_date, HORIZONS,
                                 locations['location'].unique(), locations)
  return train_x, train_y, test_x


# Copied verbatim from the notebook (cell 17).
def interval_score(
    observed: float, lower: float, upper: float, alpha: float
) -> float:
  if observed < lower:
    score = (upper - lower) + (2 / alpha) * (lower - observed)
  elif observed > upper:
    score = (upper - lower) + (2 / alpha) * (observed - upper)
  else:
    score = upper - lower
  return score


def weighted_interval_score(
    predicted: pd.DataFrame,
    observed: pd.Series,
    quantile_level=QUANTILES,
    count_median_twice=True,
) -> np.array:
  """WIS is a proper scoring rule to evaluate forecasts in an interval or quantile format.

  Smaller values are better. (See Bracher et al., 2021)

  Args:
      observed: Numeric data of n observed values.
      predicted: Numeric nxN dataframe of predictive quantiles with n forecasts
        (same as number of observed values) and N quantiles per forecast. If
        `observed` is a single number, then predicted can just be an N-vector.
      quantile_level (np.array or list): Vector of of size N with the quantile
        levels for which predictions were made.
      count_median_twice (bool, optional): If True, count the median twice in
        the score. Defaults to False.

  Returns:
      np.array: a numeric vector with WIS values of size n (one per observation)
  """

  # If 'observed' is a single value (median) (WIS simplifies to absolute error)
  if observed.size == 1:
    predicted = predicted.reshape(1, -1)

  first_half_quantiles = sorted(quantile_level)[: len(quantile_level) // 2]
  K = len(first_half_quantiles)
  # WIS for each observation
  wis_scores = []
  for i in range(len(observed)):
    y = observed.iloc[i]
    predictions = predicted.iloc[i]
    weighted_interval_scores = []
    median_forecast = predictions.iloc[predictions.shape[0] // 2]
    weighted_interval_scores.append(0.5 * abs(y - median_forecast))
    if count_median_twice:
      weighted_interval_scores.append(0.5 * abs(y - median_forecast))

    for l_q in first_half_quantiles:
      # l: the lower bound of the interval
      # u: the upper bound of the interval
      u_q = round(1 - l_q, 3)
      alpha_i = 2 * l_q
      weighted_interval_scores.append(
          (alpha_i / 2)
          * (
              interval_score(
                  y,
                  predictions[f'quantile_{l_q}'],
                  predictions[f'quantile_{u_q}'],
                  alpha_i,
              )
          )
      )
    # Weighted sum to get WIS score for observation
    wis_scores.append(np.sum(np.array(weighted_interval_scores)))
  return 1 / (K + 0.5) * np.array(wis_scores)


def score_fold(dataset, test_x, preds):
  """Mean WIS of one fold, only rows with ground truth (notebook logic)."""
  f = test_x.copy()
  for c in preds.columns:
    f[c] = preds[c].values
  d = dataset.copy()
  d['target_end_date'] = pd.to_datetime(d['target_end_date'])
  m = f.merge(d[['target_end_date', 'location', TARGET_STR]],
              on=['target_end_date', 'location'], how='inner')
  if m.empty:
    return float('nan')
  return float(np.mean(weighted_interval_score(
      m[[f'quantile_{q}' for q in QUANTILES]], m[TARGET_STR])))
