"""Prepares the fixed input for both treatments.

Real data (recommended). Rebuilds the data ERA used, NHSN COVID-19 hospital
admissions as available on 30 April 2025 (the last CovidHub snapshot before
the paper's cut-off of 1 May 2025), missing values replaced by zero, and the
static populations of the CovidHub repository:
  python prepare_input.py --hub-dir covid19-forecast-hub
where covid19-forecast-hub is a clone of github.com/CDCgov/covid19-forecast-hub
(only target-data/ and auxiliary-data/ are needed, see README).

Synthetic data (smoke test without downloading anything):
  python prepare_input.py --synthetic

Output: dataset.csv and locations.csv, in the layout of the ERA task notebook.
"""
import argparse
import numpy as np
import pandas as pd

TARGET = 'Total COVID-19 Admissions'
AS_OF = '2025-04-30'


def load_real(hub_dir, as_of=AS_OF):
  ts = pd.read_parquet(f'{hub_dir}/target-data/time-series.parquet')
  ts = ts[(ts['target'] == 'wk inc covid hosp') & (ts['as_of'].astype(str) == as_of)]
  if ts.empty:
    raise SystemExit(f'No hospital data with as_of {as_of} in time-series.parquet')
  loc = pd.read_csv(f'{hub_dir}/auxiliary-data/locations.csv', dtype={'location': str})
  loc = loc[loc['location'] != 'US']  # ERA forecasts the 52 jurisdictions only
  ts = ts[ts['location'].isin(loc['location'])]
  # Complete grid of weeks x locations, missing values replaced by zero (paper, Methods)
  weeks = pd.to_datetime(sorted(ts['target_end_date'].astype(str).unique()))
  grid = pd.MultiIndex.from_product([weeks, loc['location']],
                                    names=['target_end_date', 'location']).to_frame(index=False)
  ts = ts.assign(target_end_date=pd.to_datetime(ts['target_end_date'].astype(str)))
  d = grid.merge(ts[['target_end_date', 'location', 'observation']],
                 on=['target_end_date', 'location'], how='left')
  d[TARGET] = d['observation'].fillna(0).astype(int)
  d = d.merge(loc[['location', 'location_name', 'population']], on='location')
  d['location'] = d['location'].astype(int)
  locations = loc[['location', 'location_name', 'population']].copy()
  locations['location'] = locations['location'].astype(int)
  return d[['target_end_date', 'location', 'location_name', 'population', TARGET]], locations


def make_synthetic(n_loc=52, start='2020-08-08', end='2025-04-26', seed=0):
  rng = np.random.default_rng(seed)
  weeks = pd.date_range(start, end, freq='7D')
  t = np.arange(len(weeks))
  rows, locs = [], []
  for i in range(n_loc):
    pop = float(rng.integers(500_000, 39_000_000))
    rate = 2 + 8 * (1 + np.sin(2 * np.pi * (t / 52) + rng.uniform(0, 1))) \
        * np.exp(-t / 300) + rng.gamma(2, 1, len(t))
    y = rng.poisson(rate * pop / 1e5)
    code = i + 1
    locs.append((code, f'Loc{code}', pop))
    rows += [(w, code, f'Loc{code}', pop, int(v)) for w, v in zip(weeks, y)]
  d = pd.DataFrame(rows, columns=['target_end_date', 'location', 'location_name',
                                  'population', TARGET])
  return d, pd.DataFrame(locs, columns=['location', 'location_name', 'population'])


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--hub-dir', help='clone of CDCgov/covid19-forecast-hub')
  p.add_argument('--synthetic', action='store_true')
  p.add_argument('--as-of', default=AS_OF)
  a = p.parse_args()
  if a.synthetic:
    d, loc = make_synthetic()
  elif a.hub_dir:
    d, loc = load_real(a.hub_dir, a.as_of)
  else:
    raise SystemExit('Give --hub-dir <dir> or --synthetic')
  d['target_end_date'] = pd.to_datetime(d['target_end_date']).dt.strftime('%Y-%m-%d')
  d.to_csv('dataset.csv', index=False)
  loc.to_csv('locations.csv', index=False)
  print(f'rows: {len(d)} | locations: {d.location.nunique()} | '
        f'weeks: {d.target_end_date.nunique()} '
        f'({d.target_end_date.min()} to {d.target_end_date.max()})')
  print('written to dataset.csv and locations.csv')
