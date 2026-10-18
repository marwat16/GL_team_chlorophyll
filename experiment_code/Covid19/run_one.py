"""Runs ONE treatment once, for one reference date. This is the unit that
Experiment Runner and EnergiBridge will call later, once per run.

  python run_one.py --treatment ai
  python run_one.py --treatment human
  python run_one.py --treatment human --load-only   (loading baseline)

Loading data, building the fold and importing libraries happen before the
measured section and are reported separately. The WIS is computed after the
measured section, only as a check that the forecast is sensible.
"""
import argparse
import json
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TQDM_DISABLE', '1')

import pandas as pd

DEFAULT_REF_DATE = '2025-02-01'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--treatment', choices=['ai', 'human'], required=True)
  p.add_argument('--ref-date', default=DEFAULT_REF_DATE)
  p.add_argument('--ai-config', type=int, default=0,
                 help='index in config_list of node 1937 (0 = first config)')
  p.add_argument('--num-bags', type=int, default=100,
                 help='human only, 100 is the operational value')
  p.add_argument('--load-only', action='store_true')
  a = p.parse_args()

  t0 = time.perf_counter()
  from harness import make_fold, score_fold
  dataset = pd.read_csv('dataset.csv')
  locations = pd.read_csv('locations.csv')
  train_x, train_y, test_x = make_fold(dataset, locations, a.ref_date)
  if a.treatment == 'ai':
    from ai_node1937 import fit_and_predict_fn, config_list
  else:
    from human_gbqr import fit_and_predict_human
  t_load = time.perf_counter() - t0

  if a.load_only:
    print(json.dumps({'treatment': a.treatment, 'load_s': round(t_load, 3)}))
    return

  t1 = time.perf_counter()
  if a.treatment == 'ai':
    preds = fit_and_predict_fn(train_x.copy(), train_y.copy(), test_x.copy(),
                               config_list[a.ai_config])
  else:
    preds = fit_and_predict_human(train_x.copy(), train_y.copy(), test_x.copy(),
                                  num_bags=a.num_bags)
  t_method = time.perf_counter() - t1

  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  peak_mb = rss / (1024 * 1024) if sys.platform == 'darwin' else rss / 1024
  qcols = [c for c in preds.columns if c.startswith('quantile_')]
  print(json.dumps({
      'treatment': a.treatment,
      'ref_date': a.ref_date,
      'load_s': round(t_load, 3),
      'method_s': round(t_method, 3),
      'peak_rss_mb': round(peak_mb, 1),
      'n_rows': int(len(preds)),
      'n_quantiles': len(qcols),
      'any_nan': bool(preds[qcols].isna().any().any()),
      'monotone': bool((preds[qcols].astype(float).diff(axis=1).iloc[:, 1:] >= -1e-9).all().all()),
      'wis_check': round(score_fold(dataset, test_x, preds[qcols].astype(float)), 2),
  }))


if __name__ == '__main__':
  main()
