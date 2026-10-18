"""Runs ONE treatment once. This is the unit that Experiment Runner and
EnergiBridge will call later, once per run.

  python run_one.py --treatment ai    --input input_20k.h5ad
  python run_one.py --treatment human --input input_20k.h5ad

Loading the data happens before the measured section and is reported
separately, so its cost can be subtracted (or measured with --load-only).
"""
import argparse
import json
import resource
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import anndata as ad


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--treatment', choices=['ai', 'human'], required=True)
  p.add_argument('--input', default='input_20k.h5ad')
  p.add_argument('--load-only', action='store_true',
                 help='only load data and imports (baseline for subtraction)')
  a = p.parse_args()

  t0 = time.perf_counter()
  adata = ad.read_h5ad(a.input)
  # As in the notebook, the method never sees the cell type labels.
  if 'cell_type' in adata.obs:
    adata.obs.pop('cell_type')
  if a.treatment == 'ai':
    from ai_node429 import eliminate_batch_effect_fn, config
  else:
    from human_bbknn import bbknn_human
  t_load = time.perf_counter() - t0

  if a.load_only:
    print(json.dumps({'treatment': a.treatment, 'load_s': round(t_load, 3)}))
    return

  t1 = time.perf_counter()
  if a.treatment == 'ai':
    out = eliminate_batch_effect_fn(adata, config)
  else:
    out = bbknn_human(adata)
  t_method = time.perf_counter() - t1

  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  peak_mb = rss / (1024 * 1024) if sys.platform == 'darwin' else rss / 1024
  print(json.dumps({
      'treatment': a.treatment,
      'load_s': round(t_load, 3),
      'method_s': round(t_method, 3),
      'peak_rss_mb': round(peak_mb, 1),
      'n_obs': int(out.n_obs),
      'has_embedding': 'X_emb' in out.obsm,
      'has_graph': 'connectivities' in out.obsp,
      'graph_edges': int(out.obsp['connectivities'].nnz),
  }))


if __name__ == '__main__':
  main()
