"""Prepares the fixed input for both treatments.

Real data:
  python prepare_input.py --train datasets/single-cell-batch-integration/ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad
Synthetic data (for a smoke test without downloading anything):
  python prepare_input.py --synthetic
Output: input_20k.h5ad (raw counts in .X, obs with 'batch' and 'cell_type').
"""
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def subsample_adata(adata, target_size, batch_col='batch',
                    celltype_col='cell_type', seed=42):
  """Copied from the ERA task notebook (cell 13), unchanged in logic."""
  np.random.seed(seed)
  obs = adata.obs[[batch_col, celltype_col]].copy()
  obs['index'] = obs.index
  group_counts = obs.groupby([batch_col, celltype_col], observed=True).size()
  total_counts = len(adata)
  group_proportions = group_counts / total_counts
  group_sample_sizes = (group_proportions * target_size).astype(int)
  remaining = target_size - group_sample_sizes.sum()
  if remaining != 0:
    sorted_groups = group_proportions.sort_values(ascending=False).index
    for group in sorted_groups:
      if remaining > 0:
        group_sample_sizes[group] += 1
        remaining -= 1
      elif remaining < 0:
        if group_sample_sizes[group] > 0:
          group_sample_sizes[group] -= 1
          remaining += 1
      if remaining == 0:
        break
  sampled_indices = []
  for group, size in group_sample_sizes.items():
    group_indices = obs[
        (obs[batch_col] == group[0]) & (obs[celltype_col] == group[1])
    ]['index'].values
    sampled_indices.extend(
        np.random.choice(group_indices, size=size, replace=False))
  return adata[sampled_indices, :].copy()


def load_real(path):
  # The notebook reads X from layers/counts together with obs and var.
  full = ad.read_h5ad(path)
  if 'counts' not in full.layers:
    raise SystemExit('No layers/counts in the file. Check that this is the '
                     '*-train-dataset.h5ad file from the ERA Kaggle dataset.')
  return ad.AnnData(X=full.layers['counts'], obs=full.obs.copy(),
                    var=full.var.copy())


def make_synthetic(n=20000, g=2000, n_batches=8, n_types=15, seed=0):
  rng = np.random.default_rng(seed)
  batch = rng.integers(0, n_batches, n)
  ct = rng.integers(0, n_types, n)
  mu = rng.gamma(0.5, 1, (n_types, g))
  shift = rng.lognormal(0, 0.3, (n_batches, g))
  X = sp.csr_matrix(rng.poisson(mu[ct] * shift[batch] * 2).astype(np.float32))
  obs = pd.DataFrame({
      'batch': pd.Categorical([f'b{b}' for b in batch]),
      'cell_type': pd.Categorical([f't{c}' for c in ct]),
  }, index=[f'c{i}' for i in range(n)])
  var = pd.DataFrame(index=[f'g{j}' for j in range(g)])
  return ad.AnnData(X=X, obs=obs, var=var)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--train', help='path to the ERA *-train-dataset.h5ad')
  p.add_argument('--synthetic', action='store_true')
  p.add_argument('--size', type=int, default=20000)
  p.add_argument('--seed', type=int, default=42)
  p.add_argument('--out', default='input_20k.h5ad')
  a = p.parse_args()
  if a.synthetic:
    adata = make_synthetic(n=a.size)
  elif a.train:
    adata = subsample_adata(load_real(a.train), a.size, seed=a.seed)
  else:
    raise SystemExit('Give --train <file> or --synthetic')
  adata.write_h5ad(a.out)
  print(adata)
  print('batches:', adata.obs['batch'].nunique(),
        '| cell types:', adata.obs['cell_type'].nunique())
  print('written to', a.out)
