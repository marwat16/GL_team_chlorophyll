# Human treatment. BBKNN as run by OpenProblems
# (openproblems-bio/task_batch_integration, src/methods/bbknn/script.py),
# with the VIASH and file I/O parts removed and the OpenProblems
# normalization step (log_cp10k) added in front, so that both treatments
# start from the same raw counts.
import anndata as ad
import scanpy as sc
import bbknn

# OpenProblems defaults from src/methods/bbknn/config.vsh.yaml
DEFAULT_PAR = {
    'annoy_n_trees': 10,
    'neighbors_within_batch': 3,
    'n_hvg': 2000,
}


def normalize_log_cp10k(adata: ad.AnnData) -> ad.AnnData:
  """OpenProblems log_cp10k normalization (counts per 10k, then log1p)."""
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  return adata


def bbknn_human(adata: ad.AnnData, par: dict = DEFAULT_PAR) -> ad.AnnData:
  """Runs normalization and the OpenProblems BBKNN method on raw counts."""
  adata = adata.copy()
  normalize_log_cp10k(adata)

  # Original script. Select top n_hvg genes by hvg_score, then PCA.
  # The ERA data is already restricted to 2000 HVGs. If hvg_score is not
  # present, all genes are kept, which is equivalent in that case.
  if par['n_hvg']:
    if 'hvg_score' in adata.var:
      idx = adata.var['hvg_score'].to_numpy().argsort()[::-1][:par['n_hvg']]
      adata = adata[:, idx].copy()
    sc.pp.pca(adata)

  ad_bbknn = bbknn.bbknn(
      adata,
      batch_key='batch',
      copy=True,
      annoy_n_trees=par['annoy_n_trees'],
      neighbors_within_batch=par['neighbors_within_batch'],
  )

  # Same output structure as the original script (graph only).
  return ad.AnnData(
      obs=adata.obs[[]],
      var=adata.var[[]],
      obsp={
          'connectivities': ad_bbknn.obsp['connectivities'],
          'distances': ad_bbknn.obsp['distances'],
      },
      uns={'neighbors': ad_bbknn.uns['neighbors']},
  )
