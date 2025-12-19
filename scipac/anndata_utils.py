"""
AnnData utilities for SCIPAC - lazy loading to avoid import errors when scanpy is not installed
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Any
import warnings
from datetime import datetime


def has_scanpy() -> bool:
    """Check if scanpy is available."""
    try:
        import scanpy
        return True
    except ImportError:
        return False


def has_anndata() -> bool:
    """Check if anndata is available."""
    try:
        import anndata
        return True
    except ImportError:
        return False


def is_anndata(obj: Any) -> bool:
    """
    Check if object is an AnnData instance without importing anndata.

    Parameters
    ----------
    obj : Any
        Object to check

    Returns
    -------
    bool
        True if obj is an AnnData instance
    """
    return (
        type(obj).__name__ == 'AnnData' and
        hasattr(obj, 'obs') and
        hasattr(obj, 'var') and
        hasattr(obj, 'X')
    )


def to_dense_array(X: Any, max_elements: int = int(1e9)) -> np.ndarray:
    """
    Convert sparse matrix or array-like to dense numpy array.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data
    max_elements : int
        Warning threshold for large matrices (default: 1 billion elements)

    Returns
    -------
    np.ndarray
        Dense numpy array
    """
    try:
        from scipy import sparse
        if sparse.issparse(X):
            n_elements = X.shape[0] * X.shape[1]
            if n_elements > max_elements:
                warnings.warn(
                    f"Converting large sparse matrix ({X.shape}) to dense. "
                    f"This may use significant memory. Consider subsetting cells first."
                )
            return X.toarray()
    except ImportError:
        pass

    return np.asarray(X)


def extract_expression_matrix(
    adata: Any,
    layer: Optional[str] = None,
    use_raw: bool = False
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """
    Extract expression matrix from AnnData.

    AnnData stores data as (cells x genes), but SCIPAC expects (genes x cells).
    This function extracts and transposes the matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    layer : str, optional
        Layer to use. If None, uses adata.X
    use_raw : bool
        Whether to use adata.raw

    Returns
    -------
    Tuple[np.ndarray, pd.Index, pd.Index]
        (expression_matrix (genes x cells), gene_names, cell_names)
    """
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData. Available layers: {list(adata.layers.keys())}")
        X = adata.layers[layer]
        var_names = adata.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # Convert to dense and transpose (AnnData is cells x genes, we need genes x cells)
    X_dense = to_dense_array(X)
    X_transposed = X_dense.T  # Now genes x cells

    return X_transposed, var_names, adata.obs_names


def select_hvg_scanpy(
    adata: Any,
    n_top_genes: int = 2000,
    flavor: str = 'seurat',
    batch_key: Optional[str] = None,
    layer: Optional[str] = None,
    inplace: bool = False
) -> np.ndarray:
    """
    Select highly variable genes using scanpy.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_top_genes : int
        Number of HVGs to select
    flavor : str
        Method: 'seurat', 'cell_ranger', or 'seurat_v3'
    batch_key : str, optional
        Batch key for batch-aware HVG selection
    layer : str, optional
        Layer to use
    inplace : bool
        Whether to modify adata in place

    Returns
    -------
    np.ndarray
        Boolean array indicating HVG status for each gene
    """
    if not has_scanpy():
        raise ImportError("scanpy is required for this function. Install with: pip install scanpy")

    import scanpy as sc

    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()

    # Map our flavor names to scanpy's
    flavor_map = {
        'seurat': 'seurat',
        'cell_ranger': 'cell_ranger',
        'seurat_v3': 'seurat_v3'
    }
    sc_flavor = flavor_map.get(flavor, 'seurat')

    # Run HVG selection
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=sc_flavor,
        batch_key=batch_key,
        layer=layer
    )

    return adata.var['highly_variable'].values


def store_scipac_results(
    adata: Any,
    results: pd.DataFrame,
    cluster_results: dict,
    params: dict,
    prefix: str = 'scipac_'
) -> None:
    """
    Store SCIPAC results back into AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (modified in place)
    results : pd.DataFrame
        SCIPAC results with Lambda values
    cluster_results : dict
        Clustering results from seurat_ct
    params : dict
        Parameters used in the analysis
    prefix : str
        Prefix for column names in adata.obs
    """
    # Verify that results align with adata
    if len(results) != adata.n_obs:
        raise ValueError(
            f"Results length ({len(results)}) does not match adata.n_obs ({adata.n_obs})"
        )

    # Store per-cell results in obs
    adata.obs[f'{prefix}cluster'] = results['cluster_assignment'].values
    adata.obs[f'{prefix}lambda_est'] = results['Lambda.est'].values
    adata.obs[f'{prefix}lambda_upper'] = results['Lambda.upper'].values
    adata.obs[f'{prefix}lambda_lower'] = results['Lambda.lower'].values
    adata.obs[f'{prefix}significance'] = pd.Categorical(
        results['sig'].values,
        categories=['Sig.pos', 'Sig.neg', 'Not.sig']
    )
    adata.obs[f'{prefix}log_pval'] = results['log.pval'].values

    # Store metadata in uns
    adata.uns['scipac'] = {
        'family': params.get('family'),
        'n_clusters': cluster_results['k'],
        'cluster_centers': cluster_results['centers'],
        'n_bootstrap': params.get('bt_size'),
        'ela_net_alpha': params.get('ela_net_alpha'),
        'ci_alpha': params.get('ci_alpha'),
        'hvg_method': params.get('hvg_method'),
        'n_hvg': params.get('n_hvg'),
        'run_date': datetime.now().isoformat()
    }
