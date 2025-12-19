"""
Preprocessing functions for SCIPAC
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Any
import warnings


def preprocess_sc_bulk_dat(
    sc_dat: Union[np.ndarray, pd.DataFrame, Any],
    bulk_dat: Union[np.ndarray, pd.DataFrame],
    hvg: int = 2000,
    hvg_method: str = 'cv2',
    batch_key: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Preprocess single-cell and bulk RNA-seq data for SCIPAC analysis.

    Parameters
    ----------
    sc_dat : array-like or AnnData
        Single-cell expression matrix (genes x cells) or AnnData object (cells x genes).
        If AnnData is provided, the matrix will be automatically transposed.
    bulk_dat : array-like
        Bulk expression matrix (genes x samples)
    hvg : int
        Number of highly variable genes to select
    hvg_method : str
        HVG selection method:
        - 'cv2': Coefficient of variation squared (default, works without scanpy)
        - 'seurat': Seurat v2 method (requires scanpy)
        - 'cell_ranger': Cell Ranger method (requires scanpy)
        - 'seurat_v3': Seurat v3 method (requires scanpy, uses raw counts)
    batch_key : str, optional
        Batch key for batch-aware HVG selection (only with scanpy methods).
        Must be a column name in adata.obs if using AnnData input.
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Dictionary containing:
        - 'sc_dat_preprocessed': Preprocessed single-cell data (genes x cells)
        - 'bulk_dat_preprocessed': Preprocessed bulk data (genes x samples)
        - 'gene_names': Selected gene names
        - 'hvg_method': HVG method used
        - 'adata': Original AnnData object if provided (for storing results later)
    """
    from .anndata_utils import is_anndata, has_scanpy, extract_expression_matrix, select_hvg_scanpy

    result = {}
    adata_input = None
    scanpy_hvg_genes = None

    # Handle AnnData input
    if is_anndata(sc_dat):
        adata_input = sc_dat
        result['adata'] = adata_input

        if verbose:
            print(f"Input: AnnData with {sc_dat.n_obs} cells and {sc_dat.n_vars} genes")

        # Extract expression matrix (returns genes x cells)
        sc_arr, gene_names_sc, cell_names = extract_expression_matrix(sc_dat)
        sc_df = pd.DataFrame(sc_arr, index=gene_names_sc, columns=cell_names)

        # Handle HVG selection with scanpy if available and method is scanpy-based
        if hvg_method in ['seurat', 'cell_ranger', 'seurat_v3']:
            if has_scanpy():
                if verbose:
                    print(f"Using scanpy for HVG selection (method='{hvg_method}')")
                hvg_mask = select_hvg_scanpy(
                    sc_dat, n_top_genes=hvg, flavor=hvg_method, batch_key=batch_key
                )
                scanpy_hvg_genes = sc_dat.var_names[hvg_mask]
            else:
                warnings.warn(
                    f"scanpy not available for hvg_method='{hvg_method}'. "
                    f"Falling back to 'cv2' method. Install scanpy: pip install scanpy"
                )
                hvg_method = 'cv2'
    else:
        # Convert numpy to DataFrame
        if isinstance(sc_dat, np.ndarray):
            sc_df = pd.DataFrame(sc_dat)
        else:
            sc_df = sc_dat.copy()

        # Warn if scanpy method requested without AnnData
        if hvg_method in ['seurat', 'cell_ranger', 'seurat_v3']:
            warnings.warn(
                f"hvg_method='{hvg_method}' requires AnnData input for optimal results. "
                f"Falling back to 'cv2' method."
            )
            hvg_method = 'cv2'

    # Handle bulk data
    if isinstance(bulk_dat, np.ndarray):
        bulk_df = pd.DataFrame(bulk_dat)
    else:
        bulk_df = bulk_dat.copy()

    if verbose:
        print(f"Input dimensions - SC: {sc_df.shape}, Bulk: {bulk_df.shape}")

    # Find common genes
    common_genes = sc_df.index.intersection(bulk_df.index)

    if len(common_genes) == 0:
        raise ValueError(
            "No common genes found between single-cell and bulk data. "
            "Ensure both datasets have gene names as row indices."
        )

    if verbose:
        print(f"Found {len(common_genes)} common genes")

    # Subset to common genes
    sc_common = sc_df.loc[common_genes]
    bulk_common = bulk_df.loc[common_genes]

    # Normalize single-cell data (library size normalization + log1p)
    sc_normalized = sc_common.copy()

    # Library size normalization for single cells
    total_counts = sc_normalized.sum(axis=0)
    # Handle cells with zero total counts to avoid divide-by-zero
    total_counts = total_counts.replace(0, 1)
    sc_normalized = sc_normalized.divide(total_counts, axis=1) * 10000
    sc_normalized = np.log1p(sc_normalized)

    # Log-transform bulk data (assuming it's already normalized, e.g., TPM or FPKM)
    # Clip negative values (shouldn't exist but handle edge cases)
    bulk_common = bulk_common.clip(lower=0)
    bulk_normalized = np.log1p(bulk_common)

    # HVG selection
    if hvg_method == 'cv2':
        # CV2 method (default, works without scanpy)
        if verbose:
            print(f"Selecting HVGs using CV2 method")
        gene_means = sc_normalized.mean(axis=1)
        gene_vars = sc_normalized.var(axis=1)
        gene_cv2 = gene_vars / (gene_means ** 2 + 1e-10)
        n_hvgs = min(hvg, len(gene_means))
        hvg_genes = gene_cv2.nlargest(n_hvgs).index
    elif scanpy_hvg_genes is not None:
        # Use scanpy-selected HVGs (already computed above for AnnData)
        hvg_genes = scanpy_hvg_genes.intersection(common_genes)
        n_hvgs = len(hvg_genes)
        if n_hvgs < hvg:
            if verbose:
                print(f"Note: Only {n_hvgs} of {hvg} scanpy HVGs found in common genes")
    else:
        # Fallback to CV2 if something went wrong
        if verbose:
            print(f"Selecting HVGs using CV2 method (fallback)")
        gene_means = sc_normalized.mean(axis=1)
        gene_vars = sc_normalized.var(axis=1)
        gene_cv2 = gene_vars / (gene_means ** 2 + 1e-10)
        n_hvgs = min(hvg, len(gene_means))
        hvg_genes = gene_cv2.nlargest(n_hvgs).index

    if verbose:
        print(f"Selected {len(hvg_genes)} highly variable genes")

    # Subset to HVGs
    sc_hvg = sc_normalized.loc[hvg_genes]
    bulk_hvg = bulk_normalized.loc[hvg_genes]

    # Scale the data (z-score normalization)
    sc_scaled = (sc_hvg - sc_hvg.mean(axis=1).values.reshape(-1, 1)) / (sc_hvg.std(axis=1).values.reshape(-1, 1) + 1e-10)
    bulk_scaled = (bulk_hvg - bulk_hvg.mean(axis=1).values.reshape(-1, 1)) / (bulk_hvg.std(axis=1).values.reshape(-1, 1) + 1e-10)

    result.update({
        'sc_dat_preprocessed': sc_scaled,
        'bulk_dat_preprocessed': bulk_scaled,
        'gene_names': list(hvg_genes),
        'hvg_method': hvg_method
    })

    return result
