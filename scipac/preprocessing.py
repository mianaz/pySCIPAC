"""
Preprocessing functions for SCIPAC
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import warnings


def preprocess_sc_bulk_dat(
    sc_dat: Union[np.ndarray, pd.DataFrame],
    bulk_dat: Union[np.ndarray, pd.DataFrame],
    hvg: int = 2000,
    verbose: bool = True
) -> dict:
    """
    Preprocess single-cell and bulk RNA-seq data for SCIPAC analysis.
    
    Parameters
    ----------
    sc_dat : array-like
        Single-cell expression matrix (genes x cells)
    bulk_dat : array-like
        Bulk expression matrix (genes x samples)
    hvg : int
        Number of highly variable genes to select
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'sc_dat_preprocessed': Preprocessed single-cell data
        - 'bulk_dat_preprocessed': Preprocessed bulk data
        - 'gene_names': Selected gene names
    """
    
    # Convert to pandas DataFrames if necessary
    if isinstance(sc_dat, np.ndarray):
        sc_df = pd.DataFrame(sc_dat)
    else:
        sc_df = sc_dat.copy()
        
    if isinstance(bulk_dat, np.ndarray):
        bulk_df = pd.DataFrame(bulk_dat)
    else:
        bulk_df = bulk_dat.copy()
    
    if verbose:
        print(f"Input dimensions - SC: {sc_df.shape}, Bulk: {bulk_df.shape}")
    
    # Find common genes
    common_genes = sc_df.index.intersection(bulk_df.index)
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between single-cell and bulk data")
    
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
    
    # Select highly variable genes from single-cell data
    # Calculate mean and variance for each gene
    gene_means = sc_normalized.mean(axis=1)
    gene_vars = sc_normalized.var(axis=1)
    
    # Calculate coefficient of variation
    gene_cv2 = gene_vars / (gene_means ** 2 + 1e-10)
    
    # Select top HVGs
    n_hvgs = min(hvg, len(gene_means))
    hvg_indices = gene_cv2.nlargest(n_hvgs).index
    
    if verbose:
        print(f"Selected {n_hvgs} highly variable genes")
    
    # Subset to HVGs
    sc_hvg = sc_normalized.loc[hvg_indices]
    bulk_hvg = bulk_normalized.loc[hvg_indices]
    
    # Scale the data (z-score normalization)
    sc_scaled = (sc_hvg - sc_hvg.mean(axis=1).values.reshape(-1, 1)) / (sc_hvg.std(axis=1).values.reshape(-1, 1) + 1e-10)
    bulk_scaled = (bulk_hvg - bulk_hvg.mean(axis=1).values.reshape(-1, 1)) / (bulk_hvg.std(axis=1).values.reshape(-1, 1) + 1e-10)
    
    return {
        'sc_dat_preprocessed': sc_scaled,
        'bulk_dat_preprocessed': bulk_scaled,
        'gene_names': hvg_indices.tolist()
    }