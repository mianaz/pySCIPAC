"""
PCA and batch correction functions for SCIPAC
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Union, Tuple
import warnings

try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False


def sc_bulk_pca(
    sc_dat: Union[np.ndarray, pd.DataFrame],
    bulk_dat: Union[np.ndarray, pd.DataFrame],
    do_pca_sc: bool = False,
    n_pc: int = 60,
    batch_var: Optional[np.ndarray] = None,
    verbose: bool = True
) -> dict:
    """
    Perform PCA on single-cell and bulk data with optional batch correction.
    
    Parameters
    ----------
    sc_dat : array-like
        Preprocessed single-cell data (genes x cells)
    bulk_dat : array-like
        Preprocessed bulk data (genes x samples)
    do_pca_sc : bool
        If True, perform PCA on single-cell data first; otherwise on bulk data
    n_pc : int
        Number of principal components to retain
    batch_var : array-like, optional
        Batch assignments for each cell (for batch correction)
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'sc_dat_rot': Single-cell data in PC space
        - 'bulk_dat_rot': Bulk data in PC space
        - 'rotation_matrix': PCA rotation matrix
        - 'batch_corrected': Whether batch correction was applied
    """
    
    # Convert to numpy arrays if needed
    if isinstance(sc_dat, pd.DataFrame):
        sc_arr = sc_dat.values
        cell_names = sc_dat.columns
    else:
        sc_arr = sc_dat
        cell_names = None
        
    if isinstance(bulk_dat, pd.DataFrame):
        bulk_arr = bulk_dat.values
        sample_names = bulk_dat.columns
    else:
        bulk_arr = bulk_dat
        sample_names = None
    
    # Check dimensions
    if sc_arr.shape[0] != bulk_arr.shape[0]:
        raise ValueError("Number of genes must match between single-cell and bulk data")
    
    # Check batch_var if provided
    if batch_var is not None:
        if not HARMONY_AVAILABLE:
            raise ImportError("harmonypy is required for batch correction. Install with: pip install harmonypy")
        
        if len(batch_var) != sc_arr.shape[1]:
            raise ValueError("Length of batch_var must match the number of cells in sc_dat")
    
    # Transpose for sklearn (samples x features)
    sc_arr_t = sc_arr.T
    bulk_arr_t = bulk_arr.T
    
    # Center the data
    sc_centered = sc_arr_t - np.mean(sc_arr_t, axis=0)
    bulk_centered = bulk_arr_t - np.mean(bulk_arr_t, axis=0)
    
    if do_pca_sc:
        if verbose:
            print("Performing PCA on single-cell data...")
        
        # Fit PCA on single-cell data
        pca = PCA(n_components=n_pc)
        sc_pca = pca.fit_transform(sc_centered)
        
        # Project bulk data using the same transformation
        bulk_pca = pca.transform(bulk_centered)
        rotation_matrix = pca.components_.T
        
    else:
        if verbose:
            print("Performing PCA on bulk data...")
        
        # Fit PCA on bulk data
        pca = PCA(n_components=n_pc)
        bulk_pca = pca.fit_transform(bulk_centered)
        
        # Project single-cell data using the same transformation
        sc_pca = pca.transform(sc_centered)
        rotation_matrix = pca.components_.T
    
    # Apply batch correction if requested
    batch_corrected = False
    if batch_var is not None:
        if verbose:
            print("Applying harmony batch correction...")
        
        # Create metadata dataframe for harmony
        meta_data = pd.DataFrame({'batch': batch_var})
        if cell_names is not None:
            meta_data.index = cell_names
        
        # Run harmony
        ho = hm.run_harmony(sc_pca, meta_data, 'batch', verbose=verbose)
        sc_pca = ho.Z_corr.T  # Harmony returns cells x PCs, transpose back
        batch_corrected = True
        
        if verbose:
            print("Batch correction completed.")
    
    # Create result dictionary
    result = {
        'sc_dat_rot': sc_pca,
        'bulk_dat_rot': bulk_pca,
        'rotation_matrix': rotation_matrix,
        'batch_corrected': batch_corrected
    }
    
    # Add cell and sample names if available
    if cell_names is not None:
        result['cell_names'] = cell_names
    if sample_names is not None:
        result['sample_names'] = sample_names
    
    return result