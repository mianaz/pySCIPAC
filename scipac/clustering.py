"""
Clustering functions for SCIPAC

Implements scanpy-like clustering using neighbor graphs and Leiden/Louvain algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Optional, Union
import warnings


def seurat_ct(
    sc_dat_rot: Union[np.ndarray, pd.DataFrame],
    res: float = 0.8,
    n_neighbors: int = 20,
    random_state: int = 42,
    verbose: bool = True,
    method: str = 'auto',
    algorithm: str = 'leiden'
) -> dict:
    """
    Perform scanpy-like clustering on dimensionality-reduced single-cell data.

    Uses scanpy's neighbor graph construction (UMAP-style fuzzy simplicial set)
    followed by Leiden or Louvain community detection.

    Parameters
    ----------
    sc_dat_rot : array-like
        Dimensionality-reduced single-cell data (cells x PCs)
    res : float
        Resolution parameter for clustering (higher = more clusters)
    n_neighbors : int
        Number of nearest neighbors for graph construction
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress messages
    method : str
        Clustering method: 'auto' (try scanpy, fall back to KMeans),
        'scanpy', or 'kmeans'
    algorithm : str
        Community detection algorithm: 'leiden' (default) or 'louvain'

    Returns
    -------
    dict
        Dictionary containing:
        - 'k': Number of clusters
        - 'ct_assignment': DataFrame with cluster assignments
        - 'centers': Cluster centroids (PCs x clusters)
    """
    # Check if scanpy-based clustering is available
    try:
        import scanpy as sc
        import anndata as ad
        SCANPY_AVAILABLE = True
    except ImportError:
        SCANPY_AVAILABLE = False

    # Determine which method to use
    if method == 'auto':
        use_scanpy = SCANPY_AVAILABLE
    elif method == 'scanpy' or method == 'graph':
        if not SCANPY_AVAILABLE:
            raise ImportError(
                "Scanpy-based clustering requires scanpy. "
                "Install with: pip install scanpy"
            )
        use_scanpy = True
    elif method == 'kmeans':
        use_scanpy = False
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'auto', 'scanpy', or 'kmeans'")

    # Validate algorithm
    if algorithm not in ['leiden', 'louvain']:
        raise ValueError(f"Invalid algorithm: {algorithm}. Choose 'leiden' or 'louvain'")

    # Convert to numpy array if needed
    if isinstance(sc_dat_rot, pd.DataFrame):
        data = sc_dat_rot.values
        cell_names = list(sc_dat_rot.index)
    else:
        data = np.asarray(sc_dat_rot)
        cell_names = [f"Cell_{i}" for i in range(data.shape[0])]

    n_cells, n_pcs = data.shape

    if use_scanpy:
        if verbose:
            print(f"Building neighbor graph for {n_cells} cells (k={n_neighbors})...")

        # Create AnnData object
        adata = ad.AnnData(X=data)
        adata.obsm['X_pca'] = data

        # Compute neighbors using scanpy (UMAP-style fuzzy simplicial set)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca', random_state=random_state)

        if verbose:
            n_edges = adata.obsp['connectivities'].nnz // 2
            print(f"Graph has {n_edges} edges")
            print(f"Running {algorithm.capitalize()} clustering (resolution={res})...")

        # Run community detection
        if algorithm == 'leiden':
            sc.tl.leiden(adata, resolution=res, random_state=random_state)
            cluster_labels = adata.obs['leiden'].astype(int).values
        else:  # louvain
            sc.tl.louvain(adata, resolution=res, random_state=random_state)
            cluster_labels = adata.obs['louvain'].astype(int).values

        n_clusters = len(np.unique(cluster_labels))

        if verbose:
            print(f"{algorithm.capitalize()} found {n_clusters} clusters")

    else:
        # Fallback to KMeans clustering
        if verbose:
            if method == 'auto':
                print("Note: Scanpy not available, using KMeans.")
                print("For graph-based clustering, install: pip install scanpy")
            print(f"KMeans clustering {n_cells} cells using {n_pcs} PCs...")

        # Estimate number of clusters based on resolution and data size
        n_clusters = max(2, min(int(res * np.sqrt(n_cells / 100)), n_cells // 10))

        if verbose:
            print(f"Using {n_clusters} clusters (resolution={res})")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

    # Remap cluster labels to be consecutive starting from 0
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels = np.array([label_map[l] for l in cluster_labels])

    # Check for empty clusters and warn
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    empty_clusters = [k for k in range(n_clusters) if cluster_counts.get(k, 0) == 0]
    if empty_clusters:
        warnings.warn(f"Empty clusters detected: {empty_clusters}. This may indicate issues with clustering parameters.")

    # Calculate cluster centroids
    centers = np.zeros((n_pcs, n_clusters))
    for k in range(n_clusters):
        cluster_cells = data[cluster_labels == k]
        if len(cluster_cells) > 0:
            centers[:, k] = np.mean(cluster_cells, axis=0)

    # Create cluster assignment dataframe
    ct_assignment = pd.DataFrame({
        'cluster_assignment': cluster_labels
    }, index=cell_names)

    if verbose:
        print("Cluster sizes:")
        for k in range(min(n_clusters, 20)):
            print(f"  Cluster {k}: {cluster_counts.get(k, 0)} cells")
        if n_clusters > 20:
            print(f"  ... and {n_clusters - 20} more clusters")

    return {
        'k': n_clusters,
        'ct_assignment': ct_assignment,
        'centers': centers
    }


def leiden_clustering(
    sc_dat_rot: Union[np.ndarray, pd.DataFrame],
    res: float = 0.8,
    n_neighbors: int = 20,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Clustering using Leiden algorithm (requires scanpy).

    Parameters
    ----------
    sc_dat_rot : array-like
        Dimensionality-reduced single-cell data (cells x PCs)
    res : float
        Resolution parameter for clustering
    n_neighbors : int
        Number of nearest neighbors for graph construction
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Clustering results with 'k', 'ct_assignment', and 'centers'
    """
    return seurat_ct(
        sc_dat_rot,
        res=res,
        n_neighbors=n_neighbors,
        random_state=random_state,
        verbose=verbose,
        method='scanpy',
        algorithm='leiden'
    )


def louvain_clustering(
    sc_dat_rot: Union[np.ndarray, pd.DataFrame],
    res: float = 0.8,
    n_neighbors: int = 20,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Clustering using Louvain algorithm (requires scanpy).

    Parameters
    ----------
    sc_dat_rot : array-like
        Dimensionality-reduced single-cell data (cells x PCs)
    res : float
        Resolution parameter for clustering
    n_neighbors : int
        Number of nearest neighbors for graph construction
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Clustering results with 'k', 'ct_assignment', and 'centers'
    """
    return seurat_ct(
        sc_dat_rot,
        res=res,
        n_neighbors=n_neighbors,
        random_state=random_state,
        verbose=verbose,
        method='scanpy',
        algorithm='louvain'
    )
