"""
Clustering functions for SCIPAC
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from typing import Optional, Union
import warnings


def seurat_ct(
    sc_dat_rot: Union[np.ndarray, pd.DataFrame],
    res: float = 0.8,
    n_neighbors: int = 20,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Perform Seurat-like clustering on dimensionality-reduced single-cell data.
    
    This function mimics Seurat's clustering approach using Louvain/Leiden-like
    clustering based on a k-nearest neighbor graph.
    
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
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'k': Number of clusters
        - 'ct_assignment': DataFrame with cluster assignments
        - 'centers': Cluster centroids (PCs x clusters)
    """
    
    # Convert to numpy array if needed
    if isinstance(sc_dat_rot, pd.DataFrame):
        data = sc_dat_rot.values
        cell_names = sc_dat_rot.index
    else:
        data = sc_dat_rot
        cell_names = [f"Cell_{i}" for i in range(data.shape[0])]
    
    n_cells, n_pcs = data.shape
    
    if verbose:
        print(f"Clustering {n_cells} cells using {n_pcs} PCs...")
    
    # For simplicity, we'll use KMeans clustering with the number of clusters
    # determined by the resolution parameter
    # In a full implementation, this would use Louvain/Leiden on a KNN graph
    
    # Estimate number of clusters based on resolution and data size
    # This is a simplified heuristic
    n_clusters = max(2, min(int(res * np.sqrt(n_cells / 100)), n_cells // 10))
    
    if verbose:
        print(f"Using {n_clusters} clusters (resolution={res})")
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
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
        print(f"Clustering complete. Found {n_clusters} clusters.")
        for k in range(n_clusters):
            n_cells_k = np.sum(cluster_labels == k)
            print(f"  Cluster {k}: {n_cells_k} cells")
    
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
    Alternative clustering using Leiden algorithm (requires python-igraph and leidenalg).
    
    This is closer to Seurat's actual implementation but requires additional dependencies.
    """
    try:
        import igraph as ig
        import leidenalg
        LEIDEN_AVAILABLE = True
    except ImportError:
        LEIDEN_AVAILABLE = False
        warnings.warn("python-igraph and leidenalg not available. Using KMeans clustering instead.")
        return seurat_ct(sc_dat_rot, res, n_neighbors, random_state, verbose)
    
    # Convert to numpy array if needed
    if isinstance(sc_dat_rot, pd.DataFrame):
        data = sc_dat_rot.values
        cell_names = sc_dat_rot.index
    else:
        data = sc_dat_rot
        cell_names = [f"Cell_{i}" for i in range(data.shape[0])]
    
    n_cells, n_pcs = data.shape
    
    if verbose:
        print(f"Leiden clustering {n_cells} cells using {n_pcs} PCs...")
    
    # Build KNN graph
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    
    # Create adjacency matrix
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    weights = 1.0 / (1.0 + distances.flatten())  # Convert distances to weights
    
    # Create igraph from adjacency matrix
    edges = list(zip(rows, cols))
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    
    # Run Leiden clustering
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        seed=random_state,
        weights='weight'
    )
    
    cluster_labels = np.array(partition.membership)
    n_clusters = len(set(cluster_labels))
    
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
        print(f"Clustering complete. Found {n_clusters} clusters.")
    
    return {
        'k': n_clusters,
        'ct_assignment': ct_assignment,
        'centers': centers
    }