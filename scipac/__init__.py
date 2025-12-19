"""
SCIPAC: Single Cell Identifier of Phenotype-Associated Cells
Python implementation of the SCIPAC R package
"""

from .core import SCIPAC
from .preprocessing import preprocess_sc_bulk_dat
from .pca import sc_bulk_pca
from .clustering import seurat_ct, leiden_clustering

__version__ = "0.1.0"
__all__ = ["SCIPAC", "preprocess_sc_bulk_dat", "sc_bulk_pca", "seurat_ct", "leiden_clustering"]