# pySCIPAC

Python implementation of SCIPAC (Single Cell Identifier of Phenotype-Associated Cells).

## Overview

pySCIPAC identifies single cells associated with clinical phenotypes by integrating single-cell RNA-seq with bulk RNA-seq data and clinical outcomes. This Python implementation provides the same functionality as the [original R package](https://github.com/RavenGan/SCIPAC) with additional features for batch correction and parallel processing.

## Features

- **Multi-phenotype support**: Binary, continuous, ordinal, and survival outcomes
- **Batch correction**: Integrated Harmony support for removing batch effects
- **Parallel processing**: Bootstrap parallelization with stratified sampling
- **Graph-based clustering**: Leiden algorithm (Seurat-like) with KMeans fallback
- **Proper ordinal regression**: Uses `mord` package for proportional odds model
- **Memory efficient**: Handles large-scale single-cell datasets

## Installation

### Requirements
- Python 3.8+
- NumPy, Pandas, Scikit-learn, SciPy

### Method 1: Install from PyPI (when available)
```bash
pip install pyscipac

# With all optional dependencies
pip install pyscipac[all]
```

### Method 2: Install from GitHub
```bash
pip install git+https://github.com/mianaz/pySCIPAC.git

# With optional dependencies
pip install "pyscipac[all] @ git+https://github.com/mianaz/pySCIPAC.git"
```

### Method 3: Clone and install locally
```bash
git clone https://github.com/mianaz/pySCIPAC.git
cd pySCIPAC

# Standard installation
pip install .

# Editable installation (for development)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

### Method 4: Install with all dependencies for development
```bash
git clone https://github.com/mianaz/pySCIPAC.git
cd pySCIPAC
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Optional dependency groups

Install specific features as needed:

```bash
# Leiden clustering (recommended, matches Seurat)
pip install pyscipac[leiden]

# Ordinal regression
pip install pyscipac[ordinal]

# Survival analysis (Cox regression)
pip install pyscipac[survival]

# Batch correction (Harmony)
pip install pyscipac[batch]

# Scanpy integration
pip install pyscipac[scanpy]

# Visualization
pip install pyscipac[viz]

# Everything
pip install pyscipac[all]

# Multiple groups
pip install pyscipac[leiden,survival,viz]
```

## Quick Start

```python
from scipac import SCIPAC, preprocess_sc_bulk_dat, sc_bulk_pca, seurat_ct

# 1. Load your data
# sc_data: genes x cells expression matrix
# bulk_data: genes x samples expression matrix
# y: phenotype labels or survival data

# 2. Preprocess data
prep_res = preprocess_sc_bulk_dat(sc_data, bulk_data, hvg=1000)
sc_prep = prep_res['sc_dat_preprocessed']
bulk_prep = prep_res['bulk_dat_preprocessed']

# 3. PCA with optional batch correction
pca_res = sc_bulk_pca(
    sc_prep, bulk_prep, 
    do_pca_sc=False,  # Use bulk PCA space
    n_pc=60,
    batch_var=batch_labels  # Optional: array of batch IDs for each cell
)

# 4. Cluster cells
ct_res = seurat_ct(pca_res['sc_dat_rot'], res=0.8)

# 5. Run SCIPAC
results = SCIPAC(
    bulk_dat=pca_res['bulk_dat_rot'],
    y=y,  # Phenotype labels
    family='binomial',  # or 'gaussian', 'cox', 'cumulative'
    ct_res=ct_res,
    ela_net_alpha=0.4,
    bt_size=50,
    n_jobs=-1  # Use all CPU cores
)

# Results include Lambda values and significance for each cell
print(f"Significant cells: {(results['sig'] != 'Not.sig').sum()}")
```

## Usage Examples

### Binary Outcome (e.g., Disease Status)
```python
# y is array of 0s and 1s
results = SCIPAC(bulk_pca, y, family='binomial', ct_res=ct_res)
```

### Continuous Outcome (e.g., Gene Expression)
```python
# y is array of continuous values
results = SCIPAC(bulk_pca, y, family='gaussian', ct_res=ct_res)
```

### Survival Analysis
```python
# y is DataFrame with 'time' and 'status' columns
import pandas as pd
y = pd.DataFrame({'time': survival_times, 'status': event_indicators})
results = SCIPAC(bulk_pca, y, family='cox', ct_res=ct_res)
```

### Ordinal Outcome (e.g., Disease Stage)
```python
# y is array of ordered categories (1, 2, 3, ...)
results = SCIPAC(bulk_pca, y, family='cumulative', ct_res=ct_res)
```

## Batch Correction

pySCIPAC integrates Harmony for batch correction:

```python
# Provide batch labels for each cell
batch_labels = ['batch1', 'batch2', 'batch1', ...]  # Length = n_cells

pca_res = sc_bulk_pca(
    sc_prep, bulk_prep,
    n_pc=60,
    batch_var=batch_labels  # Harmony correction applied automatically
)
```

## Output Format

SCIPAC returns a pandas DataFrame with the following columns:
- `cluster_assignment`: Cluster ID for each cell
- `Lambda.est`: Estimated phenotype association score
- `Lambda.upper`: Upper confidence bound
- `Lambda.lower`: Lower confidence bound  
- `sig`: Significance category
  - `Sig.pos`: Significantly positive association
  - `Sig.neg`: Significantly negative association
  - `Not.sig`: Non-significant
- `log.pval`: Log10 p-value with sign

## Performance Tips

1. **Parallel Processing**: Use `n_jobs=-1` to utilize all CPU cores
2. **Bootstrap Samples**: Reduce `bt_size` for faster testing (default: 50)
3. **PCA Components**: Use fewer PCs for large datasets (default: 60)
4. **Memory Management**: For large datasets, consider processing in batches

## Differences from R Implementation

- **Clustering**: Leiden by default (matches Seurat), KMeans as fallback
- **Ordinal Regression**: Uses `mord` package (proportional odds model)
- **Bootstrap Sampling**: Stratified for binomial/ordinal (matches R)
- **Data Structures**: Uses pandas/numpy instead of R matrices
- **Parallel Backend**: joblib instead of R's parallel package


## License

MIT License - see LICENSE file for details

## Citation

This is a Python implementation of the original SCIPAC R package developed by @RavenGan. If you use pySCIPAC, please cite the original author:

Gan, D., Zhu, Y., Lu, X., & Li, J. (2024). SCIPAC: quantitative estimation of cell-phenotype associations. Genome Biology, 25(1), 119.
