# Changelog

All notable changes to pySCIPAC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Code simplification: utility functions `_to_array()`, `_resample_y()`, `_stabilize()` in core.py
- Consolidated numerical stability handling across bootstrap and summarization

### Changed
- Reduced code duplication in `classifier_lambda_core()` for y-value handling

## [0.1.0] - 2024-XX-XX

### Added
- Initial Python implementation of SCIPAC
- Support for four outcome types: binary, continuous, ordinal, survival
- AnnData integration with `run_scipac_anndata()` convenience function
- Scanpy-compatible HVG selection methods (seurat, cell_ranger, seurat_v3)
- Scanpy neighbor graph + Leiden/Louvain clustering
- KMeans fallback when scanpy unavailable
- Harmony batch correction integration
- Parallel bootstrap processing with joblib
- Stratified sampling for binomial/ordinal families
- Comprehensive input validation
- Graceful dependency handling with informative error messages

### Dependencies
- Core: numpy, pandas, scikit-learn, scipy, joblib, tqdm
- Optional: scanpy, anndata, leidenalg, mord, lifelines, harmonypy, matplotlib, seaborn
