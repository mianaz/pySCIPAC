from setuptools import setup, find_packages

setup(
    name="scipac",
    version="0.1.0",
    description="Single Cell Identifier of Phenotype-Associated Cells - Python Implementation",
    author="SCIPAC Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "leiden": [
            "python-igraph>=0.10.0",
            "leidenalg>=0.9.0",
        ],
        "ordinal": [
            "mord>=0.6",
        ],
        "survival": [
            "lifelines>=0.27.0",
        ],
        "batch": [
            "harmonypy>=0.0.9",
        ],
        "scanpy": [
            "scanpy>=1.8.0",
            "anndata>=0.8.0",
        ],
        "all": [
            "python-igraph>=0.10.0",
            "leidenalg>=0.9.0",
            "mord>=0.6",
            "lifelines>=0.27.0",
            "harmonypy>=0.0.9",
            "scanpy>=1.8.0",
            "anndata>=0.8.0",
        ],
    },
    python_requires=">=3.8",
)