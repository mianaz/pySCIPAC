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
        "scanpy>=1.8.0",
        "anndata>=0.8.0",
        "harmonypy>=0.0.9",
        "glmnet>=2.0.0",
        "lifelines>=0.27.0",  # for cox regression
        "joblib>=1.0.0",  # for parallel processing
        "tqdm>=4.60.0",  # for progress bars
    ],
    python_requires=">=3.7",
)