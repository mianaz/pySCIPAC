"""
Core SCIPAC functions for phenotype-associated cell identification
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from joblib import Parallel, delayed
from typing import Union, Optional, Literal
import warnings
from tqdm import tqdm


def classifier_lambda_core(
    bulk_dat: np.ndarray,
    y: Union[np.ndarray, pd.DataFrame],
    family: str,
    k_means_res: dict,
    ela_net_alpha: float = 0.4,
    nfold: int = 10,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Core function to calculate Lambda values using different regression methods.
    
    Parameters
    ----------
    bulk_dat : array-like
        Dimension-reduced bulk data (samples x PCs)
    y : array-like
        Class labels or survival time
    family : str
        Regression family: 'binomial', 'gaussian', 'cox', 'cumulative'
    k_means_res : dict
        Clustering results from seurat_ct
    ela_net_alpha : float
        ElasticNet mixing parameter (0=Ridge, 1=Lasso)
    nfold : int
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
    """
    
    k_means_cent = k_means_res['centers']
    k = k_means_res['k']
    ct_assign = k_means_res['ct_assignment'].copy()
    
    n_samples = bulk_dat.shape[0]

    # Bootstrap sampling
    if random_state is not None:
        np.random.seed(random_state)

    # Determine if stratified sampling is needed (for binomial and cumulative families)
    use_stratified = family in ['binomial', 'cumulative']

    if use_stratified:
        # Stratified bootstrap: sample within each class to maintain class proportions
        # This matches R's implementation which samples separately per class
        if isinstance(y, pd.DataFrame):
            y_values = y.values.flatten()
        else:
            y_values = np.asarray(y).flatten()

        unique_classes = np.unique(y_values)
        resample_idx = []

        for class_label in unique_classes:
            class_idx = np.where(y_values == class_label)[0]
            # Sample with replacement within this class
            resampled_class_idx = np.random.choice(class_idx, len(class_idx), replace=True)
            resample_idx.extend(resampled_class_idx)

        resample_idx = np.array(resample_idx)
        # Shuffle to mix classes (order shouldn't matter but matches R behavior)
        np.random.shuffle(resample_idx)
    else:
        # Simple random sampling for gaussian and cox families
        resample_idx = np.random.choice(n_samples, n_samples, replace=True)

    new_sample = bulk_dat[resample_idx]
    new_sample_ave = np.mean(new_sample, axis=0)

    if family == 'binomial':
        # Prepare labels
        if isinstance(y, pd.DataFrame):
            y_values = y.values.flatten()
        else:
            y_values = y.flatten()
        
        new_y = y_values[resample_idx]
        
        # Logistic regression with ElasticNet
        if ela_net_alpha == 0:
            # Use Ridge
            model = LogisticRegressionCV(
                penalty='l2',
                cv=nfold,
                random_state=random_state,
                max_iter=1000
            )
        elif ela_net_alpha == 1:
            # Use Lasso
            model = LogisticRegressionCV(
                penalty='l1',
                cv=nfold,
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            )
        else:
            # Use ElasticNet
            model = LogisticRegressionCV(
                penalty='elasticnet',
                l1_ratios=[ela_net_alpha],
                cv=nfold,
                solver='saga',
                random_state=random_state,
                max_iter=1000
            )
        
        model.fit(new_sample, new_y)
        beta = model.coef_.flatten()
        
    elif family == 'gaussian':
        # Linear regression
        new_y = y[resample_idx]
        
        # Use ElasticNetCV for linear regression
        model = ElasticNetCV(
            l1_ratio=ela_net_alpha,
            cv=nfold,
            random_state=random_state,
            max_iter=1000
        )
        
        model.fit(new_sample, new_y)
        beta = model.coef_
        
    elif family == 'cox':
        # Cox regression
        try:
            from lifelines import CoxPHFitter
            from lifelines.utils import k_fold_cross_validation
        except ImportError:
            raise ImportError("lifelines is required for Cox regression. Install with: pip install lifelines")
        
        # Prepare survival data
        if isinstance(y, pd.DataFrame):
            surv_data = y.iloc[resample_idx].copy()
        else:
            surv_data = pd.DataFrame(y[resample_idx], columns=['time', 'status'])
        
        # Add covariates
        for i in range(new_sample.shape[1]):
            surv_data[f'PC{i+1}'] = new_sample[:, i]
        
        # Fit Cox model with ElasticNet penalty
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=ela_net_alpha)
        cph.fit(surv_data, duration_col='time', event_col='status')
        
        # Extract coefficients
        beta = cph.params_.values
        
    elif family == 'cumulative':
        # Ordinal regression using mord package (proportional odds model)
        # This is equivalent to R's ordinalNet with family="cumulative"
        new_y = y[resample_idx]

        try:
            import mord

            # Use LogisticAT (All-Threshold) for ordinal regression with regularization
            # alpha parameter in mord controls regularization strength (higher = more regularization)
            # We use a moderate regularization similar to glmnet defaults
            model = mord.LogisticAT(alpha=1.0, max_iter=1000)
            model.fit(new_sample, new_y)
            beta = model.coef_.flatten()

        except ImportError:
            # Fallback: use multinomial logistic regression (less accurate for ordinal data)
            warnings.warn(
                "mord package not installed. Using multinomial logistic regression as fallback. "
                "For proper ordinal regression, install mord: pip install mord"
            )

            if ela_net_alpha == 0:
                model = LogisticRegressionCV(
                    penalty='l2',
                    cv=nfold,
                    multi_class='multinomial',
                    solver='lbfgs',
                    random_state=random_state,
                    max_iter=1000
                )
            elif ela_net_alpha == 1:
                model = LogisticRegressionCV(
                    penalty='l1',
                    cv=nfold,
                    multi_class='ovr',
                    solver='liblinear',
                    random_state=random_state,
                    max_iter=1000
                )
            else:
                model = LogisticRegressionCV(
                    penalty='elasticnet',
                    l1_ratios=[ela_net_alpha],
                    cv=nfold,
                    multi_class='multinomial',
                    solver='saga',
                    random_state=random_state,
                    max_iter=1000
                )

            model.fit(new_sample, new_y)
            beta = model.coef_.mean(axis=0)  # Average across classes
        
    else:
        raise ValueError(f"Invalid family: {family}. Choose from 'binomial', 'gaussian', 'cox', 'cumulative'")

    # Numerical stability: clip extreme coefficient values to prevent overflow
    beta = np.clip(beta, -1e10, 1e10)
    # Replace any NaN or Inf values with 0
    beta = np.nan_to_num(beta, nan=0.0, posinf=1e10, neginf=-1e10)

    # Calculate Lambda values
    lambda_values = np.zeros(k)
    for cluster_idx in range(k):
        chosen_cen = k_means_cent[:, cluster_idx]
        x = chosen_cen - new_sample_ave
        # Clip x values to prevent overflow in dot product
        x = np.clip(x, -1e10, 1e10)
        lambda_values[cluster_idx] = np.dot(beta, x)
    
    # Assign Lambda to cells based on cluster
    ct_idx = ct_assign['cluster_assignment'].values
    ct_assign['Lambda'] = lambda_values[ct_idx]

    # Clip Lambda values before standardization to prevent extreme outliers
    ct_assign['Lambda'] = np.clip(ct_assign['Lambda'], -1e10, 1e10)

    # Standardize Lambda values
    lambda_std = np.std(ct_assign['Lambda'])
    if lambda_std > 1e-10:
        ct_assign['Lambda'] = (ct_assign['Lambda'] - np.mean(ct_assign['Lambda'])) / lambda_std
    else:
        warnings.warn("Lambda values have near-zero variance")
        ct_assign['Lambda'] = np.zeros(len(ct_assign))
    
    return ct_assign


def classifier_lambda(
    bulk_dat: np.ndarray,
    y: Union[np.ndarray, pd.DataFrame],
    family: str,
    k_means_res: dict,
    ela_net_alpha: float = 0.4,
    bt_size: int = 50,
    nfold: int = 10,
    n_jobs: int = -1,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply parallel computing to calculate Lambda values with bootstrap.
    
    Parameters
    ----------
    bulk_dat : array-like
        Dimension-reduced bulk data (samples x PCs)
    y : array-like
        Class labels or survival time
    family : str
        Regression family
    k_means_res : dict
        Clustering results
    ela_net_alpha : float
        ElasticNet mixing parameter
    bt_size : int
        Number of bootstrap samples
    nfold : int
        Number of CV folds
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    verbose : bool
        Show progress bar
    """
    
    def compute_lambda(seed):
        try:
            return classifier_lambda_core(
                bulk_dat, y, family, k_means_res,
                ela_net_alpha, nfold, random_state=seed
            )['Lambda'].values
        except Exception as e:
            warnings.warn(f"Bootstrap sample {seed} failed: {str(e)}")
            return None
    
    # Run bootstrap samples in parallel
    if verbose:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_lambda)(seed) 
            for seed in tqdm(range(bt_size), desc="Bootstrap samples")
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_lambda)(seed) for seed in range(bt_size)
        )
    
    # Filter out failed samples
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        raise RuntimeError("All bootstrap samples failed")
    
    if len(valid_results) < bt_size * 0.5:
        warnings.warn(f"Only {len(valid_results)} out of {bt_size} bootstrap samples succeeded")
    
    # Stack results
    lambda_res = np.column_stack(valid_results)
    
    return lambda_res


def obtain_ct_lambda(
    lambda_res: np.ndarray,
    k_means_res: dict,
    ci_alpha: float = 0.05
) -> pd.DataFrame:
    """
    Summarize Lambda values and determine significance.
    
    Parameters
    ----------
    lambda_res : array-like
        Lambda values for each cell across bootstrap samples
    k_means_res : dict
        Clustering results
    ci_alpha : float
        Significance level for confidence intervals
    """
    
    ct_assign = k_means_res['ct_assignment'].copy()

    # Calculate statistics with numerical stability
    lambda_est = np.mean(lambda_res, axis=1)
    lambda_std = np.std(lambda_res, axis=1)

    # Handle NaN/Inf values that may arise from failed bootstrap samples
    lambda_est = np.nan_to_num(lambda_est, nan=0.0, posinf=1e10, neginf=-1e10)
    lambda_std = np.nan_to_num(lambda_std, nan=1e-10, posinf=1e10, neginf=0.0)
    # Ensure std is never zero to avoid division issues
    lambda_std = np.maximum(lambda_std, 1e-10)

    # Calculate confidence intervals
    z_score = stats.norm.ppf(1 - ci_alpha/2)
    lambda_upper = lambda_est + z_score * lambda_std
    lambda_lower = lambda_est - z_score * lambda_std

    # Calculate p-values with numerical stability
    lambda_z = lambda_est / lambda_std
    # Clip z-scores to prevent extreme p-values
    lambda_z = np.clip(lambda_z, -37, 37)  # norm.sf(-37) ~ 1, norm.sf(37) ~ 0
    lambda_pval = 2 * stats.norm.sf(np.abs(lambda_z))
    # Avoid log10(0) by ensuring minimum p-value
    lambda_pval = np.maximum(lambda_pval, 1e-300)
    lambda_log_pval = -np.log10(lambda_pval) * np.sign(lambda_z)
    
    # Determine significance
    sig = np.full(len(lambda_res), 'Not.sig', dtype=object)
    sig[(lambda_upper > 0) & (lambda_lower > 0)] = 'Sig.pos'
    sig[(lambda_upper < 0) & (lambda_lower < 0)] = 'Sig.neg'
    
    # Add results to dataframe
    ct_assign['Lambda.est'] = lambda_est
    ct_assign['Lambda.upper'] = lambda_upper
    ct_assign['Lambda.lower'] = lambda_lower
    ct_assign['sig'] = pd.Categorical(sig, categories=['Sig.pos', 'Sig.neg', 'Not.sig'])
    ct_assign['log.pval'] = lambda_log_pval
    
    return ct_assign


def SCIPAC(
    bulk_dat: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    family: Literal['binomial', 'gaussian', 'cox', 'cumulative'],
    ct_res: dict,
    ela_net_alpha: float = 0.4,
    bt_size: int = 50,
    n_jobs: int = -1,
    ci_alpha: float = 0.05,
    nfold: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main SCIPAC function to identify phenotype-associated cells.
    
    Parameters
    ----------
    bulk_dat : array-like
        Dimension-reduced bulk data (samples x PCs)
    y : array-like
        Class labels or survival data
    family : str
        Regression family: 'binomial', 'gaussian', 'cox', 'cumulative'
    ct_res : dict
        Clustering results from seurat_ct
    ela_net_alpha : float
        ElasticNet mixing parameter (0=Ridge, 1=Lasso)
    bt_size : int
        Number of bootstrap samples
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    ci_alpha : float
        Significance level for confidence intervals
    nfold : int
        Number of cross-validation folds
    verbose : bool
        Show progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - cluster_assignment: Cluster ID for each cell
        - Lambda.est: Estimated Lambda value
        - Lambda.upper: Upper confidence bound
        - Lambda.lower: Lower confidence bound
        - sig: Significance category (Sig.pos/Sig.neg/Not.sig)
        - log.pval: Log10 p-value with sign
    """
    
    if verbose:
        print(f"Running SCIPAC with family='{family}', {bt_size} bootstrap samples")
    
    # Convert to numpy if needed
    if isinstance(bulk_dat, pd.DataFrame):
        bulk_arr = bulk_dat.values
    else:
        bulk_arr = bulk_dat
    
    # Calculate Lambda values with bootstrap
    lambda_res = classifier_lambda(
        bulk_arr, y, family, ct_res,
        ela_net_alpha, bt_size, nfold, n_jobs, verbose
    )
    
    # Summarize results
    ct_assign = obtain_ct_lambda(lambda_res, ct_res, ci_alpha)
    
    if verbose:
        n_pos = (ct_assign['sig'] == 'Sig.pos').sum()
        n_neg = (ct_assign['sig'] == 'Sig.neg').sum()
        n_ns = (ct_assign['sig'] == 'Not.sig').sum()
        print(f"Results: {n_pos} positive, {n_neg} negative, {n_ns} non-significant cells")
    
    return ct_assign