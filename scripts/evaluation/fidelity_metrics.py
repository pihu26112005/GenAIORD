import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sdmetrics.single_column import KSComplement

def calculate_pairwise_correlation_error(real_df, synth_df):
    """Calculates the absolute difference between correlation matrices."""
    # Only use numeric columns for correlation
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    
    real_corr = real_df[num_cols].corr().fillna(0).values
    synth_corr = synth_df[num_cols].corr().fillna(0).values
    
    # Mean absolute error of the correlation matrices
    pce = np.mean(np.abs(real_corr - synth_corr))
    return pce

def calculate_fidelity(real_df, synth_df):
    """Calculates Wasserstein, JSD, Marginals, and PCE."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    
    wasserstein_scores = []
    jsd_scores = []
    marginal_scores = []

    for col in num_cols:
        real_col = real_df[col].dropna()
        synth_col = synth_df[col].dropna()
        
        # 1. Wasserstein Distance
        wd = wasserstein_distance(real_col, synth_col)
        wasserstein_scores.append(wd)
        
        # 2. Jensen-Shannon Divergence (requires binning for continuous data)
        hist_real, bin_edges = np.histogram(real_col, bins=20, density=True)
        hist_synth, _ = np.histogram(synth_col, bins=bin_edges, density=True)
        # Add epsilon to avoid division by zero
        jsd = jensenshannon(hist_real + 1e-10, hist_synth + 1e-10)
        jsd_scores.append(jsd)
        
        # 3. SDMetrics Marginal (KS-Complement)
        ks_score = KSComplement.compute(real_col, synth_col)
        marginal_scores.append(ks_score)

    return {
        "Avg_Wasserstein": np.mean(wasserstein_scores),
        "Avg_JSD": np.mean(jsd_scores),
        "Avg_Marginal_KS": np.mean(marginal_scores),
        "Pairwise_Corr_Error": calculate_pairwise_correlation_error(real_df, synth_df)
    }