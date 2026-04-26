import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_privacy(real_df, synth_df, sample_size=2000):
    """
    Calculates DCR and NNDR. 
    Uses sampling to prevent massive RAM overhead on large datasets.
    """
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    
    real_num = real_df[num_cols].copy()
    synth_num = synth_df[num_cols].copy()
    
    # Downsample if too large to speed up nearest neighbors
    if len(synth_num) > sample_size:
        synth_num = synth_num.sample(n=sample_size, random_state=42)
    
    # Scale data (Distances are meaningless without scaling)
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_num)
    synth_scaled = scaler.transform(synth_num)

    # Fit KDTree on REAL data
    # n_neighbors=2 so we can get the 1st and 2nd closest real points
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=-1)
    nn.fit(real_scaled)

    # Find closest real points for each synthetic point
    distances, _ = nn.kneighbors(synth_scaled)

    # DCR: Distance to the 1st closest real record
    dcr_values = distances[:, 0]
    
    # NNDR: Ratio of distance to 1st closest vs 2nd closest
    # Add small epsilon to denominator to avoid division by zero
    nndr_values = distances[:, 0] / (distances[:, 1] + 1e-10)

    # We report the 5th percentile as worst-case privacy risk
    return {
        "DCR_mean": np.mean(dcr_values),
        "DCR_5th_perc": np.percentile(dcr_values, 5),
        "NNDR_mean": np.mean(nndr_values),
        "NNDR_5th_perc": np.percentile(nndr_values, 5)
    }