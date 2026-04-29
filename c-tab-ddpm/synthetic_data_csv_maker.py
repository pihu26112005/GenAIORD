import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OrdinalEncoder

# Paths (Adjust if your parent_dir is different)
PARENT_DIR = "exp/adult/ddpm_mlp_best"
INFO_PATH = "../data/adult/Data_Tabddpm/info.json"
OUTPUT_CSV = "../data/adult/synthetic/synthetic_minority_tabddpm.csv" # Change this name based on what you just generated

INPUT_CSV = '../data/adult/imbalanced_noord.csv'
real_df = pd.read_csv(INPUT_CSV, skipinitialspace=True)

# 1. Load the generated arrays
X_num = np.load(f"{PARENT_DIR}/X_num_train.npy", allow_pickle=True)
X_cat = np.load(f"{PARENT_DIR}/X_cat_train.npy", allow_pickle=True)
y = np.load(f"{PARENT_DIR}/y_train.npy", allow_pickle=True)

# 2. Get the column names from your info.json
with open(INFO_PATH, 'r') as f:
    info = json.load(f)

# Extract column lists 
# (Based on the custom process_dataset.py script you showed earlier)
idx_name_mapping = info['idx_name_mapping']
num_col_idx = info['num_col_idx']
cat_col_idx = info['cat_col_idx']
target_col_idx = info['target_col_idx']

num_cols = [idx_name_mapping[str(i)] for i in num_col_idx]
cat_cols = [idx_name_mapping[str(i)] for i in cat_col_idx]
target_col = [idx_name_mapping[str(i)] for i in target_col_idx]

# Re-fit the encoder on the real data
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(real_df[cat_cols])

# CRITICAL: Reverse the categorical encoding!
X_cat_restored = oe.inverse_transform(X_cat)

# 3. Combine into a Pandas DataFrame
df_num = pd.DataFrame(X_num, columns=num_cols)
df_cat = pd.DataFrame(X_cat_restored, columns=cat_cols)
df_target = pd.DataFrame(y, columns=target_col)

# Concatenate them side-by-side
final_df = pd.concat([df_num, df_cat, df_target], axis=1)

# Ensure exact column order matches real data to prevent XGBoost errors
final_df = final_df[real_df.columns]

# 4. Save to CSV
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Successfully saved synthetic data to {OUTPUT_CSV}")