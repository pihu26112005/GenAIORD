import numpy as np
import pandas as pd
import json
import os
import argparse
from sklearn.preprocessing import OrdinalEncoder

# -- Command line argument parsing --
parser = argparse.ArgumentParser(description='Create Synthetic CSV from DDPM output')
parser.add_argument('--dataname', type=str, default='fintech', help='Target dataset identifier')
args = parser.parse_args()

DATASET = args.dataname
PARENT_DIR = f"exp/{DATASET}"
INFO_PATH = f"../data/{DATASET}/Tabddpm/info.json"
OUTPUT_CSV = f"../data/{DATASET}/synthetic/synthetic_minority_tabddpm.csv" 
INPUT_CSV = f"../data/{DATASET}/imbalanced_noord.csv"

real_df = pd.read_csv(INPUT_CSV, skipinitialspace=True)

# 1. Read info.json FIRST so we know the architecture
with open(INFO_PATH, 'r') as f:
    info = json.load(f)

idx_name_mapping = info['idx_name_mapping']
num_col_idx = info['num_col_idx']
cat_col_idx = info['cat_col_idx']
target_col_idx = info['target_col_idx']

num_cols = [idx_name_mapping[str(i)] for i in num_col_idx]
cat_cols = [idx_name_mapping[str(i)] for i in cat_col_idx]
target_col = [idx_name_mapping[str(i)] for i in target_col_idx]

# 2. Load Numerical and Target arrays (These will always exist)
X_num = np.load(f"{PARENT_DIR}/X_num_train.npy", allow_pickle=True)
y = np.load(f"{PARENT_DIR}/y_train.npy", allow_pickle=True)

df_num = pd.DataFrame(X_num, columns=num_cols)
df_target = pd.DataFrame(y, columns=target_col)

# 3. Safely handle Categorical data ONLY if it exists
if len(cat_cols) > 0:
    X_cat = np.load(f"{PARENT_DIR}/X_cat_train.npy", allow_pickle=True)
    # Re-fit the encoder on the real data
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(real_df[cat_cols])
    # Reverse the categorical encoding
    X_cat_restored = oe.inverse_transform(X_cat)
    df_cat = pd.DataFrame(X_cat_restored, columns=cat_cols)
else:
    # If no categorical columns, just make an empty dataframe
    df_cat = pd.DataFrame()

# 4. Combine into a Pandas DataFrame
final_df = pd.concat([df_num, df_cat, df_target], axis=1)

# 5. SMART REORDERING: Only reorder columns that ACTUALLY exist in our generated data
# This prevents KeyErrors if original CSV had useless columns (like 'user' IDs)
ordered_cols = [col for col in real_df.columns if col in final_df.columns]
final_df = final_df[ordered_cols]

# 6. Save to CSV safely
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"[{DATASET.upper()}] Successfully saved synthetic data to {OUTPUT_CSV}")