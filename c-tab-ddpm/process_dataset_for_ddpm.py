import pandas as pd
import numpy as np
import json
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# -- Command line argument parsing --
parser = argparse.ArgumentParser(description='Compile datasets for DDPM architecture')
parser.add_argument('--dataname', type=str, default='fintech', help='Target dataset identifier')
args = parser.parse_args()

DATASET = args.dataname
BASE_INFO_PATH = f'../data/Info/{DATASET}.json'
INPUT_DATA_CSV = f'../data/{DATASET}/imbalanced_noord.csv'
TARGET_EXPORT_DIR = f'../data/{DATASET}/Data_Tabddpm'

os.makedirs(TARGET_EXPORT_DIR, exist_ok=True)

# -- Configuration loading phase --
with open(BASE_INFO_PATH, 'r') as f:
    config_map = json.load(f)

df = pd.read_csv(INPUT_DATA_CSV, skipinitialspace=True)
columns = list(df.columns)

# -- Column mapping execution --
num_cols = [columns[i] for i in config_map['num_col_idx']]
cat_cols = [columns[i] for i in config_map['cat_col_idx']]
target_col = columns[config_map['target_col_idx'][0]]

X = df[num_cols + cat_cols]
y = df[target_col]

# -- Data segmentation (Train/Val/Test) --
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

# -- Categorical processing with empty-list safeguard --
cat_cardinalities = []
if len(cat_cols) > 0:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_train = oe.fit_transform(X_train[cat_cols]).astype(np.int64)
    X_cat_val = oe.transform(X_val[cat_cols]).astype(np.int64)
    X_cat_test = oe.transform(X_test[cat_cols]).astype(np.int64)
    
    cat_cardinalities = [len(oe.categories_[i]) for i in range(len(cat_cols))]
    
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_train.npy'), X_cat_train)
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_val.npy'), X_cat_val)
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_test.npy'), X_cat_test)
else:
    # Generate empty arrays to satisfy DDPM loading constraints
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_train.npy'), np.empty((len(X_train), 0)))
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_val.npy'), np.empty((len(X_val), 0)))
    np.save(os.path.join(TARGET_EXPORT_DIR, 'X_cat_test.npy'), np.empty((len(X_test), 0)))

# -- Numerical processing --
X_num_train = X_train[num_cols].values.astype(np.float32)
X_num_val = X_val[num_cols].values.astype(np.float32)
X_num_test = X_test[num_cols].values.astype(np.float32)

np.save(os.path.join(TARGET_EXPORT_DIR, 'X_num_train.npy'), X_num_train)
np.save(os.path.join(TARGET_EXPORT_DIR, 'X_num_val.npy'), X_num_val)
np.save(os.path.join(TARGET_EXPORT_DIR, 'X_num_test.npy'), X_num_test)

# -- Target label normalization --
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train).astype(np.int64)
y_val_encoded = le.transform(y_val).astype(np.int64)
y_test_encoded = le.transform(y_test).astype(np.int64)

np.save(os.path.join(TARGET_EXPORT_DIR, 'y_train.npy'), y_train_encoded)
np.save(os.path.join(TARGET_EXPORT_DIR, 'y_val.npy'), y_val_encoded)
np.save(os.path.join(TARGET_EXPORT_DIR, 'y_test.npy'), y_test_encoded)

# -- DDPM JSON Compilation --
all_cols = num_cols + cat_cols + [target_col]
idx_name_mapping = {str(i): name for i, name in enumerate(all_cols)}

info_ddpm = {
    "name": f"{DATASET}_imbalanced",
    "task_type": "binclass",
    "n_num_features": len(num_cols),
    "n_cat_features": len(cat_cols),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "cat_cardinalities": cat_cardinalities,
    "idx_name_mapping": idx_name_mapping,
    "num_col_idx": list(range(len(num_cols))),
    "cat_col_idx": list(range(len(num_cols), len(num_cols) + len(cat_cols))),
    "target_col_idx": [len(num_cols) + len(cat_cols)]
}

with open(os.path.join(TARGET_EXPORT_DIR, 'info.json'), 'w') as f:
    json.dump(info_ddpm, f, indent=4)

print(f"[COMPLETED] DDPM Tensors generated successfully for: {DATASET}")