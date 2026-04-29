import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# --- Paths configuration ---
INPUT_CSV = 'data/adult/imbalanced_noord.csv'
OUTPUT_DIR = 'data/adult/Data_Tabddpm'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the Data
df = pd.read_csv(INPUT_CSV, skipinitialspace=True)

# 2. Define Feature Types based precisely on your CSV columns
target_col = 'income'

num_cols = [
    'age', 'fnlwgt', 'educational-num', 
    'capital-gain', 'capital-loss', 'hours-per-week'
]

# Note: Removed 'education' as it is absent from your CSV
cat_cols = [
    'workclass', 'marital-status', 'occupation', 
    'relationship', 'race', 'gender', 'native-country'
]

X = df[num_cols + cat_cols]
y = df[target_col]

# 3. Two-Step Train/Val/Test Split
# Step 1: Split off 20% for testing
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Step 2: Split remaining 80% into Train and Validation (80% Train, 20% Val)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

# 4. Process Categorical Features
# Re-encoding ensures contiguous 0 to N-1 integers required by PyTorch Embeddings
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_train = oe.fit_transform(X_train[cat_cols]).astype(np.int64)
X_cat_val = oe.transform(X_val[cat_cols]).astype(np.int64)
X_cat_test = oe.transform(X_test[cat_cols]).astype(np.int64)

# 5. Process Numerical Features
X_num_train = X_train[num_cols].values.astype(np.float32)
X_num_val = X_val[num_cols].values.astype(np.float32)
X_num_test = X_test[num_cols].values.astype(np.float32)

# 6. Process Target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train).astype(np.int64)
y_val_encoded = le.transform(y_val).astype(np.int64)
y_test_encoded = le.transform(y_test).astype(np.int64)

# 7. Save Arrays to Disk
np.save(os.path.join(OUTPUT_DIR, 'X_num_train.npy'), X_num_train)
np.save(os.path.join(OUTPUT_DIR, 'X_num_val.npy'), X_num_val)
np.save(os.path.join(OUTPUT_DIR, 'X_num_test.npy'), X_num_test)

np.save(os.path.join(OUTPUT_DIR, 'X_cat_train.npy'), X_cat_train)
np.save(os.path.join(OUTPUT_DIR, 'X_cat_val.npy'), X_cat_val)
np.save(os.path.join(OUTPUT_DIR, 'X_cat_test.npy'), X_cat_test)

np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train_encoded)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val_encoded)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test_encoded)

# 8. Create info.json
# Crucial: TabDDPM needs exact category counts per column to build the embedding matrix
# Create exact indices mapping for the downstream stitcher script
all_cols = num_cols + cat_cols + [target_col]
idx_name_mapping = {str(i): name for i, name in enumerate(all_cols)}

num_col_idx = list(range(len(num_cols)))
cat_col_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
target_col_idx = [len(num_cols) + len(cat_cols)]

cat_cardinalities = [len(oe.categories_[i]) for i in range(len(cat_cols))]

info = {
    "name": "adult_imbalanced",
    "task_type": "binclass",
    "n_num_features": len(num_cols),
    "n_cat_features": len(cat_cols),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "cat_cardinalities": cat_cardinalities,
    # Newly added fields for seamless CSV reconstruction
    "idx_name_mapping": idx_name_mapping,
    "num_col_idx": num_col_idx,
    "cat_col_idx": cat_col_idx,
    "target_col_idx": target_col_idx
}

with open(os.path.join(OUTPUT_DIR, 'info.json'), 'w') as f:
    json.dump(info, f, indent=4)

print(f"Data successfully processed and saved to {OUTPUT_DIR}/")
print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")