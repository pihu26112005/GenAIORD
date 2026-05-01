import sys
import os
import pandas as pd
import datetime as dt
import argparse
from sklearn.preprocessing import OrdinalEncoder

# Import from your modules
from utility_metrics import calculate_utility, calculate_sdmetrics
from fidelity_metrics import calculate_fidelity
from privacy_metrics import calculate_privacy

# =====================================================================
# ⚙️ CONFIGURATION HUB - EDIT THESE BEFORE RUNNING FOR A NEW DATASET
# =====================================================================
# TARGET_DATASET = 'fintech'       # Options: 'adult', 'heloc', 'fintech'
# TARGET_COLUMN = 'churn'          # Options: 'income', 'RiskPerformance', 'churn' (Change if needed)
# GAMMAS = ['0.7', '0.75', '0.8', '0.85', '0.9', '0.95'] # Add your specific gamma tags
# =====================================================================

def evaluate_strategy(model_name, strat_id, train_augmented, test_real, target_col):
    print(f"    -> Evaluating Strategy: {strat_id} ...")
    
    # Make safe copies to prevent Pandas warnings and cross-contamination
    train_aug = train_augmented.copy()
    test_r = test_real.copy()

    # --- FIX 3: The Imputation Layer (Squashing NaNs and Mixed Types) ---
    # We must fill missing values BEFORE the math algorithms or encoders see them
    for col in train_aug.columns:
        if train_aug[col].dtype == 'object' or test_r[col].dtype == 'object':
            # For text columns: Fill NaNs with the word 'Missing' and force everything to be a string
            train_aug[col] = train_aug[col].fillna('Missing').astype(str)
            test_r[col] = test_r[col].fillna('Missing').astype(str)
        else:
            # For number columns: Fill NaNs with the median value of that column
            med = train_aug[col].median()
            if pd.isna(med): med = 0 # Fallback just in case
            train_aug[col] = train_aug[col].fillna(med)
            test_r[col] = test_r[col].fillna(med)
    # --------------------------------------------------------------------
    
    # --- 2. RUN SDMETRICS HERE ---
    # Run this BEFORE Ordinal Encoding so SDMetrics knows what columns are categoricals
    sdmetrics = calculate_sdmetrics(train_aug, test_r, target_col)

    # --- FIX 1: Encode String/Object Columns for XGBoost ---
    # Find all columns that are objects (strings) and convert them to numeric labels
    object_cols = train_aug.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fit on combined data to ensure all possible labels are caught
        oe.fit(pd.concat([train_aug[object_cols], test_r[object_cols]]))
        train_aug[object_cols] = oe.transform(train_aug[object_cols])
        test_r[object_cols] = oe.transform(test_r[object_cols])
    # -------------------------------------------------------
    
    X_train = train_aug.drop(columns=[target_col])
    y_train = train_aug[target_col]
    X_test = test_r.drop(columns=[target_col])
    y_test = test_r[target_col]

    synth_subset = train_aug.sample(n=min(2000, len(train_aug))) 
    real_subset = test_r.sample(n=min(2000, len(test_r)))

    utility = calculate_utility(X_train, y_train, X_test, y_test, use_gpu=True, RANDOM_SEED=42)
    fidelity = calculate_fidelity(real_subset, synth_subset)
    privacy = calculate_privacy(real_subset, synth_subset)
    
    result = {"Model": model_name, "Strat": strat_id}
    result.update(utility)
    result.update(sdmetrics)  # NEW: SDMetrics utility
    result.update(fidelity)
    result.update(privacy)
    
    return result
def evaluate_model_suite(model_name, min_path, maj_path, real_majority, real_minority, real_C00, test_df, target_col, N_maj, N_def, N_01, expected_cols):
    results = []
    try:
        # Load synthetic data
        syn_min = pd.read_csv(min_path)
        syn_maj = pd.read_csv(maj_path)
        
        syn_min[target_col] = 1
        syn_maj[target_col] = 0
        
        if 'cond' in syn_min.columns: syn_min.drop('cond', axis=1, inplace=True)
        if 'cond' in syn_maj.columns: syn_maj.drop('cond', axis=1, inplace=True)

        # --- FIX 2: Bulletproof Dynamic Column Matching ---
        # A column ONLY survives if it exists in every single dataframe we are about to use
        valid_dfs = [syn_min, syn_maj, test_df, real_C00]
        shared_cols = [col for col in expected_cols if all(col in df.columns for df in valid_dfs)]
        
        # Subset synthetic data to these shared columns
        syn_min = syn_min[shared_cols]
        syn_maj = syn_maj[shared_cols]
        
        # Subset real data (train and test) to these shared columns
        r_maj = real_majority[shared_cols]
        r_min = real_minority[shared_cols]
        r_c00 = real_C00[shared_cols]
        t_df  = test_df[shared_cols]
        # ----------------------------------------
        
        # STRATEGY 1
        s1_min = syn_min.sample(n=N_def, replace=True, random_state=42)
        train_s1 = pd.concat([r_maj, r_min, s1_min]).sample(frac=1, random_state=42).reset_index(drop=True)
        results.append(evaluate_strategy(model_name, "S1_Balanced", train_s1, t_df, target_col))
        
        # STRATEGY 2
        s2_maj = syn_maj.sample(n=N_maj, replace=True, random_state=42)
        s2_min = syn_min.sample(n=N_def, replace=True, random_state=42)
        train_s2 = pd.concat([s2_maj, r_min, s2_min]).sample(frac=1, random_state=42).reset_index(drop=True)
        results.append(evaluate_strategy(model_name, "S2_SynMaj", train_s2, t_df, target_col))

        # STRATEGY 3
        s3_maj = syn_maj.sample(n=N_01, replace=True, random_state=42)
        s3_min = syn_min.sample(n=N_def, replace=True, random_state=42)
        train_s3 = pd.concat([r_c00, s3_maj, r_min, s3_min]).sample(frac=1, random_state=42).reset_index(drop=True)
        results.append(evaluate_strategy(model_name, "S3_Overlap", train_s3, t_df, target_col))
        
    except FileNotFoundError:
        print(f"    [!] Missing files for {model_name}. Skipping.")
    except Exception as e:
        print(f"    [!] Error evaluating {model_name}: {e}")
        
    return results

def main(TARGET_DATASET, TARGET_COLUMN, gamma_start, gamma_end, gamma_step):
    print(f"\n{'='*50}\nEvaluating Dataset: {TARGET_DATASET.upper()}\n{'='*50}")
    base_path = f"../../data/{TARGET_DATASET}" 

    # Generate gamma values dynamically
    num_steps = int(round((gamma_end - gamma_start) / gamma_step)) + 1
    GAMMAS = [f"{(gamma_start + i * gamma_step):.2f}" for i in range(num_steps)]
    print(f"Evaluating the following Gammas: {GAMMAS}")
    
    try:
        test_df = pd.read_csv(f'{base_path}/test_orig.csv')
        real_train = pd.read_csv(f'{base_path}/imbalanced_noord.csv')
        real_ord = pd.read_csv(f'{base_path}/imbalanced_ord.csv')
    except FileNotFoundError:
        print(f"Error: Base data missing for {TARGET_DATASET}.")
        sys.exit(1)
        
    real_minority = real_train[real_train[TARGET_COLUMN] == 1]
    real_majority = real_train[real_train[TARGET_COLUMN] == 0]
    
    N_maj = len(real_majority)
    # Clamp to 0 to prevent negative sampling crashes
    N_def = max(0, len(real_majority) - len(real_minority))
    
    real_C00 = real_ord[real_ord['cond'] == 0].copy()
    real_C00[TARGET_COLUMN] = 0
    real_C00.drop('cond', axis=1, inplace=True)
    # Clamp to 0 to prevent negative sampling crashes
    N_01 = max(0, N_maj - len(real_C00))
    
    expected_cols = real_train.columns.tolist()

    # 1. Evaluate Baselines (Runs ONCE to save time)
    print("\n--- Phase 1: Evaluating Baselines ---")
    baseline_results = []
    baselines = {
        "Original_TabSyn": {
            "min": f"{base_path}/synthetic/synthetic_minority_original.csv",
            "maj": f"{base_path}/synthetic/synthetic_majority_original.csv"
        },
        "TabDDPM": {
            "min": f"{base_path}/synthetic/synthetic_minority_tabddpm.csv",
            "maj": f"{base_path}/synthetic/synthetic_majority_tabddpm.csv"
        }
    }
    
    for model_name, paths in baselines.items():
        print(f"\n  [ Baseline ] {model_name}")
        res = evaluate_model_suite(model_name, paths["min"], paths["maj"], real_majority, real_minority, real_C00, test_df, TARGET_COLUMN, N_maj, N_def, N_01, expected_cols)
        baseline_results.extend(res)

    # 2. Setup Folder Structure
    results_dir = f"../../results/{TARGET_DATASET}"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize master list with baseline results
    all_results = []
    all_results.extend(baseline_results)

    # 3. Evaluate CFG for each Gamma
    print("\n--- Phase 2: Evaluating CFG Models ---")
    for gamma in GAMMAS:
        model_name = f"Proposed_CFG_Gamma_{gamma}"
        print(f"\n  [ Proposed ] {model_name}")
        
        cfg_min_path = f"{base_path}/synthetic/synthetic_minority_cfg_gamma={gamma}.csv"
        cfg_maj_path = f"{base_path}/synthetic/synthetic_majority_cfg_gamma={gamma}.csv"
        
        cfg_results = evaluate_model_suite(model_name, cfg_min_path, cfg_maj_path, real_majority, real_minority, real_C00, test_df, TARGET_COLUMN, N_maj, N_def, N_01, expected_cols)
        
        # Add these specific gamma results to our master list
        all_results.extend(cfg_results)

    # 4. Save ONE combined report
    print("\n--- Phase 3: Saving Combined Report ---")
    if all_results:
        report_df = pd.DataFrame(all_results)
        out_filename = f"{results_dir}/{timestamp}_eval_combined_{TARGET_DATASET}.csv"
        report_df.to_csv(out_filename, index=False)
        print(f"    [+] Saved COMPLETE combined report to: {out_filename}")

    print(f"\n[SUCCESS] Completed evaluations for {TARGET_DATASET.upper()}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic datasets with various gammas.")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset (e.g., heloc, fintech)")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name (e.g., RiskPerformance, churn)")
    
    # Optional arguments with your requested defaults
    parser.add_argument("--gamma_start", type=float, default=0.6, help="Starting gamma value (default: 0.6)")
    parser.add_argument("--gamma_end", type=float, default=1.2, help="Ending gamma value (default: 1.2)")
    parser.add_argument("--gamma_step", type=float, default=0.02, help="Step size for gammas (default: 0.02)")

    args = parser.parse_args()
    
    # Pass arguments into main
    main(args.dataset, args.target_col, args.gamma_start, args.gamma_end, args.gamma_step)