import sys
import os
import pandas as pd
import datetime as dt

# Import from your modules
from utility_metrics import calculate_utility
from fidelity_metrics import calculate_fidelity
from privacy_metrics import calculate_privacy

# =====================================================================
# ⚙️ CONFIGURATION HUB - EDIT THESE BEFORE RUNNING FOR A NEW DATASET
# =====================================================================
TARGET_DATASET = 'fintech'         # Options: 'adult', 'heloc', 'fintech'
TARGET_COLUMN = 'is_referred'         # Options: 'income', 'RiskPerformance', 'is_referred'
GAMMAS = ['0.7','0.75','0.8', '0.85','0.9','0.95'] # Add or remove your specific gamma tags
# =====================================================================

def evaluate_strategy(model_name, strat_id, train_augmented, test_real, target_col):
    print(f"    -> Evaluating Strategy: {strat_id} ...")
    
    X_train = train_augmented.drop(columns=[target_col])
    y_train = train_augmented[target_col]
    X_test = test_real.drop(columns=[target_col])
    y_test = test_real[target_col]

    synth_subset = train_augmented.sample(n=min(2000, len(train_augmented))) 
    real_subset = test_real.sample(n=min(2000, len(test_real)))

    utility = calculate_utility(X_train, y_train, X_test, y_test, use_gpu=True, RANDOM_SEED=42)
    fidelity = calculate_fidelity(real_subset, synth_subset)
    privacy = calculate_privacy(real_subset, synth_subset)
    
    result = {"Model": model_name, "Strat": strat_id}
    result.update(utility)
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

        # --- THE FIX: Dynamic Column Matching ---
        # Find columns that exist in BOTH the original data and the generated data
        shared_cols = [col for col in expected_cols if col in syn_min.columns and col in syn_maj.columns]
        
        # Subset synthetic data to these shared columns
        syn_min = syn_min[shared_cols]
        syn_maj = syn_maj[shared_cols]
        
        # Subset real data (train and test) to these shared columns so XGBoost doesn't crash
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
def main():
    print(f"\n{'='*50}\nEvaluating Dataset: {TARGET_DATASET.upper()}\n{'='*50}")
    base_path = f"../../data/{TARGET_DATASET}" 
    
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
    N_def = len(real_majority) - len(real_minority)
    
    real_C00 = real_ord[real_ord['cond'] == 0].copy()
    real_C00[TARGET_COLUMN] = 0
    real_C00.drop('cond', axis=1, inplace=True)
    N_01 = N_maj - len(real_C00) 
    
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

    # 3. Evaluate CFG for each Gamma and save distinct files
    print("\n--- Phase 2: Evaluating CFG Models & Saving Reports ---")
    for gamma in GAMMAS:
        model_name = f"Proposed_CFG_Gamma_{gamma}"
        print(f"\n  [ Proposed ] {model_name}")
        
        cfg_min_path = f"{base_path}/synthetic/synthetic_minority_cfg_gamma={gamma}.csv"
        cfg_maj_path = f"{base_path}/synthetic/synthetic_majority_cfg_gamma={gamma}.csv"
        
        cfg_results = evaluate_model_suite(model_name, cfg_min_path, cfg_maj_path, real_majority, real_minority, real_C00, test_df, TARGET_COLUMN, N_maj, N_def, N_01, expected_cols)
        
        # Combine the baseline metrics with this specific Gamma's metrics
        final_report_data = baseline_results + cfg_results
        
        if final_report_data:
            report_df = pd.DataFrame(final_report_data)
            out_filename = f"{results_dir}/{timestamp}_eval_gamma{gamma}.csv"
            report_df.to_csv(out_filename, index=False)
            print(f"    [+] Saved report: {out_filename}")

    print(f"\n[SUCCESS] Completed evaluations for {TARGET_DATASET.upper()}. All files saved to {results_dir}/")

if __name__ == "__main__":
    main()