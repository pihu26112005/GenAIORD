import sys
import os
import argparse
import pandas as pd
import datetime as dt

# Import from your modules
from utility_metrics import calculate_utility
from fidelity_metrics import calculate_fidelity
from privacy_metrics import calculate_privacy

def evaluate_strategy(model_name, strat_id, train_augmented, test_real, target_col):
    print(f"  -> Evaluating {model_name} | {strat_id} ...")
    
    # Split features and targets
    X_train = train_augmented.drop(columns=[target_col])
    y_train = train_augmented[target_col]
    X_test = test_real.drop(columns=[target_col])
    y_test = test_real[target_col]

    # Assume rows added at the end are synthetic (or filter them if you have a flag)
    # For a general proxy, we just compare the whole augmented train vs original test
    # In a perfect setup, you would pass EXACTLY the synthetic chunk here for fidelity
    synth_subset = train_augmented.sample(n=min(2000, len(train_augmented))) 
    real_subset = test_real.sample(n=min(2000, len(test_real)))

    # 1. Utility
    utility = calculate_utility(X_train, y_train, X_test, y_test, use_gpu=True)
    
    # 2. Fidelity 
    fidelity = calculate_fidelity(real_subset, synth_subset)
    
    # 3. Privacy 
    privacy = calculate_privacy(real_subset, synth_subset)
    
    # Merge dictionaries
    result = {"Model": model_name, "Strat": strat_id}
    result.update(utility)
    result.update(fidelity)
    result.update(privacy)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--tag', type=str, default='baseline', help='Ablation tag e.g., gamma0.8')
    args = parser.parse_args()

    # Load Base Data (Update paths to match your project root)
    base_path = f"../../data/{args.dataset}" 
    try:
        test_df = pd.read_csv(f'{base_path}/test_orig.csv')
        real_train = pd.read_csv(f'{base_path}/imbalanced_noord.csv')
    except FileNotFoundError:
        print(f"Error: Could not find data in {base_path}. Run from scripts/evaluation/")
        sys.exit(1)
        
    real_minority = real_train[real_train[args.target] == 1]
    real_majority = real_train[real_train[args.target] == 0]
    N_def = len(real_majority) - len(real_minority)

    # Registry of synthetic data paths
    SYNTHETIC_MODELS = {
        "Original_ORD": {
            "min": f"{base_path}/synthetic/synthetic_minority_original.csv",
            "maj": f"{base_path}/synthetic/synthetic_majority_original.csv"
        },
        "Proposed_CFG": {
            "min": f"{base_path}/synthetic/synthetic_minority_cfg.csv",
            "maj": f"{base_path}/synthetic/synthetic_majority_cfg.csv"
        },
        "TabDDPM": {
            "min": f"{base_path}/synthetic/synthetic_minority_tabddpm.csv",
            "maj": f"{base_path}/synthetic/synthetic_majority_tabddpm.csv"
        }
    }

    all_results = []

    for model_name, paths in SYNTHETIC_MODELS.items():
        try:
            syn_min = pd.read_csv(paths["min"])
            syn_min[args.target] = 1
            if 'cond' in syn_min.columns: syn_min.drop('cond', axis=1, inplace=True)
            
            # S1: Real Maj + Real Min + Syn Min
            s1_min = syn_min.sample(n=N_def, replace=True, random_state=42)
            train_s1 = pd.concat([real_majority, real_minority, s1_min]).sample(frac=1).reset_index(drop=True)
            
            res_s1 = evaluate_strategy(model_name, "S1_Balanced", train_s1, test_df, args.target)
            all_results.append(res_s1)
            
            # You can add S2 and S3 logic here mirroring your old compute_mle.py script
            
        except Exception as e:
            print(f"Skipping {model_name}: {e}")

    # Save Output with Naming Convention
    if all_results:
        report_df = pd.DataFrame(all_results)
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"../../results/{timestamp}_{args.dataset}_full_eval_{args.tag}.csv"
        
        report_df.to_csv(out_filename, index=False)
        print(f"\n[SUCCESS] Saved comprehensive metrics to: {out_filename}")

if __name__ == "__main__":
    main()


#     cd scripts/evaluation
# python run_evaluation.py --dataset adult --tag cfg_gamma0_8