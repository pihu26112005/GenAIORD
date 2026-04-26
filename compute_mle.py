
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
import numpy as np
import datetime as dt

import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import CalibratedClassifierCV

import sdmetrics
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import (BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier)
import argparse



def expected_calibration_error(samples, true_labels, M=10, threshold=0.5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    # predicted_label = np.argmax(samples, axis=1)
    predicted_label = (np.array([x[1] for x in samples]) >= threshold).astype(int)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()

            # print(f"Bin: {bin_lower.item():.3f} - {bin_upper.item():.3f} | Frac: {prob_in_bin:.3f} | Accuracy: {accuracy_in_bin:.3f}")
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()


def xgboost_train(X_train, y_train, EPOCHS=200, use_gpu=True):
    # defining the XGBoost train and test loaders
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)

    # defining the hyperparameters and training the model
    params = {
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss'
    }
    
    if use_gpu:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
        
    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=EPOCHS)   
    return model

def xgb_predict(model, X_test, y_test, model_name, strategy_name, threshold=0.5, save_dir="plots"):
    preds = model.predict(xgb.DMatrix(X_test))
    y_pred = [pred >= threshold for pred in preds]
    
    # Core Metrics
    acc = accuracy_score(y_test, y_pred)
    min_acc = recall_score(y_test, y_pred, pos_label=1)
    maj_acc = recall_score(y_test, y_pred, pos_label=0)
    
    preds_proba = np.array([1-preds, preds]).T
    ece = expected_calibration_error(preds_proba, y_test)
    
    # Ensure plots directory exists
    os.makedirs(save_dir, exist_ok=True)
    safe_strat_name = strategy_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'CM: {model_name}\n{strategy_name}')
    plt.savefig(f'{save_dir}/CM_{model_name}_{safe_strat_name}.png', bbox_inches='tight')
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, thresholds_roc = roc_curve(y_test, preds) 
    roc_auc = auc(fpr, tpr)
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {model_name}\n{strategy_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/ROC_{model_name}_{safe_strat_name}.png', bbox_inches='tight')
    plt.close()
        
    return {
        "Accuracy": acc * 100,
        "Min_Acc": min_acc * 100,
        "Maj_Acc": maj_acc * 100,
        "ECE": ece
    }

def find_meta_data(df, UNIQ_THRESHOLD=20):
    """segregating all columns into categorical (contains object type, integer
     and float type with lesser than UNIQ_THRESHOLD unique elements), integer type and float type"""

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    int_cols = [col for col in df.columns if df[col].dtype == 'int64']
    float_cols = [col for col in df.columns if df[col].dtype == 'float64']
    disc_int_cols = [col for col in int_cols if df[col].nunique() < UNIQ_THRESHOLD]
    disc_float_cols = [col for col in float_cols if df[col].nunique() < UNIQ_THRESHOLD]
    
    discrete_cols = cat_cols + disc_int_cols + disc_float_cols
    int_cols1 = [col for col in int_cols if col not in disc_int_cols]
    float_cols1 = [col for col in float_cols if col not in disc_float_cols]


    # Defining the metadata as is required by SDMetrics
    conti_cols = int_cols1 + float_cols1 
    metadata = dict()
    column_dict = dict()
    for col in conti_cols:
        column_dict[col] = {"sdtype": "numerical"}
    for col in discrete_cols:
        column_dict[col] = {"sdtype": "categorical"}
    metadata['columns']  = column_dict
    return metadata

def MachineLearningAccuracy(test, train, metadata, target_col):
    ada = BinaryAdaBoostClassifier.compute(test_data=test, train_data=train, target=target_col, metadata=metadata)
    dt = BinaryDecisionTreeClassifier.compute(test_data=test, train_data=train, target=target_col, metadata=metadata)
    lr = BinaryLogisticRegression.compute(test_data=test, train_data=train, target=target_col, metadata=metadata)
    mlp = BinaryMLPClassifier.compute(test_data=test, train_data=train, target=target_col, metadata=metadata)
    avg_score = (ada + dt + lr + mlp) / 4
    return {
        "Ada": ada,
        "DT": dt,
        "LR": lr,
        "MLP": mlp,
        "Avg": avg_score
    }

def load_and_clean_synthetic(filepath, target_col, forced_label):
    """Loads a synthetic dataset, assigns the correct target label, and drops the 'cond' column."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    df = pd.read_csv(filepath)
    df[target_col] = forced_label
    if 'cond' in df.columns:
        df = df.drop('cond', axis=1)
    return df

# ==========================================================
# EVALUATION PIPELINE
# ==========================================================

def run_evaluation(model_name, strategy_name, strat_id, train_df, test_df, X_test, y_test, target_col, fast_mode=False):
    print(f"  -> Evaluating Strategy {strat_id} ...", end=" ", flush=True)
    
    # XGBoost
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    xgb_model = xgboost_train(X_train, y_train, use_gpu=True)
    xgb_metrics = xgb_predict(xgb_model, X_test, y_test, model_name, strategy_name)
    
    # SDMetrics (Bypassable)
    if not fast_mode:
        metadata = find_meta_data(train_df)
        sdm_scores = MachineLearningAccuracy(test_df, train_df, metadata, target_col)
    else:
        sdm_scores = {"Ada": 0, "DT": 0, "LR": 0, "MLP": 0, "Avg": 0}
    
    print("Done!")
    return {
        "Model": model_name,
        "Strat": strat_id,
        "XG_Acc": f"{xgb_metrics['Accuracy']:.2f}%",
        "Min_Acc": f"{xgb_metrics['Min_Acc']:.2f}%",
        "Maj_Acc": f"{xgb_metrics['Maj_Acc']:.2f}%",
        "ECE": f"{xgb_metrics['ECE']:.4f}",
        "SD_Ada": f"{sdm_scores['Ada']:.4f}",
        "SD_DT":  f"{sdm_scores['DT']:.4f}",
        "SD_LR":  f"{sdm_scores['LR']:.4f}",
        "SD_MLP": f"{sdm_scores['MLP']:.4f}",
        "SD_Avg": f"{sdm_scores['Avg']:.4f}"
    }
    
# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--fast', action='store_true', help='Skip SDMetrics for much faster XGBoost-only prototyping')
    args = parser.parse_args()

    DATANAME = args.dataname
    TARGET = args.target
    FAST_MODE = args.fast
    
    if FAST_MODE:
        print("\n[WARNING] --fast flag enabled. Bypassing CPU-heavy SDMetrics.")

    print("\n[1/3] Loading Real Data Sets...")
    
    # 1. Test Data
    test = pd.read_csv(f'data/{DATANAME}/test_orig.csv')
    X_test = test.drop(TARGET, axis=1)
    y_test = test[TARGET]

    # 2. Imbalanced Base Data
    real_train = pd.read_csv(f'data/{DATANAME}/imbalanced_noord.csv')
    real_majority = real_train[real_train[TARGET] == 0]
    real_minority = real_train[real_train[TARGET] == 1]

    # 3. Imbalanced ORD Data
    real_ord = pd.read_csv(f'data/{DATANAME}/imbalanced_ord.csv')
    
    # Calculate Data Geometries
    real_C00 = real_ord[real_ord['cond'] == 0].copy()
    real_C00[TARGET] = 0
    real_C00.drop('cond', axis=1, inplace=True)
    
    N_maj = len(real_majority)               # Total Real Majority
    N_min = len(real_minority)               # Total Real Minority
    N_00 = len(real_C00)                     # Non-Overlap Majority
    N_01 = N_maj - N_00                      # Overlap Majority
    N_def = N_maj - N_min                    # Deficit to balance classes

    print(f"      Majority: {N_maj} | Minority: {N_min} | Deficit: {N_def}")
    print(f"      Clear Maj (C00): {N_00} | Overlap Maj (C01): {N_01}")
    
    # ==========================================================
    # EXPERIMENT REGISTRY 
    # Add your ablation models here!
    # ==========================================================
    SYNTHETIC_MODELS = {
        "Original_ORD": {
            "minority": f"data/{DATANAME}/synthetic/synthetic_minority_original.csv",
            "majority": f"data/{DATANAME}/synthetic/synthetic_majority_original.csv"
        },
        "Proposed_CFG": {
            "minority": f"data/{DATANAME}/synthetic/synthetic_minority_cfg.csv",
            "majority": f"data/{DATANAME}/synthetic/synthetic_majority_cfg.csv"
        }
    }
    
    all_results = []
    
    print("\n[2/3] Executing Augmentation Strategies...")
    for model_name, paths in SYNTHETIC_MODELS.items():
        print(f"\n--- Testing Model: {model_name} ---")
        
        try:
            synth_min_pool = load_and_clean_synthetic(paths["minority"], TARGET, forced_label=1)
            synth_maj_pool = load_and_clean_synthetic(paths["majority"], TARGET, forced_label=0)
            
            # ----------------------------------------------------------
            # STRATEGY 1: Real Maj + Real Min + Syn Min (Balanced)
            # ----------------------------------------------------------
            s1_synth_min = synth_min_pool.sample(n=N_def, random_state=42, replace=True)
            train_s1 = pd.concat([real_majority, real_minority, s1_synth_min])
            train_s1 = train_s1.sample(frac=1, random_state=42).reset_index(drop=True)
            
            res_s1 = run_evaluation(model_name, "RealMaj_RealMin_SynMin", "S1", train_s1, test, X_test, y_test, TARGET)
            all_results.append(res_s1)

            # ----------------------------------------------------------
            # STRATEGY 2: Syn Maj + Syn Min + Real Min (Balanced)
            # ----------------------------------------------------------
            s2_synth_maj = synth_maj_pool.sample(n=N_maj, random_state=42, replace=True)
            s2_synth_min = synth_min_pool.sample(n=N_def, random_state=42, replace=True)
            train_s2 = pd.concat([s2_synth_maj, real_minority, s2_synth_min])
            train_s2 = train_s2.sample(frac=1, random_state=42).reset_index(drop=True)
            
            res_s2 = run_evaluation(model_name, "SynMaj_SynMin_RealMin", "S2", train_s2, test, X_test, y_test, TARGET)
            all_results.append(res_s2)

            # ----------------------------------------------------------
            # STRATEGY 3: Real C00 + Syn C01 (compensate overlap) + Real Min + Syn Min
            # ----------------------------------------------------------
            s3_synth_maj = synth_maj_pool.sample(n=N_01, random_state=42, replace=True)
            s3_synth_min = synth_min_pool.sample(n=N_def, random_state=42, replace=True)
            train_s3 = pd.concat([real_C00, s3_synth_maj, real_minority, s3_synth_min])
            train_s3 = train_s3.sample(frac=1, random_state=42).reset_index(drop=True)
            
            res_s3 = run_evaluation(model_name, "RealC00_SynC01_RealMin_SynMin", "S3", train_s3, test, X_test, y_test, TARGET)
            all_results.append(res_s3)

        except FileNotFoundError as e:
            print(f"  -> Skipping {model_name} ({e})")
            
    # ==========================================================
    # FINAL REPORTING & STORAGE
    # ==========================================================
    print("\n[3/3] Final Aggregated Report")
    print("="*120)
    if len(all_results) > 0:
        report_df = pd.DataFrame(all_results)
        # Sort by Strategy, then Model for easy cross-model comparison
        report_df = report_df.sort_values(by=["Strat", "Model"]).reset_index(drop=True)
        print(report_df.to_string(index=False))
        
        # --- ROBUST STORAGE SYSTEM ---
        # 1. Create a dedicated results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 2. Generate a unique timestamp (e.g., 20231024_153022)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{results_dir}/augmentation_summary_{timestamp}.csv"
        
        # 3. Save the dataframe permanently
        report_df.to_csv(csv_filename, index=False)
        
        print(f"\n[SUCCESS] Full report securely saved to: {csv_filename}")
        print("[SUCCESS] All CM and ROC plots have been saved to the 'plots/' directory.")
    else:
        print("No evaluations completed. Check missing files.")
    print("="*120)
