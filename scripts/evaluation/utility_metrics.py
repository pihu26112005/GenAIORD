import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
from sdmetrics.single_table import (BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier)
import warnings

warnings.simplefilter(action='ignore', category=Warning)

def expected_calibration_error(samples, true_labels, M=10, threshold=0.5):
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(samples, axis=1)
    predicted_label = (np.array([x[1] for x in samples]) >= threshold).astype(int)
    accuracies = predicted_label == true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

def find_meta_data(df, UNIQ_THRESHOLD=20):
    """Segregating columns into categorical and continuous for SDMetrics."""
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    int_cols = [col for col in df.columns if df[col].dtype in ['int64', 'int32']]
    float_cols = [col for col in df.columns if df[col].dtype in ['float64', 'float32']]
    
    disc_int_cols = [col for col in int_cols if df[col].nunique() < UNIQ_THRESHOLD]
    disc_float_cols = [col for col in float_cols if df[col].nunique() < UNIQ_THRESHOLD]
    
    discrete_cols = cat_cols + disc_int_cols + disc_float_cols
    int_cols1 = [col for col in int_cols if col not in disc_int_cols]
    float_cols1 = [col for col in float_cols if col not in disc_float_cols]

    conti_cols = int_cols1 + float_cols1 
    metadata = {'columns': {}}
    
    for col in conti_cols:
        metadata['columns'][col] = {"sdtype": "numerical"}
    for col in discrete_cols:
        metadata['columns'][col] = {"sdtype": "categorical"}
        
    return metadata

def calculate_sdmetrics(train_df, test_df, target_col):
    """Calculates all SDMetrics ML efficacy scores."""
    print("      -> Running SDMetrics Classifiers (This may take a moment)...")
    metadata = find_meta_data(train_df)
    
    try:
        ada = BinaryAdaBoostClassifier.compute(test_data=test_df, train_data=train_df, target=target_col, metadata=metadata)
        dt = BinaryDecisionTreeClassifier.compute(test_data=test_df, train_data=train_df, target=target_col, metadata=metadata)
        lr = BinaryLogisticRegression.compute(test_data=test_df, train_data=train_df, target=target_col, metadata=metadata)
        mlp = BinaryMLPClassifier.compute(test_data=test_df, train_data=train_df, target=target_col, metadata=metadata)
        
        avg_score = (ada + dt + lr + mlp) / 4
        
        return {
            "SD_Ada": ada,
            "SD_DT": dt,
            "SD_LR": lr,
            "SD_MLP": mlp,
            "SD_Avg": avg_score
        }
        
    except Exception as e:
        # If the generative model missed categories, SDMetrics will fail. 
        # We catch it here so it doesn't break the whole loop.
        print(f"         [!] SDMetrics evaluation failed: {e}")
        print("         [!] Defaulting SDMetrics scores to 0.0 for this model.")
        return {
            "SD_Ada": 0.0,
            "SD_DT": 0.0,
            "SD_LR": 0.0,
            "SD_MLP": 0.0,
            "SD_Avg": 0.0
        }

def calculate_utility(X_train, y_train, X_test, y_test, use_gpu=True, RANDOM_SEED=42):
    """Trains an XGBoost model on augmented data and evaluates on real test data."""
    xgb_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    xgb_test = xgb.DMatrix(X_test, enable_categorical=True)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': RANDOM_SEED  # <--- Add this to lock the model's random state
    }
    if use_gpu:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'

    # Train model
    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=200)
    
    # Predict
    preds_prob = model.predict(xgb_test)
    y_pred = (preds_prob >= 0.5).astype(int)
    
    # Format probas for ECE
    preds_proba_2d = np.array([1-preds_prob, preds_prob]).T

    return {
        "XG_Acc": accuracy_score(y_test, y_pred) * 100,
        "Min_Acc": recall_score(y_test, y_pred, pos_label=1) * 100,
        "Maj_Acc": recall_score(y_test, y_pred, pos_label=0) * 100,
        "Macro_F1": f1_score(y_test, y_pred, average='macro'),
        "G_Mean": geometric_mean_score(y_test, y_pred),
        "AUPRC": average_precision_score(y_test, preds_prob),
        "ECE": expected_calibration_error(preds_proba_2d, y_test)
    }