import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import argparse

def find_boundary(df, TARGET, RANDOM_STATE=42, threshold=0.4):
    print("="*50)
    print("Executing Overlap Region Detection (ORD)")
    print("="*50)

    # 1. Separate the classes cleanly
    df_class1 = df[df[TARGET] == 1].copy()
    df_class0 = df[df[TARGET] == 0].copy()
    
    # 2. Reset index of majority class to allow safe, mathematical splitting
    df_class0 = df_class0.reset_index(drop=True)
    
    # We will track boundaries directly in the safe df_class0 dataframe
    df_class0['isBoundary'] = 0
    
    mid = len(df_class0) // 2
    
    for i in range(2):
        print(f"\nEvaluating Split {i+1}/2...")
        
        # Use np.arange and .iloc to get exact positional rows safely
        if i == 0:
            test_idx = np.arange(0, mid)
            train_idx = np.arange(mid, len(df_class0))
        else:
            test_idx = np.arange(mid, len(df_class0))
            train_idx = np.arange(0, mid)
            
        test_maj = df_class0.iloc[test_idx]
        train_maj = df_class0.iloc[train_idx]
        
        # Train on half of the majority + ALL of the minority
        train = pd.concat([train_maj, df_class1], axis=0).reset_index(drop=True)
        # Test ONLY on the held-out majority
        test = test_maj.reset_index(drop=True)
        
        X_train = train.drop([TARGET, 'isBoundary'], axis=1, errors='ignore')
        y_train = train[TARGET]
        
        X_test = test.drop([TARGET, 'isBoundary'], axis=1, errors='ignore')
        y_test = test[TARGET]
        
        # Run Random Forest (n_estimators=50 matches your original RandomizedSearchCV goal)
        rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        proba_test = rf.predict_proba(X_test)
        
        # Identify boundaries in the test set
        boundary_list = []
        for j in range(len(y_test)):
            # Fallback to ensure we don't crash if RF only finds 1 class
            if len(rf.classes_) > 1:
                prob_0, prob_1 = proba_test[j][0], proba_test[j][1]
            else:
                prob_0 = 1.0 if rf.classes_[0] == 0 else 0.0
                prob_1 = 1.0 - prob_0
                
            # A point is overlap if misclassified (predicted 1) OR highly uncertain
            if (y_test.iloc[j] != y_pred[j]) or (prob_0 >= threshold and prob_1 >= threshold):
                # Save the actual index relative to df_class0
                boundary_list.append(test_maj.index[j])
        
        print(f" -> Found {len(boundary_list)} overlap points.")
        
        # Map the found boundary indices back to the master majority dataframe
        df_class0.loc[boundary_list, 'isBoundary'] = 1

    print("\n" + "-"*50)
    print("Final Target Geometry:")
    print(f"   Clear Majority (C00):   {len(df_class0[df_class0['isBoundary'] == 0])}")
    print(f"   Overlap Majority (C01): {len(df_class0[df_class0['isBoundary'] == 1])}")
    print(f"   Minority (C1):          {len(df_class1)}")
    print("-" * 50)

    # 3. Create the final ternary 'cond' column
    df_class0['cond'] = 0
    df_class0.loc[df_class0['isBoundary'] == 1, 'cond'] = 1
    df_class0 = df_class0.drop(['isBoundary', TARGET], axis=1)

    # Minority is safely and exclusively assigned cond=2
    df_class1['cond'] = 2
    df_class1 = df_class1.drop([TARGET], axis=1)

    # 4. Recombine into the final, geometrically perfect dataframe
    bnd = pd.concat([df_class0, df_class1], axis=0).reset_index(drop=True)
    return bnd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--target', type=str, default='income') # Fixed default from 'target' to 'income'
    args = parser.parse_args()

    DATANAME = args.dataname
    THRESHOLD = args.threshold
    TARGET = args.target

    path = f'data/{DATANAME}/imbalanced_noord.csv'
    
    try:
        df = pd.read_csv(path)
        bndry = find_boundary(df, TARGET, threshold=THRESHOLD)
        
        save_path = f'data/{DATANAME}/imbalanced_ord.csv'
        bndry.to_csv(save_path, index=False)
        print(f"\n[SUCCESS] Saved Ternary Target to {save_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {path}. Have you run preprocess.py yet?")