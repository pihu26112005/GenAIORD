import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import optuna
import argparse
import os
import json
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tabsyn.vae.model import Model_VAE, Encoder_model, LatentStructuringLoss
from utils_train import preprocess

warnings.filterwarnings('ignore')

# --- HYPERPARAMETERS ---
LR = 1e-3
WD = 0
D_TOKEN = 4
N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2
EPOCHS_OPTUNA = 300 # Reduced epochs for faster Optuna search

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    
    return mse_loss, ce_loss, loss_kld

def quick_latent_classifier(train_z, y_train):
    """Probes the latent space to see if the classes are linearly separable."""
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    # 3-fold cross validation for a robust F1 score
    scores = cross_val_score(clf, train_z, y_train, cv=3, scoring='f1')
    return scores.mean()

def train_vae_for_optuna(trial, args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    device = args.device

    # 1. Suggest lambda and margin values
    lambda_struct = trial.suggest_float('lambda_struct', 0.1, 2.0)
    # FIX: Since we use mean-scaled distance, expected distance is ~1.4. 
    # A boundary between 1.0 and 5.0 is the correct scale.
    margin_val = trial.suggest_float('margin', 1.0, 5.0) 
    
    print(f"\n--- Testing lambda = {lambda_struct:.3f} | margin = {margin_val:.3f} ---")

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    # 2. Extract Data
    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type=info['task_type'])
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # Obtain y_train directly from CSV
    target_col = 'income' 
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    y_train = train_df[target_col].values
    
    y_train_tensor = torch.tensor(y_train).float()
    X_train_num_t = torch.tensor(X_train_num).float()
    X_train_cat_t = torch.tensor(X_train_cat)

    train_data = TensorDataset(X_train_num_t, X_train_cat_t, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=4096, shuffle=True, num_workers=4)

    # 3. Initialize Models
    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR, bias=True).to(device)
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    latent_dim_flat = (d_numerical + len(categories)) * D_TOKEN
    
    # FIX: Explicitly pass the Optuna 'margin_val' and our optimized 'alpha' into the Loss Module
    structuring_criterion = LatentStructuringLoss(
        latent_dim=latent_dim_flat, 
        margin=margin_val, 
        alpha=0.90
    ).to(device)

    # 4. Training Loop
    model.train()
    beta = 1e-2

    final_recon_loss = 0.0

    for epoch in range(EPOCHS_OPTUNA):
        epoch_mse = 0
        epoch_ce = 0
        batches = 0

        for batch_num, batch_cat, batch_y in train_loader:
            optimizer.zero_grad()
            batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
            loss_mse, loss_ce, loss_kld = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            mu_z_flat = mu_z.view(mu_z.shape[0], -1)
            struct_loss = structuring_criterion(mu_z_flat, batch_y)
            
            loss = loss_mse + loss_ce + (beta * loss_kld) + (lambda_struct * struct_loss)
            loss.backward()
            optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_ce += loss_ce.item()
            batches += 1

        final_recon_loss = (epoch_mse + epoch_ce) / batches

    # 5. Extract Latent Embeddings (train_z)
    model.eval()
    pre_encoder.load_weights(model)
    with torch.no_grad():
        train_z = pre_encoder(X_train_num_t.to(device), X_train_cat_t.to(device)).detach().cpu().numpy()
        train_z_flat = train_z.reshape(train_z.shape[0], -1)

    # 6. Evaluate Separation
    separation_f1 = quick_latent_classifier(train_z_flat, y_train)
    print(f"Result: Recon Penalty = {final_recon_loss:.4f} | Separation F1 = {separation_f1:.4f}")

    # 7. Composite Score
    # FIX: Reduced the F1 multiplier from 5.0 to 2.0 to prevent Optuna from entirely abandoning 
    # the reconstruction loss just to gain a tiny fraction of separation F1.
    composite_score = final_recon_loss - (2.0 * separation_f1)
    
    return composite_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Run the Optuna Study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: train_vae_for_optuna(trial, args), n_trials=30)

    # FIX: Print both of the winning hyperparameters
    print("\n" + "="*50)
    print(f"BEST LAMBDA: {study.best_params['lambda_struct']:.4f}")
    print(f"BEST MARGIN: {study.best_params['margin']:.4f}")
    print("="*50)