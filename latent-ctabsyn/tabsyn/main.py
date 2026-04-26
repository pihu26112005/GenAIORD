import os
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If using multi-GPU
    
    # Force deterministic operations in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args): 
    seed_everything(42)
    
    device = args.device

    train_z, _, dataset_dir, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    #####
    # get labels/
    label_path = f'{dataset_dir}/y_train.npy'
    label = np.load(label_path)
    label = label.reshape(-1)
    # label_hot = np.zeros((label.shape[0], len(np.unique(label))), dtype=int)
    # label_hot[np.arange(label.size), label] = 1
    # label_dim = label_hot.shape[1] # n_classes
    #####

    # Safely convert label to a PyTorch tensor for indexing
    label_tensor = torch.tensor(label)

    # 1. Latent-Space Overlap Detection & Continuous Weighting
    minority_z = train_z[label_tensor == 1]
    
    # Calculate minority centroid (mu_1) and variance (sigma^2)
    mu_1 = minority_z.mean(dim=0)
    dist_sq = torch.sum((minority_z - mu_1)**2, dim=1)
    sigma_sq = dist_sq.mean()  # Isotropic variance approximation
    
    # Calculate continuous overlap severity w(z) for all points using RBF kernel
    all_dist_sq = torch.sum((train_z - mu_1)**2, dim=1)
    w_z = torch.exp(-all_dist_sq / (2 * sigma_sq)).unsqueeze(1)
    
    # Force pure minority points to exactly 1.0 to anchor the distribution
    w_z[label_tensor == 1, 0] = 1.0 
    
    # Define the overlap region (D_01) to find its centroid for repulsion
    overlap_mask = (label == 0) & (w_z.squeeze() > 0.3).numpy()
    mu_01 = train_z[overlap_mask].mean(dim=0)
    
    # Save mu_01 so the sampling script can access it for the repulsion gradient
    torch.save(mu_01, f'{ckpt_path}/mu_01.pt')

    in_dim = train_z.shape[1] 
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    
    # Concatenate the continuous 1D weight instead of one-hot labels
    train_data = torch.cat([train_z, w_z], dim=1).numpy()
    
    label_dim = 1 # Now a 1D continuous scalar instead of n_classes

    #####
    # Fix the Weighted Sampler logic by using the original number of classes
    n_original_classes = len(np.unique(label))
    class_counts = [0] * n_original_classes
    for i in label:
        class_counts[i] += 1
    
    sample_weights = [0] * len(label)
    for i in range(len(label)):
        sample_weights[i] = np.log(class_counts[label[i]])
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    #####
    
    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        sampler = sampler, # DON'T FORGET TO ADD THE SAMPLER BACK HERE! (It was missing in your snippet)
        num_workers = 4,
    )

    num_epochs = 10000 + 1
    #####
    denoise_fn = MLPDiffusion(in_dim, label_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            #####
            label = inputs[:,-label_dim:]
            inputs = inputs[:,:-label_dim]
            #####
            loss = model(inputs, label)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'