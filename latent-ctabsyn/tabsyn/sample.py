import torch

import os
import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    save_path = args.save_path
    cond_val = args.condition_by

    #####
    # if cond_val == 0 , one_hot = 1,0,0
    # if cond_val == 1 , one_hot = 0,1,0
    # if cond_val == 2 , one_hot = 0,0,1
    # one_hot = torch.zeros(n_classes)
    # one_hot[cond_val] = 1
 
    # label = torch.tensor(one_hot)
    # label_dim = label.shape[0]
    # label = label.to(device)
    #####
    
    # # We are now using continuous weighting. 1.0 for minority, 0.0 for majority
    # if cond_val == 1:
    #     label = torch.ones((args.steps if args.steps else 10000, 1)) # Assuming you generate steps amount of data
    # else:
    #     label = torch.zeros((args.steps if args.steps else 10000, 1))
        
    # label = label.to(device)
    
    label_dim = 1

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 
    mean = train_z.mean(0)
    
    # 1. Ensure the number of generated samples is strictly defined
    num_samples = train_z.shape[0]
    
    # 2. Create the continuous labels matching the exact num_samples
    if cond_val == 1:
        label = torch.ones((num_samples, 1), device=device) 
    else:
        label = torch.zeros((num_samples, 1), device=device)
        
    # 3. Load the overlap centroid (mu_01) for Latent Repulsion
    mu_01_path = f'{ckpt_path}/mu_01.pt'
    if os.path.exists(mu_01_path):
        mu_01 = torch.load(mu_01_path).to(device)
        print("Loaded mu_01 centroid for Latent Repulsion.")
    else:
        mu_01 = None
        print("No mu_01 centroid found. Running without Latent Repulsion.")
    
    ######
    denoise_fn = MLPDiffusion(in_dim, label_dim, 1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    #####
    
    # 4. CRITICAL: PASS mu_01 AND gamma_penalty TO THE SAMPLER
    # gamma_penalty = 0.1 # Tuning parameter for the strength of repulsion
    gamma_penalty = 0.1
    
    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, label, mu_01=mu_01, gamma_penalty=gamma_penalty, device=device)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    
    # Replaced n_classes with condition_by
    parser.add_argument('--condition_by', type=int, default=1, help='1 for minority, 0 for majority')
    parser.add_argument('--save_path', type=str, default='syn_data.csv', help='Path to save the generated data.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'