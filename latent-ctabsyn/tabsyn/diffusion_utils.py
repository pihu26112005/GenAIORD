"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from scipy.stats import betaprime
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

#####
def sample(net, num_samples, dim, label, mu_01=None, gamma_penalty=0.1, num_steps = 50, device = 'cuda:0'):
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, label, mu_01, gamma_penalty)
            #####

    return x_next

def sample_step(net, num_steps, i, t_cur, t_next, x_next, label, mu_01=None, gamma_penalty=0.1):

    x_cur = x_next
    # Increase noise temporarily.
    gamma_churn = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma_churn * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    #####
    denoised = net(x_hat, t_hat, label).to(torch.float32)
    
    # --- NEW: LATENT REPULSION ENERGY GRADIENT ---
    repulsion_grad = 0
    # # Only repel if we are synthesizing minority points (where label is 1.0)
    # if mu_01 is not None and torch.all(label == 1.0):
    #     epsilon = 1e-5
    #     # Energy E = 1 / (||x_hat - mu_01||^2 + eps)
    #     dist_sq = torch.sum((x_hat - mu_01)**2, dim=1, keepdim=True)
    #     # Gradient of E w.r.t x_hat
    #     energy_grad = -2 * (x_hat - mu_01) / ((dist_sq + epsilon)**2)
    #     repulsion_grad = gamma_penalty * energy_grad
    
    # FIX: Add gamma_penalty > 0.0 to short-circuit the NaN trap
    if gamma_penalty > 0.0 and mu_01 is not None and torch.all(label == 1.0):
        epsilon = 1e-5
        dist_sq = torch.sum((x_hat - mu_01)**2, dim=1, keepdim=True)
        energy_grad = -2 * (x_hat - mu_01) / ((dist_sq + epsilon)**2)
        repulsion_grad = gamma_penalty * energy_grad
    
    # d_cur = (x_hat - denoised) / t_hat
    d_cur = (x_hat - denoised) / t_hat - repulsion_grad
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        #####
        denoised = net(x_next, t_next, label).to(torch.float32)
        
        repulsion_grad_prime = 0
        # if mu_01 is not None and torch.all(label == 1.0):
        #     dist_sq_prime = torch.sum((x_next - mu_01)**2, dim=1, keepdim=True)
        #     energy_grad_prime = -2 * (x_next - mu_01) / ((dist_sq_prime + epsilon)**2)
        #     repulsion_grad_prime = gamma_penalty * energy_grad_prime
        
        if gamma_penalty > 0.0 and mu_01 is not None and torch.all(label == 1.0):
            epsilon = 1e-5
            dist_sq_prime = torch.sum((x_next - mu_01)**2, dim=1, keepdim=True)
            energy_grad_prime = -2 * (x_next - mu_01) / ((dist_sq_prime + epsilon)**2)
            repulsion_grad_prime = gamma_penalty * energy_grad_prime
            
        # d_prime = (x_next - denoised) / t_next
        d_prime = (x_next - denoised) / t_next - repulsion_grad_prime
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data, label):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma, label)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss

