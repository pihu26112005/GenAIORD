from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tabsyn.diffusion_utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, n_classes=2, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.label_embed = nn.Sequential(
            nn.Linear(n_classes, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, label, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        #####
        label = label.float()
        label_emb = self.label_embed(label)
        #####
        x = self.proj(x) + emb + label_emb
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, label):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten(), label)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x, label):
        loss = self.loss_fn(self.denoise_fn_D, x, label)
        return loss.mean(-1).mean()
    

# ====== ADD THIS CFG WRAPPER CLASS ======
class CFGWrapper(torch.nn.Module):
    def __init__(self, model, gamma):
        super().__init__()
        self.model = model
        self.gamma = gamma
    
    # ====== PASS-THROUGH PROPERTIES FOR THE SOLVER ======
    @property
    def sigma_min(self):
        return self.model.sigma_min
        
    @property
    def sigma_max(self):
        return self.model.sigma_max
        
    def round_sigma(self, sigma):
        return self.model.round_sigma(sigma)
    # ====================================================

    def forward(self, x, sigma, label):
        # 1. Get the unconditional prediction (null label)
        null_label = torch.zeros_like(label)
        uncond_pred = self.model(x, sigma, null_label)
        
        # 2. Get the conditional prediction (actual label)
        cond_pred = self.model(x, sigma, label)
        
        # 3. Extrapolate (The CFG Math)
        return uncond_pred + self.gamma * (cond_pred - uncond_pred)
# ========================================


class AdvancedCFGWrapper(torch.nn.Module):
    def __init__(self, model, gamma_maj=1.1, gamma_min_start=0.9, gamma_min_end=1.0):
        super().__init__()
        self.model = model
        
        # Static gamma for Clear Majority (C00) -> Keeps them tight and precise
        self.gamma_maj = gamma_maj
        
        # Dynamic gamma bounds for Overlap (C01) and Minority (C1) -> Prevents starvation early, enforces boundary late
        self.gamma_min_start = gamma_min_start
        self.gamma_min_end = gamma_min_end
    
    # ====== PASS-THROUGH PROPERTIES ======
    @property
    def sigma_min(self):
        return self.model.sigma_min
        
    @property
    def sigma_max(self):
        return self.model.sigma_max
        
    def round_sigma(self, sigma):
        return self.model.round_sigma(sigma)
    # =====================================

    def forward(self, x, sigma, label):
        # 1. Get unconditional and conditional predictions
        null_label = torch.zeros_like(label)
        uncond_pred = self.model(x, sigma, null_label)
        cond_pred = self.model(x, sigma, label)
        
        # 2. Robustly identify if the target is Clear Majority (Class 0)
        is_maj = label[..., 0:1]
            
        # 3. Calculate diffusion progress (1.0 at start/high noise -> 0.0 at end/low noise)
        progress = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        progress = progress.view(-1, 1) # Reshape for tensor broadcasting
        
        # 4. Calculate Timestep-Scheduled Gamma for Minority/Overlap
        # Math: 1.0 - ((1.0 - 0.88) * progress)
        # Evaluates to 0.88 when progress is 1.0
        # Evaluates to 1.0 when progress is 0.0
        gamma_min_t = self.gamma_min_end - ((self.gamma_min_end - self.gamma_min_start) * progress)
        
        # 5. Blend the Asymmetric Gammas
        # Majority points get the constant gamma_maj.
        # Minority/Overlap points get the dynamic gamma_min_t.
        gamma = (is_maj * self.gamma_maj) + ((1.0 - is_maj) * gamma_min_t)
        
        # 6. Apply CFG Math
        return uncond_pred + gamma * (cond_pred - uncond_pred)