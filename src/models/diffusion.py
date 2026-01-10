"""
Gaussian Diffusion implementation from scratch.

This is the core of DDPM (Denoising Diffusion Probabilistic Models).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from tqdm import tqdm

from ..utils.helpers import extract, default


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model implementing DDPM.
    
    The key insight of diffusion models:
    1. Forward process: Gradually add noise to data (fixed, not learned)
    2. Reverse process: Learn to denoise step by step
    
    Mathematical foundation:
    - Forward: q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
    - Reverse: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    """
    
    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Initialize the diffusion model.
        
        Args:
            model: The neural network (usually UNet) that predicts noise
            image_size: Size of the images (assumes square images)
            timesteps: Number of diffusion steps (T)
            beta_schedule: Type of noise schedule ('linear' or 'cosine')
            beta_start: Starting value of beta
            beta_end: Ending value of beta
        """
        super().__init__()
        
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        
        # Create the noise schedule
        # β_t controls how much noise is added at each step
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        # These are all derived from betas and are used throughout training/sampling
        alphas = 1.0 - betas                          # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{s=1}^t α_s
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}
        
        # Register as buffers (saved with model, moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Quantities for q(x_t | x_0) - the forward process
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Quantities for the posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Clipped log variance for numerical stability
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )
        
        # Coefficients for posterior mean
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
        
        This schedule provides better quality at high noise levels.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Add noise to x_0 to get x_t directly (not step by step).
        
        The key equation: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        
        Args:
            x_start: Clean images x_0, shape (B, C, H, W)
            t: Timesteps, shape (B,)
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (noisy images x_t, noise ε)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get the scaling factors for this batch of timesteps
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Apply the forward process equation
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy, noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the training loss.
        
        The model learns to predict the noise that was added.
        Loss = ||ε - ε_θ(x_t, t)||²
        
        Args:
            x_start: Clean images, shape (B, C, H, W)
            t: Randomly sampled timesteps, shape (B,)
            noise: Optional pre-generated noise
            
        Returns:
            Mean squared error loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get noisy images
        x_noisy, _ = self.q_sample(x_start, t, noise)
        
        # Predict the noise
        predicted_noise = self.model(x_noisy, t)
        
        # Simple L2 loss between true and predicted noise
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        
        Randomly samples timesteps and computes the loss.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps uniformly
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        return self.p_losses(x, t)
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: p_θ(x_{t-1} | x_t)
        
        Uses the model to predict noise, then computes the previous sample.
        
        Args:
            x: Current noisy samples x_t, shape (B, C, H, W)
            t: Current timestep tensor, shape (B,)
            t_index: Current timestep as integer
            
        Returns:
            Denoised samples x_{t-1}
        """
        # Get model's noise prediction
        predicted_noise = self.model(x, t)
        
        # Get the required coefficients for this timestep
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Compute the mean of p(x_{t-1} | x_t)
        # μ_θ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            # No noise at the final step
            return model_mean
        else:
            # Add noise scaled by posterior variance
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        channels: int = 3,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using the reverse diffusion process.
        
        Starting from pure noise, iteratively denoise to get clean images.
        
        Args:
            batch_size: Number of samples to generate
            channels: Number of image channels
            return_all_timesteps: If True, return samples at all timesteps
            
        Returns:
            Generated images, shape (B, C, H, W) or list of such tensors
        """
        device = self.betas.device
        shape = (batch_size, channels, self.image_size, self.image_size)
        
        # Start from pure Gaussian noise
        img = torch.randn(shape, device=device)
        
        imgs = [img] if return_all_timesteps else None
        
        # Reverse diffusion process: t = T-1, T-2, ..., 0
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            
            if return_all_timesteps:
                imgs.append(img)
        
        return imgs if return_all_timesteps else img
    
    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int = 16,
        channels: int = 3,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM sampling for faster generation.
        
        DDIM allows skipping timesteps, trading quality for speed.
        eta=0 gives deterministic sampling, eta=1 gives DDPM.
        
        Args:
            batch_size: Number of samples to generate
            channels: Number of image channels
            ddim_steps: Number of sampling steps (can be much less than timesteps)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
            
        Returns:
            Generated images
        """
        device = self.betas.device
        shape = (batch_size, channels, self.image_size, self.image_size)
        
        # Create a subsequence of timesteps
        times = torch.linspace(-1, self.timesteps - 1, ddim_steps + 1, device=device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn(shape, device=device)
        
        for time, time_prev in tqdm(time_pairs, desc='DDIM Sampling'):
            t = torch.full((batch_size,), time, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(img, t)
            
            # Get alpha values
            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev] if time_prev >= 0 else torch.tensor(1.0)
            
            # DDIM update step
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            
            # Predict x_0
            pred_x0 = (img - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
            
            # Random noise
            noise = torch.randn_like(img) if time_prev > 0 else 0
            
            img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
        
        return img
