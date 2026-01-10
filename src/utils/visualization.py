"""
Visualization utilities for diffusion models.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from PIL import Image


def show_images(
    images: torch.Tensor,
    num_images: int = 16,
    nrow: int = 4,
    title: Optional[str] = None,
    figsize: tuple = (10, 10)
):
    """
    Display a grid of images.
    
    Args:
        images: Tensor of shape (B, C, H, W) with values in [-1, 1] or [0, 1]
        num_images: Number of images to display
        nrow: Number of images per row
        title: Optional title for the figure
        figsize: Figure size
    """
    images = images[:num_images].cpu()
    
    # Normalize to [0, 1] if needed
    if images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Create grid
    ncol = (num_images + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx].permute(1, 2, 0).numpy()
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_forward_process(
    image: torch.Tensor,
    diffusion,
    timesteps: List[int] = None,
    figsize: tuple = (15, 3)
):
    """
    Visualize the forward diffusion process on a single image.
    
    Args:
        image: Single image tensor of shape (C, H, W)
        diffusion: GaussianDiffusion instance
        timesteps: List of timesteps to visualize
        figsize: Figure size
    """
    if timesteps is None:
        timesteps = [0, 50, 100, 200, 500, 999]
    
    fig, axes = plt.subplots(1, len(timesteps), figsize=figsize)
    
    for idx, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=image.device)
        noisy_image, _ = diffusion.q_sample(image.unsqueeze(0), t_tensor)
        
        img = noisy_image[0].cpu()
        img = (img + 1) / 2  # Denormalize
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(img)
        
        axes[idx].set_title(f't = {t}')
        axes[idx].axis('off')
    
    plt.suptitle('Forward Diffusion Process: Adding Noise Over Time', fontsize=12)
    plt.tight_layout()
    plt.show()


def show_reverse_process(
    samples_history: List[torch.Tensor],
    timesteps: List[int] = None,
    num_images: int = 4,
    figsize: tuple = (15, 8)
):
    """
    Visualize the reverse diffusion process (denoising).
    
    Args:
        samples_history: List of sample tensors at different timesteps
        timesteps: List of timesteps corresponding to samples_history
        num_images: Number of samples to show
        figsize: Figure size
    """
    if timesteps is None:
        total = len(samples_history)
        indices = [0, total//4, total//2, 3*total//4, total-1]
        timesteps = [len(samples_history) - 1 - i for i in indices]
    
    fig, axes = plt.subplots(num_images, len(timesteps), figsize=figsize)
    
    for row in range(num_images):
        for col, t_idx in enumerate(range(0, len(samples_history), len(samples_history)//len(timesteps) or 1)[:len(timesteps)]):
            img = samples_history[t_idx][row].cpu()
            img = (img + 1) / 2
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[row, col].imshow(img, cmap='gray')
            else:
                axes[row, col].imshow(img)
            
            if row == 0:
                axes[row, col].set_title(f't ≈ {len(samples_history) - t_idx}')
            axes[row, col].axis('off')
    
    plt.suptitle('Reverse Diffusion Process: Denoising Over Time', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_noise_schedule(betas: torch.Tensor, figsize: tuple = (12, 4)):
    """
    Visualize the noise schedule parameters.
    
    Args:
        betas: Beta schedule tensor
        figsize: Figure size
    """
    betas = betas.cpu().numpy()
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Beta schedule
    axes[0].plot(betas)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('β(t)')
    axes[0].set_title('Beta Schedule')
    
    # Alpha schedule
    axes[1].plot(alphas)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('α(t) = 1 - β(t)')
    axes[1].set_title('Alpha Schedule')
    
    # Cumulative alpha (signal remaining)
    axes[2].plot(alphas_cumprod)
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('ᾱ(t)')
    axes[2].set_title('Cumulative Alpha (Signal Remaining)')
    
    plt.tight_layout()
    plt.show()
