"""
Helper functions for diffusion models.
"""
import torch
from typing import Optional, Any


def exists(val: Any) -> bool:
    """Check if a value exists (is not None)."""
    return val is not None


def default(val: Any, default_val: Any) -> Any:
    """Return val if it exists, otherwise return default_val."""
    return val if exists(val) else default_val


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract values from tensor 'a' at indices 't' and reshape for broadcasting.
    
    This is a crucial function in diffusion models! It allows us to extract
    the correct noise schedule parameters for each sample in a batch.
    
    Args:
        a: 1D tensor of values (e.g., alphas, betas) of shape (T,)
        t: Batch of timestep indices of shape (B,)
        x_shape: Shape of the data tensor (B, C, H, W)
    
    Returns:
        Tensor of shape (B, 1, 1, 1) for proper broadcasting
    
    Example:
        >>> alphas = torch.linspace(0.9, 0.1, 1000)  # Shape: (1000,)
        >>> t = torch.tensor([0, 100, 500])          # Shape: (3,)
        >>> x_shape = (3, 3, 32, 32)                 # Batch of 3 images
        >>> result = extract(alphas, t, x_shape)     # Shape: (3, 1, 1, 1)
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # Select values at indices t
    
    # Reshape to (B, 1, 1, 1) for broadcasting with (B, C, H, W)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    """Normalize image from [0, 1] to [-1, 1]."""
    return img * 2 - 1


def unnormalize_to_zero_to_one(img: torch.Tensor) -> torch.Tensor:
    """Unnormalize image from [-1, 1] to [0, 1]."""
    return (img + 1) / 2


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
