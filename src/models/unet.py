"""
UNet Architecture for Diffusion Models.

This is the neural network that learns to predict noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for timesteps.
    
    Same idea as positional encoding in Transformers!
    This helps the network understand "how noisy" the input is.
    
    The embedding uses sin and cos functions at different frequencies,
    allowing the model to distinguish between different timesteps.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timestep tensor of shape (B,)
            
        Returns:
            Embeddings of shape (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Compute frequencies: exp(-log(10000) * i / (d/2))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # time: (B,) -> (B, 1), embeddings: (d/2,) -> (1, d/2)
        # Result: (B, d/2)
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos: (B, d)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class Block(nn.Module):
    """
    Basic convolutional block with GroupNorm and SiLU activation.
    
    GroupNorm is preferred over BatchNorm in diffusion models because:
    - Works well with small batch sizes
    - More stable for generative models
    """
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) works great for diffusion
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with time embedding injection.
    
    The time embedding is added to the intermediate features,
    allowing the network to adapt its behavior based on the noise level.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8
    ):
        super().__init__()
        
        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups)
        
        # Project time embedding to match channel dimension
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Skip connection (identity or 1x1 conv if dimensions differ)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, C, H, W)
            time_emb: Time embeddings, shape (B, time_emb_dim)
        """
        h = self.block1(x)
        
        # Add time embedding (broadcast over spatial dimensions)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.skip(x)


class Attention(nn.Module):
    """
    Self-attention layer for capturing long-range dependencies.
    
    Added at lower resolutions where it's computationally feasible.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention: softmax(QK^T / sqrt(d)) * V
        attn = torch.einsum('bnci,bncj->bnij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bnij,bncj->bnci', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class Downsample(nn.Module):
    """Downsample spatial dimensions by 2x using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample spatial dimensions by 2x using nearest neighbor + conv."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for diffusion models.
    
    Key features:
    - Encoder-decoder structure with skip connections
    - Time embeddings injected at each resolution
    - Self-attention at lower resolutions
    - Residual connections throughout
    
    The UNet predicts the noise ε that was added to the image.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mults: List[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (2, 4),
        num_heads: int = 4,
        groups: int = 8
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            out_channels: Output channels (same as input for noise prediction)
            model_channels: Base channel count
            channel_mults: Channel multipliers at each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Which levels get attention (0=highest res)
            num_heads: Number of attention heads
            groups: Groups for GroupNorm
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        channels = model_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(channels, out_ch, time_emb_dim, groups))
                channels = out_ch
                
                # Add attention at specified resolutions
                if level in attention_resolutions:
                    self.downs.append(Attention(channels, num_heads))
            
            # Downsample (except at the last level)
            if level < len(channel_mults) - 1:
                self.downs.append(Downsample(channels))
        
        # Middle (bottleneck)
        self.mid = nn.ModuleList([
            ResnetBlock(channels, channels, time_emb_dim, groups),
            Attention(channels, num_heads),
            ResnetBlock(channels, channels, time_emb_dim, groups)
        ])
        
        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                # Skip connection from encoder doubles channels
                skip_channels = channels if i == 0 else out_ch
                self.ups.append(ResnetBlock(channels + skip_channels, out_ch, time_emb_dim, groups))
                channels = out_ch
                
                if level in attention_resolutions:
                    self.ups.append(Attention(channels, num_heads))
            
            # Upsample (except at the last level)
            if level < len(channel_mults) - 1:
                self.ups.append(Upsample(channels))
        
        # Output
        self.final_block = Block(channels, channels, groups)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)
        
        # Store encoder outputs for skip connections
        self._downs_features = []
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise.
        
        Args:
            x: Noisy images, shape (B, C, H, W)
            t: Timesteps, shape (B,)
            
        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Store for skip connections
        skip_connections = [x]
        
        # Encoder
        for module in self.downs:
            if isinstance(module, ResnetBlock):
                x = module(x, t_emb)
                skip_connections.append(x)
            elif isinstance(module, Attention):
                x = module(x)
            else:  # Downsample
                x = module(x)
                skip_connections.append(x)
        
        # Middle
        for module in self.mid:
            if isinstance(module, ResnetBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        
        # Decoder
        for module in self.ups:
            if isinstance(module, ResnetBlock):
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
                x = module(x, t_emb)
            elif isinstance(module, Attention):
                x = module(x)
            else:  # Upsample
                x = module(x)
        
        # Output
        x = self.final_block(x)
        x = self.final_conv(x)
        
        return x
