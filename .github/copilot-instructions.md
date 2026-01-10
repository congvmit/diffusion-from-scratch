# Diffusion Models from Scratch - Copilot Instructions

## Project Overview
An educational codebase for learning diffusion models (DDPM) from scratch with PyTorch. The project emphasizes understanding over abstraction—every component is implemented with detailed comments explaining the mathematics.

## Architecture

### Key Components
- `src/models/diffusion.py` - `GaussianDiffusion` class implementing forward/reverse processes
- `src/models/unet.py` - `UNet` noise prediction network with time conditioning
- `src/utils/helpers.py` - Critical `extract()` function for schedule indexing
- `notebooks/01_diffusion_mentor.ipynb` - Interactive learning notebook

### Core Mathematical Pattern
All diffusion operations rely on the `extract(a, t, x_shape)` pattern:
```python
# Extract schedule values at timesteps t, reshape for broadcasting
sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x.shape)  # (B,) -> (B,1,1,1)
```

## Code Conventions

### Tensor Shapes
- Images: `(B, C, H, W)` normalized to `[-1, 1]`
- Timesteps: `(B,)` as `torch.long`, range `[0, T)`
- Schedule tensors: `(T,)` - use `extract()` to index

### Naming Conventions
- `x_0` - clean data, `x_t` - noisy data at timestep t
- `betas` (β), `alphas` (α), `alphas_cumprod` (ᾱ) - noise schedule
- `q_sample` - forward process, `p_sample` - reverse step

### Model Design
- Use `nn.GroupNorm` (not BatchNorm) for stable training
- Time embeddings via sinusoidal encoding → MLP → add to features
- SiLU activation throughout (better than ReLU for diffusion)

## Development Workflow

### Package Management (uv + pyproject.toml)
This project uses `uv` with `pyproject.toml` for modern Python packaging:
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac

# Install package in editable mode with dependencies
uv pip install -e ".[notebook]"  # Core + Jupyter
uv pip install -e ".[all]"       # Everything including dev tools

# Add new packages: update pyproject.toml then reinstall
uv pip install -e ".[all]"
```

### Running Notebooks
```bash
uv run jupyter notebook notebooks/
```

### Training
Default config trains on MNIST at 32×32. Key hyperparameters:
- `T = 1000` timesteps (standard DDPM)
- `beta_start = 1e-4`, `beta_end = 0.02` (linear schedule)
- Learning rate: `1e-3` with cosine annealing

### Testing Changes
Always verify the forward process produces correct noise levels:
```python
# At t=0: x_t ≈ x_0, at t=T-1: x_t ≈ noise
assert alphas_cumprod[0] > 0.99
assert alphas_cumprod[-1] < 0.01
```

## Common Patterns

### Adding New Schedule Types
Add to `GaussianDiffusion.__init__`:
```python
if beta_schedule == 'your_schedule':
    betas = your_schedule_fn(timesteps)
```

### Modifying UNet
The architecture follows encoder→bottleneck→decoder with skip connections. When adding layers:
1. Add to both encoder and corresponding decoder level
2. Include time embedding injection via `ResnetBlock`
3. Match channel dimensions for skip connections

## Gotchas
- Always call `.to(device)` on schedule tensors before training
- Normalize images to `[-1, 1]`, not `[0, 1]`
- Loss ~1.0 for untrained model, ~0.02-0.05 when converged
- DDPM sampling requires all 1000 steps; use DDIM for faster inference
