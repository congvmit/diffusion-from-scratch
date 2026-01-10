# Diffusion Models from Scratch 🎨

A hands-on learning journey to understand and implement diffusion models step-by-step.

## 📚 Learning Path

| Module | Topic | Notebook |
|--------|-------|----------|
| 1 | **Foundations** - Math intuition, Gaussian noise, forward process | `01_foundations.ipynb` |
| 2 | **DDPM** - Denoising Diffusion Probabilistic Models | `02_ddpm.ipynb` |
| 3 | **UNet Architecture** - Building the denoising network | `03_unet.ipynb` |
| 4 | **Training** - Loss functions, sampling, training loop | `04_training.ipynb` |
| 5 | **Sampling Methods** - DDPM, DDIM, accelerated sampling | `05_sampling.ipynb` |
| 6 | **Conditioning** - Class-conditional and text-conditional generation | `06_conditioning.ipynb` |
| 7 | **Advanced Topics** - Latent diffusion, guidance, fine-tuning | `07_advanced.ipynb` |

## 🚀 Quick Start

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install all dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install the package with notebook dependencies
uv pip install -e ".[notebook]"

# Or install everything (including dev tools)
uv pip install -e ".[all]"

# Start learning
uv run jupyter notebook
```

## 🎯 Learning Goals

By the end of this curriculum, you will:
- Understand the mathematical foundations of diffusion models
- Implement DDPM from scratch in PyTorch
- Build and train a UNet for image denoising
- Generate images using various sampling methods
- Add conditioning for controlled generation

## 📁 Project Structure

```
diffusion-from-scratch/
├── notebooks/           # Interactive learning notebooks
├── src/
│   ├── models/         # Model implementations
│   ├── utils/          # Helper functions
│   └── data/           # Data loading utilities
├── configs/            # Training configurations
└── experiments/        # Saved models and outputs
```

## 🔧 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
