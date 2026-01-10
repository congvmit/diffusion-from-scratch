# Mathematical Foundations of Diffusion Models

> A comprehensive guide to understanding the mathematics behind Denoising Diffusion Probabilistic Models (DDPM)

---

## Table of Contents

1. [Introduction & Intuition](#1-introduction--intuition)
2. [The Forward Process](#2-the-forward-process-adding-noise)
3. [The Reverse Process](#3-the-reverse-process-removing-noise)
4. [Training Objective Derivation](#4-training-objective-derivation)
5. [Why Noise Prediction Works](#5-why-noise-prediction-works)
6. [Connection to Score Matching](#6-connection-to-score-matching)
7. [Sampling Algorithms](#7-sampling-algorithms)
8. [Key Insights & Summary](#8-key-insights--summary)

---

## 1. Introduction & Intuition

### The Core Idea

Diffusion models learn to **reverse a gradual noising process**. Imagine dropping ink into water:
- **Forward process**: Ink spreads out (easy, deterministic physics)
- **Reverse process**: Reconstructing the original ink drop from dispersed ink (hard, requires learning)

### Why This Works

1. **Breaking down a hard problem**: Generating images from pure noise in one step is nearly impossible. But removing a tiny bit of noise? That's learnable!

2. **Markov chain structure**: Each step only depends on the previous step, making the math tractable.

3. **Gaussian everything**: We use Gaussian distributions throughout, which have beautiful closed-form properties.

### Mathematical Setup

- $x_0$: Clean data (what we want to generate)
- $x_T$: Pure noise (our starting point for generation)
- $x_t$: Partially noisy data at timestep $t$
- $T$: Total number of timesteps (typically 1000)
- $\beta_t$: Noise schedule (how much noise to add at each step)

---

## 2. The Forward Process (Adding Noise)

### Step-by-Step Definition

The forward process $q$ adds Gaussian noise gradually:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})$$
-
**How to read this equation (step‑by‑step):**

- **Left side — conditional density:** $q(x_t\mid x_{t-1})$ denotes the probability *density* of the noisy variable $x_t$ given the previous clean(er) value $x_{t-1}$. It answers: "If I know $x_{t-1}$, how likely is a particular $x_t$?"

- **Right side — Gaussian shorthand:** $\mathcal{N}(x;\mu,\Sigma)$ is the multivariate normal with mean $\mu$ and covariance $\Sigma$. Here the mean is

   $$\mu = \sqrt{1-\beta_t}\;x_{t-1},$$

   and the covariance is

   $$\Sigma = \beta_t\,\mathbf{I}.$$

- **What $\beta_t\,\mathbf{I}$ means:** the covariance equals $\beta_t$ times the identity matrix. For an image tensor viewed as a flattened vector this means every pixel (and channel) is corrupted independently with variance $\beta_t$ (standard deviation $\sqrt{\beta_t}$). "Isotropic" means the same variance in every dimension and no cross‑dimension correlations are introduced by this single step.

- **Sampling interpretation (how to draw $x_t$):** to sample from the Gaussian, compute the mean and then add zero‑mean Gaussian noise:

   $$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon, \quad \epsilon\sim\mathcal{N}(0,\mathbf{I}).$$

- **Density value (likelihood) — optional:** the density value of a specific $x_t$ is

   $$q(x_t\mid x_{t-1}) = (2\pi\beta_t)^{-D/2} \exp\left(-\frac{\|x_t-\sqrt{1-\beta_t}\,x_{t-1}\|^2}{2\beta_t}\right),$$

   where $D$ is the total number of dimensions (e.g., $C\times H\times W$ for images). This shows smaller $\beta_t$ sharpens the Gaussian (higher peak, lower variance).

- **Intuition about small vs large $\beta_t$:** if $\beta_t\ll 1$ then $\sqrt{1-\beta_t}\approx 1$ and only a tiny perturbation is added each step; after many steps the noise accumulates. If $\beta_t$ is large, one step would already inject strong noise.

- **Scalar numeric example:** suppose a single scalar pixel has $x_{t-1}=0.5$ and $\beta_t=0.01$. Then mean $\mu=\sqrt{0.99}\cdot0.5\approx0.4975$ and the noise standard deviation is $\sqrt{\beta_t}=0.1$. So a sample might be $x_t\approx0.4975 + 0.1\cdot z$ with $z\sim\mathcal{N}(0,1)$.

**Breaking this down:**
- $\sqrt{1-\beta_t} \cdot x_{t-1}$: Slightly shrink the signal
- $\beta_t \mathbf{I}$: Add a small amount of zero-mean, *isotropic* Gaussian noise (covariance $=\beta_t\,\mathbf{I}$). This means each pixel/channel is perturbed independently with variance $\beta_t$ (standard deviation $\sqrt{\beta_t}$), so there is no cross-pixel correlation introduced by a single forward step.

   **Equivalently (reparameterization):**

   $$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon, \quad \epsilon\sim\mathcal{N}(0,\mathbf{I}).$$

   **Quick intuition:** If $\beta_t \ll 1$, each step adds a very small perturbation; repeated over many timesteps the noise accumulates and eventually overwhelms the signal.
- $\beta_t \in (0, 1)$: Typically $10^{-4}$ to $0.02$

### The Magic: Closed-Form Sampling

Instead of applying $T$ sequential steps, we can jump directly to any timestep!

**Define cumulative products:**
$$\alpha_t = 1 - \beta_t$$
$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \alpha_1 \cdot \alpha_2 \cdots \alpha_t$$

**The closed-form formula:**
$$\boxed{q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1-\bar{\alpha}_t) \mathbf{I})}$$

Or equivalently using the **reparameterization trick**:
$$\boxed{x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})}$$

### Parameterization trick: predicting noise vs. predicting $x_0$ or the mean

When we build a neural network to model the reverse step $p_\theta(x_{t-1}\mid x_t)$ we must choose what the network should predict. The three common choices are:

- Predict the Gaussian mean $\mu_\theta(x_t,t)$ directly.
- Predict the clean image $\hat{x}_0$ and compute the mean from it.
- Predict the forward-process noise $\epsilon_\theta(x_t,t)$ (the common choice).

Why predicting $\epsilon$ is convenient and effective:

1. From the closed-form reparameterization above, we can algebraically recover $\hat{x}_0$ from a noise prediction:

   $$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\;\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$$

2. The reverse-step mean used in DDPM can be written in terms of $\epsilon_\theta$ as:

   $$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\;\epsilon_\theta(x_t,t)\right).$$

   This follows by substituting $\hat{x}_0$ into the analytic posterior mean and simplifying (see Ho et al., 2020 for full algebra).

3. Practical benefits:

   - The target $\epsilon$ is zero-mean and approximately unit-variance, which is a well-scaled regression target.
   - Predicting noise is a residual-style task (often easier to learn than directly outputting images).
   - Converting between $\epsilon$, $\hat{x}_0$, and $\mu_\theta$ is deterministic and cheap at inference time.

Variants you may see in the literature:

 - `v`-prediction: predict a linear combination of $x_0$ and $\epsilon$ (sometimes gives numerical benefits).
 - `x_0`-prediction: predict the clean sample directly; can be useful for some guidance techniques but often slightly harder to train.

In short: predicting $\epsilon$ is simple, stable, and easily converted into the quantities needed for sampling.

### Proof of Closed-Form Formula

**Claim:** If $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, then $x_T \approx \mathcal{N}(0, I)$ as $\bar{\alpha}_T \to 0$.

**Proof by induction:**

*Base case* ($t=1$):
$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1 = \sqrt{\bar{\alpha}_1} x_0 + \sqrt{1-\bar{\alpha}_1} \epsilon_1 \checkmark$$

*Inductive step* (assume true for $t-1$, prove for $t$):
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

Substitute $x_{t-1}$:
$$x_t = \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{t-1}) + \sqrt{\beta_t} \epsilon_t$$

$$= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})} \epsilon_{t-1} + \sqrt{\beta_t} \epsilon_t$$

The two noise terms are independent Gaussians. Their sum has variance:
$$\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \bar{\alpha}_t$$

Therefore:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \bar{\epsilon}, \quad \bar{\epsilon} \sim \mathcal{N}(0, I) \quad \blacksquare$$

### Interpretation of $\bar{\alpha}_t$

| Property | When $t$ is small | When $t$ is large |
|----------|-------------------|-------------------|
| $\bar{\alpha}_t$ | Close to 1 | Close to 0 |
| $\sqrt{\bar{\alpha}_t}$ (signal) | Strong | Weak |
| $\sqrt{1-\bar{\alpha}_t}$ (noise) | Weak | Strong |
| $x_t$ looks like | $x_0$ (clean) | Pure noise |

---

## 3. The Reverse Process (Removing Noise)

### The Goal

We want to learn $p_\theta(x_{t-1} | x_t)$ that reverses the forward process.

**Key insight:** When $\beta_t$ is small, the reverse process is also approximately Gaussian!

### True Posterior (Intractable)

The true reverse distribution $q(x_{t-1} | x_t)$ requires knowing the entire data distribution—impossible.

But if we also condition on $x_0$, we get a **tractable posterior**:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$

where:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t$$

### Derivation of the Posterior

Using Bayes' theorem:
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

All three terms are Gaussians! The product of Gaussians is Gaussian.

**Completing the square** (key technique):

For Gaussians: $\mathcal{N}(x; \mu_1, \sigma_1^2) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2) \propto \mathcal{N}(x; \tilde{\mu}, \tilde{\sigma}^2)$

where: $\tilde{\sigma}^2 = \left(\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}\right)^{-1}$ and $\tilde{\mu} = \tilde{\sigma}^2 \left(\frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}\right)$

Applying this to our three Gaussians yields the posterior formulas above.

### Learning the Reverse Process

Our model $p_\theta$ parameterizes:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

**What to predict?** We have options:
1. Predict $\mu_\theta$ directly (the mean)
2. Predict $x_0$ and compute $\mu$ from it
3. **Predict $\epsilon$** and compute everything from it ← This works best!

---

## 4. Training Objective Derivation

### The Variational Lower Bound (ELBO)

We want to maximize $\log p_\theta(x_0)$. Using variational inference:

$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \mathcal{L}_{\text{VLB}}$$

### Decomposing the ELBO

After careful manipulation (see Appendix), the ELBO decomposes as:

$$\mathcal{L}_{\text{VLB}} = \underbrace{\mathbb{E}_q[\log p_\theta(x_0 | x_1)]}_{\mathcal{L}_0: \text{reconstruction}} - \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{\mathcal{L}_T: \text{prior matching}} - \sum_{t=2}^{T} \underbrace{\mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]}_{\mathcal{L}_{t-1}: \text{denoising matching}}$$

**Term-by-term:**
- $\mathcal{L}_0$: How well we reconstruct $x_0$ from $x_1$ (nearly clean)
- $\mathcal{L}_T$: Prior matching (no learnable parameters, ≈ 0)
- $\mathcal{L}_{t-1}$: How well our reverse step matches the true posterior

### KL Divergence Between Gaussians

For the denoising terms, we need KL between two Gaussians:

$$D_{KL}(\mathcal{N}(\mu_1, \sigma^2) \| \mathcal{N}(\mu_2, \sigma^2)) = \frac{1}{2\sigma^2} \|\mu_1 - \mu_2\|^2$$

(When variances are equal, KL reduces to squared difference of means!)

### The Simplified Loss

Substituting our expressions:

$$\mathcal{L}_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2\sigma_t^2} \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right]$$

**Key insight:** If we parameterize $\mu_\theta$ using a noise predictor $\epsilon_\theta$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

Then after substitution and simplification:

$$\mathcal{L}_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### The DDPM Loss (Final Form)

Ho et al. (2020) found that **dropping the weighting** works better in practice:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}$$

**This is just MSE between true noise and predicted noise!**

---

## 5. Why Noise Prediction Works

### Intuition 1: Denoising as Gradient

Predicting $\epsilon$ is equivalent to predicting the direction back toward clean data.

From $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, solving for $x_0$:

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$$

So knowing $\epsilon$ lets us estimate $x_0$!

### Intuition 2: Signal-to-Noise Ratio

At each timestep, $x_t$ is a mixture of signal and noise:
- **Signal component**: $\sqrt{\bar{\alpha}_t} x_0$
- **Noise component**: $\sqrt{1-\bar{\alpha}_t} \epsilon$

The model learns to **extract the noise** from this mixture.

### Intuition 3: Residual Learning

Like ResNets, predicting the "residual" (noise) is easier than predicting the target (clean image) directly.

The noise has nice properties:
- Zero mean: $\mathbb{E}[\epsilon] = 0$
- Unit variance: $\text{Var}[\epsilon] = 1$
- Independent of the data

### Why Drop the Weighting?

The original VLB weighting $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$ emphasizes different timesteps.

Empirically, **uniform weighting** (all timesteps equally important) gives:
- Better sample quality (lower FID)
- More stable training
- The model learns features at all noise levels

---

## 6. Connection to Score Matching

### Score Function Definition

The **score function** is the gradient of the log-probability:

$$\nabla_{x} \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

This points toward regions of higher probability!

### Diffusion Models Learn the Score

There's a beautiful connection:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p(x_t)$$

**Proof sketch:**

The score of $q(x_t | x_0)$ is:
$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

Therefore: $\epsilon = -\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log q(x_t | x_0)$

### Denoising Score Matching

Vincent (2011) showed that learning to denoise is equivalent to learning the score!

$$\mathbb{E}_{q(x_t | x_0)}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2] = \mathbb{E}[\|\nabla_{x_t} \log q(x_t) - s_\theta(x_t, t)\|^2] + C$$

This connects diffusion models to:
- Score-based generative models (Song & Ermon)
- Energy-based models
- Langevin dynamics

---

## 7. Sampling Algorithms

### DDPM Sampling (Original)

Starting from $x_T \sim \mathcal{N}(0, I)$, iterate for $t = T, T-1, \ldots, 1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ and $\sigma_t = \sqrt{\beta_t}$ or $\sigma_t = \sqrt{\tilde{\beta}_t}$.

**Complexity:** Requires all $T$ steps (slow!)

### DDIM Sampling (Fast)

Song et al. (2020) discovered a **deterministic** sampling process:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**Key advantages:**
- Can skip timesteps! (e.g., use only 50 steps instead of 1000)
- Deterministic (same noise → same image)
- Uses the same trained model as DDPM

### Comparison

| Method | Steps | Stochastic | Quality |
|--------|-------|------------|---------|
| DDPM | 1000 | Yes | Best |
| DDIM (50 steps) | 50 | No | Very good |
| DDIM (10 steps) | 10 | No | Acceptable |

---

## 8. Key Insights & Summary

### The Essential Equations

1. **Forward process:**
   $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

2. **Training loss:**
   $$L = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

3. **Reverse step:**
   $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

### Why Diffusion Models Work So Well

1. **Stable training**: No adversarial dynamics (unlike GANs)
2. **Mode coverage**: The probabilistic framework covers the full distribution
3. **Progressive refinement**: Coarse-to-fine generation at different noise levels
4. **Score matching**: Theoretical foundation from score-based models
5. **Flexibility**: Easy to add conditioning (text, class, etc.)

### Common Confusions Clarified

| Confusion | Clarification |
|-----------|---------------|
| "We predict $x_0$" | No, we predict $\epsilon$. We can *compute* $x_0$ from $\epsilon$ |
| "Training is slow" | Training is normal. *Sampling* is slow (many steps) |
| "It's autoregressive" | No, each step processes the full image, not pixel-by-pixel |
| "We need $T=1000$" | For training yes, but DDIM can sample with fewer steps |

### Historical Context

| Year | Paper | Contribution |
|------|-------|--------------|
| 2015 | Sohl-Dickstein et al. | Original diffusion concept |
| 2019 | Song & Ermon | Score matching connection |
| 2020 | Ho et al. (DDPM) | Simplified training, great results |
| 2020 | Song et al. (DDIM) | Fast deterministic sampling |
| 2021 | Dhariwal & Nichol | Diffusion beats GANs |
| 2022 | Rombach et al. | Latent diffusion (Stable Diffusion) |

---

## Appendix: Detailed ELBO Derivation

### Starting Point

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}$$

### Introducing the Variational Distribution

$$= \log \int \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} q(x_{1:T}|x_0) dx_{1:T}$$

$$= \log \mathbb{E}_{q(x_{1:T}|x_0)}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

### Applying Jensen's Inequality

$$\geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \mathcal{L}_{\text{VLB}}$$

### Expanding the Terms

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

### Final Decomposition

After algebraic manipulation:

$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q[\log p_\theta(x_0|x_1)] - D_{KL}(q(x_T|x_0) \| p(x_T)) - \sum_{t=2}^{T} \mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]$$

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
2. Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR.
3. Sohl-Dickstein, J., et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML.
4. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.
5. Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation.

---

*This document accompanies the interactive notebook `01_diffusion_mentor.ipynb`. For hands-on learning with code, see the notebook.*
