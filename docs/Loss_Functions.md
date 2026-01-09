# Loss Functions (VAE / Beta-VAE / Beta-TC-VAE)

This note documents the loss functions used in this repo, with the goal of making the math match the code.

**Code references**

- `models/beta_vae.py` (BetaVAE + optional LPIPS/TV)
- `models/beta_tc_vae.py` (BetaTCVAE: MI / TC / DW-KL decomposition)

---

## Notation

- Input image: $x \in \mathbb{R}^{3\times H\times W}$
- Reconstruction: $\hat{x} = f_\theta(z)$
- Encoder outputs Gaussian parameters: $q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x),\operatorname{diag}(\sigma^2_\phi(x)))$
- Prior: $p(z)=\mathcal{N}(0, I)$
- Reparameterization: $z=\mu + \sigma\odot\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$

---

## Reconstruction losses

### 1) MSE (Pixel-wise L2)

In code (BetaVAE / BetaTCVAE):

- BetaVAE: `mse_loss = F.mse_loss(recons, input)`
- BetaTCVAE: `recons_loss = F.mse_loss(recons, input, reduction='sum')`

Mathematically (per sample):

$$
\mathcal{L}_{\text{MSE}}(x,\hat{x}) = \frac{1}{N}\sum_{i=1}^{N} (x_i-\hat{x}_i)^2
$$

where $N = 3HW$ for RGB images.

#### Why MSE corresponds to a Gaussian likelihood

If we assume a conditional likelihood

$$
p_\theta(x\mid z) = \mathcal{N}(\hat{x}(z), \sigma_x^2 I),
$$

then the negative log-likelihood is

$$
-\log p_\theta(x\mid z)
= \frac{1}{2\sigma_x^2}\lVert x-\hat{x}(z)\rVert_2^2 + \frac{N}{2}\log(2\pi\sigma_x^2).
$$

For fixed $\sigma_x^2$, minimizing $-\log p_\theta(x\mid z)$ is equivalent (up to a constant scale and offset) to minimizing $\lVert x-\hat{x}(z)\rVert_2^2$, i.e. MSE/L2 reconstruction error.

**Practical implication:** the Gaussian assumption penalizes squared pixel differences; it often favors “averaging” uncertain high-frequency details, which can look blurry.

### 2) Total Variation loss (TV)

This repo optionally adds TV loss on reconstructions (see `tv_loss` in both BetaVAE and BetaTCVAE).

A common form is:

$$
\mathcal{L}_{\text{TV}}(\hat{x}) = \sum_{c,h,w} (\hat{x}_{c,h,w}-\hat{x}_{c,h,w+1})^2 + (\hat{x}_{c,h,w}-\hat{x}_{c,h+1,w})^2.
$$

TV loss discourages high-frequency noise (e.g. checkerboard / ripple artifacts), trading off some sharpness for smoothness.

### 3) LPIPS perceptual loss

When enabled, the repo computes LPIPS between $\hat{x}$ and $x$ and adds it to the reconstruction term:

$$
\mathcal{L}_{\text{recon}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{TV}}\,\mathcal{L}_{\text{TV}} + \lambda_{\text{LPIPS}}\,\mathcal{L}_{\text{LPIPS}}.
$$

Notes that match this codebase:

- LPIPS backbone is frozen (`requires_grad=False`) to prevent the optimizer from “cheating” by changing LPIPS weights.
- LPIPS expects inputs in the range **[-1, 1]**. Since decoders here end with `tanh`, it’s usually best to normalize images to [-1, 1] as well.

(There are extra LPIPS troubleshooting notes in `docs/LPIPS_Tips.md`.)

---

## KL divergence term (Gaussian posterior vs standard normal prior)

For diagonal Gaussians $q(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))$ and $p(z)=\mathcal{N}(0,I)$:

$$
D_{KL}(q(z\mid x)\,\|\,p(z))
= \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right).
$$

In code, this appears as:

$$
\text{KLD} = -\frac{1}{2}\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)
$$

with `log_var = log(sigma^2)`.

`M_N` in the code is a scaling factor intended to account for minibatch sampling (often $M_N \approx \frac{\text{batch}}{\text{dataset}}$ or $1/\text{dataset}$ depending on the exact convention).

---

## BetaVAE family losses (as implemented in `models/beta_vae.py`)

Let

- reconstruction term: $\mathcal{L}_{\text{recon}}$
- KL term: $\text{KLD}$
- minibatch scaling: $M_N$

### 1) Higgins β-VAE (`loss_type='H'`)

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta\,M_N\,\text{KLD}.
$$

Increasing $\beta$ pushes the posterior closer to the prior (more regularization), often improving factorization/disentanglement but hurting reconstruction.

### 2) Burgess capacity (`loss_type='B'`)

This uses a target capacity $C(t)$ that increases with training steps until `max_capacity`:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \gamma\,M_N\,\left|\text{KLD} - C(t)\right|.
$$

Intuition:

- Early: $C(t)$ is small → model is forced to compress strongly.
- Later: capacity increases → model is allowed to encode more information gradually.

### 3) PID controller (`loss_type='PID'`)

The code uses a PID controller to adjust an *effective* KL weight (an adaptive “β”) to track a target KL value.

Conceptually:

- Given a target $\text{KLD}^*$, compute error $e_t = \text{KLD}^* - \text{KLD}_t$.
- Use PID on $e_t$ to produce a control value $\beta_t$.
- Optimize:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta_t\,\text{KLD}.
$$

### 4) Cyclical annealing (`loss_type='Cyclical'`)

The code schedules $\beta$ over training steps using a cyclical annealer.

One typical template is:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta(t)\,M_N\,\text{KLD},
$$

where $\beta(t)$ periodically ramps from 0→max to encourage alternating phases of reconstruction and regularization.

---

## Beta-TC-VAE (as implemented in `models/beta_tc_vae.py`)

Beta-TC-VAE decomposes the KL part of the ELBO into three components:

- Mutual information (MI): encourages information between $x$ and $z$ to be controlled
- Total correlation (TC): encourages factorization of the aggregated posterior
- Dimension-wise KL (DW-KL): matches each marginal $q(z_j)$ to the prior

### 1) Start from the standard VAE objective

(Up to constants) the negative ELBO can be written as:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z\mid x)}\left[ -\log p(x\mid z) \right] + D_{KL}(q(z\mid x)\|p(z)).
$$

Define the *aggregated posterior*:

$$
q(z) = \int q(z\mid x)\,q(x)\,dx.
$$

Then the KL can be decomposed (Chen et al., 2018) as:

$$
\mathbb{E}_{q(x)}\left[D_{KL}(q(z\mid x)\|p(z))\right]
= I_q(x;z) + D_{KL}(q(z)\|\prod_j q(z_j)) + \sum_j D_{KL}(q(z_j)\|p(z_j)).
$$

Where:

- $I_q(x;z)$ is mutual information under $q$
- $D_{KL}(q(z)\|\prod_j q(z_j))$ is **total correlation**
- $\sum_j D_{KL}(q(z_j)\|p(z_j))$ is **dimension-wise KL**

### 2) Beta-TC-VAE objective used here

The implementation follows the standard weighting scheme:

$$
\mathcal{L}_{\text{BetaTC}} = \mathcal{L}_{\text{recon}} + \alpha\,\text{MI} + \beta\,\text{TC} + \gamma\,\text{DW-KL}.
$$

In this repo:

- `alpha`, `beta`, `gamma` are constructor parameters.
- `anneal_rate` ramps the DW-KL term for the first `anneal_steps` iterations:

$$
\mathcal{L} = \frac{\mathcal{L}_{\text{recon}}}{B} + \alpha\,\text{MI} + \beta\,\text{TC} + \underbrace{\text{anneal\_rate}}_{\in[0,1]}\,\gamma\,\text{DW-KL}.
$$

(The code divides `recons_loss` by batch size `B`.)

### 3) How MI / TC / DW-KL are computed (high level)

The code estimates:

- $\log q(z\mid x)$ using the diagonal Gaussian density
- $\log q(z)$ and $\log \prod_j q(z_j)$ via minibatch importance-weighted estimates (using `logsumexp` over a matrix of pairwise densities)

Then:

$$
\text{MI} = \mathbb{E}[\log q(z\mid x) - \log q(z)],
$$

$$
\text{TC} = \mathbb{E}[\log q(z) - \log \prod_j q(z_j)],
$$

$$
\text{DW-KL} = \mathbb{E}[\log \prod_j q(z_j) - \log p(z)].
$$

This is why `models/beta_tc_vae.py` constructs `mat_log_q_z` with shape `[B, B, D]`, adds log importance weights, then uses `logsumexp` to get $\log q(z)$ and $\log \prod_j q(z_j)$.

---

## Practical tips (matching this repo)

- If you enable LPIPS: freeze LPIPS params (already done in code) and keep inputs in [-1, 1].
- If outputs look “blurry” with MSE-only: either increase latent capacity (larger `latent_dim` or weaker KL), or add perceptual loss; but expect a trade-off with disentanglement.
- If you see high-frequency ripples: increase TV weight or check ConvTranspose artifacts.
