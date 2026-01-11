# Variation Autoencoders

This repo contains several VAE variants (mainly for CelebA face images) and a set of experiment folders under `results_*`.
The model implementations in `models/` are based on AntixK/PyTorch-VAE(https://github.com/AntixK/PyTorch-VAE.git), with additional loss options (Burgess capacity, PID, cyclical beta) and optional perceptual losses.

Them main **training pipeline** is `vae_on_celebs_train.ipynb`, and the **evaluation pipeline** is `vae_on_celebs_eval.ipynb`.

Relative docs could be find in `docs`, including loss function setup, and Beta-TC-VAE learning notes.

---

## Architecture (Model)

### Core: `BetaVAE` (default workhorse)

Implemented in `models/beta_vae.py`.

**Encoder (Conv → latent distribution)**

- Input: RGB images, shape `[B, 3, H, W]`.
- A stack of strided convolutions (stride=2) with BatchNorm + LeakyReLU.
	- Default `hidden_dims = [32, 64, 128, 256, 512]`.
	- Each block halves spatial resolution.
- The final feature map is flattened and projected into:
	- `mu = fc_mu(flat)`
	- `log_var = fc_var(flat)`
- Latent sampling uses the reparameterization trick:
	- $z = \mu + \sigma \odot \epsilon$, where $\sigma = \exp(0.5\,\text{log\_var})$ and $\epsilon \sim \mathcal{N}(0, I)$.

**Decoder (latent → image)**

- A linear layer maps `z` back to the flattened encoder feature size.
- A stack of ConvTranspose2d blocks upsamples back to the original image size.
- Final output uses `Tanh()` → output range is **[-1, 1]**.

**Important implication**

- Because the model outputs `tanh`, the cleanest training setup is to normalize input images to **[-1, 1]** as well.
- LPIPS also assumes inputs are in **[-1, 1]**.

### `BetaTCVAE` (Total Correlation variant)

Implemented in `models/beta_tc_vae.py`.

- Similar conv encoder/decoder layout.
- Loss decomposes KL into components (MI / TC / DW-KL) controlled by `(alpha, beta, gamma)`.
- Can also enable LPIPS + TV loss in the reconstruction term.

---

## Latent dimension choice (Why `latent_dim` = 128 / 256)

This project trains on CelebA faces at 64×64 and 128×128. The latent dimension is chosen as a balance between:

- **Capacity vs. detail**: higher `latent_dim` can store more information for sharper reconstructions.
- **Disentanglement pressure**: stronger KL regularization (or capacity schedules) is usually required as latent capacity grows.
- **Compute/memory**: latent size affects the FC layers and the amount of information passed through the bottleneck.

### Practical defaults used in this repo

- For **64×64** experiments, a good default is `latent_dim = 128`.
- For **128×128** experiments, a good default is `latent_dim = 256`.

You can see these choices reflected in the experiment folders, e.g.:

- `results_beta_vae_ImSize64_Lat128_...`
- `results_beta_vae_ImSize128_Lat256_...`

### When to change it

- If reconstructions look overly smooth *even with a weak KL term*, try increasing `latent_dim`.
- If reconstructions are sharp but latents are not structured/disentangled, try:
	- stronger KL regularization (e.g. larger `beta` for `loss_type='H'`), or
	- a capacity schedule (`loss_type='B'`), or
	- PID / cyclical beta.

---

## Training setup (Data, loss, hyperparameters)

### Quick Reference Tables (Slide-Ready)

#### Table 1: Training Configuration

| Component | Setting | Notes |
|-----------|---------|-------|
| **Dataset** | CelebA (~160k train) | `scripts/download_celeba.py` |
| **Preprocessing** | CenterCrop(178) → Resize → ToTensor → Normalize | |
| **Normalization** | `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)` | Maps [0,1] → [-1,1] for `tanh` output |
| **Optimizer** | Adam | |
| **Learning rate** | 1e-3 (main) / 1e-4 (lightweight) | |
| **LR Scheduler** | ReduceLROnPlateau | patience=5, mode='min' |
| **Batch size** | 256 (64×64) / 64–128 (128×128) | Adjust for GPU memory |
| **Early stopping** | patience=10 on val recon loss | Starts after epoch 10 |

#### Table 2: Model Architecture

| Image Size | Latent Dim | Hidden Dims | Encoder Output | Total Params |
|------------|------------|-------------|----------------|--------------|
| 64×64 | 128 | [32, 64, 128, 256, 512] | 512×4×4 | ~8M |
| 64×64 | 256 | [32, 64, 128, 256, 512] | 512×4×4 | ~10M |
| 128×128 | 256 | [32, 64, 128, 256, 512, 1024] | 1024×4×4 | ~42M |

#### Table 3: Loss Function Variants (BetaVAE)

| Loss Type | Formula | Key Params | Typical Values |
|-----------|---------|------------|----------------|
| **H** (Higgins) | $\mathcal{L}_{recon} + \beta \cdot \text{KLD}$ | `beta` | 0.01, 1, 5, 10 |
| **B** (Burgess) | $\mathcal{L}_{recon} + \gamma \cdot \|\text{KLD} - C(t)\|$ | `gamma`, `max_capacity` | γ=10–30, C=100–400 |
| **PID** | Adaptive β via PID controller | `exp_kld_loss` (target) | 10, 25 |
| **Cyclical** | $\beta(t)$ periodic annealing | `max_beta`, `ratio`, `mode` | max=10, ratio=0.6, linear |

#### Table 4: Beta-TC-VAE Parameters

| Param | Role | Default | Typical Range |
|-------|------|---------|---------------|
| `alpha` | MI (Mutual Info) weight | 1 | 1 (usually fixed) |
| `beta` | **TC (Total Correlation)** weight | 6 | 1–10 (↑ = more disentangled) |
| `gamma` | DW-KL weight | 1 | 1 (usually fixed) |
| `anneal_steps` | DW-KL warmup iterations | 200 | 100–500 |

#### Table 5: Perceptual Loss Settings

| Component | Weight Param | Typical Value | Effect |
|-----------|--------------|---------------|--------|
| MSE (L2) | — | always on | Base reconstruction |
| LPIPS (VGG) | `lpips_weight` | 0.5–1.5 | Perceptual sharpness |
| TV Loss | `tvl_weight` | 1e-4–1e-2 | Smoothness / anti-checkerboard |

#### Table 6: Selected Experiment Configurations

| Experiment | ImgSize | LatentDim | Loss | β/γ | Capacity | LPIPS | TV |
|------------|---------|-----------|------|-----|----------|-------|-----|
| Baseline H | 64 | 128 | H | β=1 | — | ✗ | ✗ |
| Strong KL (H) | 64 | 256 | H | β=10 | — | ✗ | ✗ |
| Burgess | 64 | 256 | B | γ=20 | 400 | ✗ | ✗ |
| Burgess+Perc | 128 | 256 | B | γ=15 | 400 | 1.2 | 0.01 |
| Cyclical | 64 | 128 | Cyc | max=10 | — | ✗ | ✗ |
| Beta-TC-VAE | 64 | 128 | TC | β=6 | — | ✗ | ✗ |
| Beta-TC-VAE (strong) | 64 | 128 | TC | β=10 | — | ✗ | ✗ |

---

### Dataset

CelebA is downloaded via torchvision:

```bash
python scripts/download_celeba.py
```

By default this places data under `./data/`.

### Preprocessing / normalization

Minimum required preprocessing:

- `Resize((image_size, image_size))`
- `ToTensor()`

Recommended for **`tanh` output + LPIPS** (always use this in practice):

```python
transforms.Compose([
    transforms.CenterCrop((178, 178)),    # Remove borders
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
])
```

### Losses

The model returns `results = [recons, input, mu, log_var]`.
The loss is computed in `models/beta_vae.py::loss_function` with:

- **Reconstruction term** (base):
	- `MSE(recons, input)`
- Optional additions when `enable_perceptual_loss=True`:
	- **TV loss**: penalizes high-frequency noise
	- **LPIPS** (VGG backbone): perceptual similarity loss
		- LPIPS parameters are frozen (`requires_grad=False`) to avoid optimization "cheating".

KL term (per batch):

$$\text{KLD} = \mathbb{E}\left[ -\tfrac{1}{2}\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2) \right]$$

`M_N` is the minibatch scaling factor: `M_N = batch_size / len(dataset)`.

### KL weighting / schedules (`loss_type`)

`loss_type` controls how KL affects the objective:

- `H`: Higgins β-VAE
	- $\mathcal{L} = \mathcal{L}_{recon} + \beta\, M_N\, \text{KLD}$
- `B`: Burgess capacity
	- $\mathcal{L} = \mathcal{L}_{recon} + \gamma\, M_N\, |\text{KLD} - C(t)|$
	- Capacity ramps up until `max_capacity`.
- `PID`: PID controller adjusts an effective β to match a target KL.
- `Cyclical`: cyclical annealing schedule for β.

---

## Results folder naming convention

To keep experiments self-describing, this repo uses folder names like:

```
results_<model>_ImSize<IMG>_Lat<LAT>_<LOSS>_...
```

Examples:

- `results_beta_tc_vae_ImSize64_Lat128_Alpha1_Beta6_Gamma1/`
- `results_beta_vae_ImSize64_Lat256_B_G20_C400/`
- `results_beta_vae_ImSize64_Lat128_TVL10_LPIPS12_H_Beta5/`

You can parse and export these into a structured JSON via:

```bash
python check_valid_training.py
```

This script scans `results_*` directories and writes a `hyperparameters.json` file under each folder it can parse.

---

## Repo layout

- `models/`: model implementations (BetaVAE, BetaTCVAE, etc.)
- `data/`: datasets (CelebA, MNIST)
- `results_*/`: saved experiment outputs
- Notebooks: exploratory / exercise notebooks (`vae_on_celebs.ipynb`, `variation_autoencoder_excersize.ipynb`, ...)

