# Cyclical Annealing for VAE Training

This document explains the Cyclical Annealing schedule and its role in training Variational Autoencoders (VAEs).

---

## The Problem: KL Vanishing (Posterior Collapse)

In standard VAE training, the loss function is:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

A common failure mode is **KL vanishing** (also called **posterior collapse**), where:

- The encoder ignores the input and outputs $q(z|x) \approx p(z) = \mathcal{N}(0, I)$
- KL divergence $\to 0$, meaning the latent space carries no information
- The decoder learns to generate outputs without using the latent code
- Result: poor reconstruction and meaningless latent representations

### Why Does This Happen?

During early training:
1. The decoder is weak and cannot utilize latent information effectively
2. The KL term pushes $q(z|x)$ toward the prior immediately
3. The model finds a "shortcut": set $q(z|x) = p(z)$ and let the decoder memorize average outputs
4. Once collapsed, the model is stuck in a local minimum

---

## The Solution: Cyclical Annealing

**Cyclical Annealing** (Fu et al., 2019) addresses this by periodically varying $\beta$ during training:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta(t) \cdot D_{KL}$$

where $\beta(t)$ follows a cyclical schedule:

```
β(t)
  ^
  |    /|    /|    /|    /|
  |   / |   / |   / |   / |
  |  /  |  /  |  /  |  /  |
  | /   | /   | /   | /   |
  |/    |/    |/    |/    |
  +-------------------------> t
     Cycle 1  Cycle 2  ...
```

### Schedule Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `total_steps` | Total training steps (epochs × batches_per_epoch) | - |
| `n_cycles` | Number of annealing cycles | 4-5 |
| `max_beta` | Maximum β value at each cycle peak | 1.0-10.0 |
| `ratio` | Fraction of cycle spent in annealing phase | 0.5-0.6 |
| `mode` | Annealing curve shape | `linear` or `sigmoid` |

### How One Cycle Works

For a cycle with `ratio = 0.5`:

1. **Annealing Phase (0% - 50%)**: $\beta$ increases from 0 to `max_beta`
   - Model focuses on reconstruction (low KL penalty)
   - Encoder learns to encode useful information
   
2. **Plateau Phase (50% - 100%)**: $\beta$ stays at `max_beta`
   - Full KL regularization is applied
   - Latent space is regularized toward the prior

---

## Why Cyclical Annealing Works

### 1. Breaks the Local Minimum

Each cycle provides a "fresh start":
- When $\beta \to 0$, the model can escape posterior collapse
- The encoder re-learns to encode meaningful information
- When $\beta$ increases again, the decoder is stronger and can utilize the latent code

### 2. Progressive Learning

```
Cycle 1: Encoder learns basic features → regularized
Cycle 2: Encoder refines features → regularized again
Cycle 3: Encoder captures finer details → regularized
...
```

Each cycle builds upon the previous one, progressively learning better representations.

### 3. Balances Reconstruction vs. Regularization

| Phase | Focus | Effect |
|-------|-------|--------|
| Low β | Reconstruction | Encoder encodes rich information |
| High β | Regularization | Latent space becomes smooth and structured |

The alternation ensures neither objective dominates entirely.

---

## Implementation in This Repo

```python
# models/cyclical_annealer.py

class CyclicalAnnealer:
    def __init__(self, total_steps, n_cycles=4, max_beta=1.0, ratio=0.5, mode='linear'):
        self.period = total_steps // n_cycles
        self.step_growth = int(self.period * ratio)
        ...
    
    def __call__(self, step):
        cycle_step = step % self.period
        
        if cycle_step < self.step_growth:
            # Annealing phase: β increases
            if self.mode == 'linear':
                return self.max_beta * (cycle_step / self.step_growth)
            elif self.mode == 'sigmoid':
                x = (cycle_step / self.step_growth) * 12.0 - 6.0
                return self.max_beta / (1.0 + np.exp(-x))
        else:
            # Plateau phase: β = max_beta
            return self.max_beta
```

### Usage in BetaVAE

```python
# models/beta_vae.py

class BetaVAE(BaseVAE):
    def __init__(self, ..., total_steps, n_cycles=5, ratio=0.6, ...):
        self.annealer = CyclicalAnnealer(
            total_steps=total_steps,
            n_cycles=n_cycles,
            max_beta=beta,
            ratio=ratio,
            mode=circular_mode
        )
    
    def loss_function(self, ...):
        if self.loss_type == 'Cyclical':
            beta = self.annealer(self.num_iter)
            loss = recons_loss + beta * kld_weight * kld_loss
```

---

## Comparison with Other Annealing Strategies

| Strategy | Schedule | Pros | Cons |
|----------|----------|------|------|
| **No Annealing** | $\beta = \text{const}$ | Simple | Prone to collapse |
| **Monotonic Annealing** | $\beta: 0 \to 1$ (once) | Helps early training | Only one chance |
| **Cyclical Annealing** | $\beta: 0 \to 1$ (repeated) | Multiple recovery chances | More hyperparameters |
| **Burgess Capacity** | $\|KL - C(t)\|$ | Controls information flow | Doesn't directly prevent collapse |

---

## Expected Training Behavior

### KL Loss Curve

With proper cyclical annealing, the KL loss should show periodic oscillations:

```
KL Loss
   ^
   |  \    /\    /\    /\    /\
   |   \  /  \  /  \  /  \  /  \
   |    \/    \/    \/    \/    \_
   +--------------------------------> epoch
```

- **Rising edges**: β decreases → encoder encodes more → KL increases
- **Falling edges**: β increases → regularization kicks in → KL decreases

### Reconstruction Loss Curve

```
Recon Loss
   ^
   |    /\    /\    /\    /\    /\
   |   /  \  /  \  /  \  /  \  /  \
   |  /    \/    \/    \/    \/    \
   +---------------------------------> epoch
```

- Inversely correlated with KL (trade-off between the two objectives)

---

## Recommended Hyperparameters

For CelebA 64×64 with latent_dim=128:

```python
LOSS_TYPE = 'Cyclical'
epochs = 60          # Enough for 4-5 complete cycles
MAX_BETA = 4         # Not too high (10 is often too aggressive)
RATIO = 0.5          # Equal annealing and plateau phases
MODE = 'linear'      # Simple and effective
n_cycles = 5         # 5 cycles over training
```

### Calculating Steps

```python
TOTAL_STEPS = epochs * len(train_loader)
# e.g., 60 epochs × 1272 batches = 76,320 steps
# With n_cycles=5: each cycle ≈ 15,264 steps ≈ 12 epochs
```

---

## Troubleshooting

### Problem: KL curve is monotonically decreasing (no oscillation)

**Possible causes**:
1. `total_steps` parameter not passed correctly to model
2. Training stopped before completing one cycle
3. `n_cycles` too large relative to total training steps

**Solution**: Ensure `total_steps = epochs × batches_per_epoch` is correctly computed and passed.

### Problem: Reconstruction quality is poor

**Possible causes**:
1. `max_beta` too high
2. Not enough cycles completed
3. Decoder capacity too low

**Solution**: 
- Reduce `max_beta` (try 1-4 instead of 10)
- Add perceptual loss (LPIPS)
- Increase decoder capacity

### Problem: Latent space is not disentangled

**Note**: Cyclical annealing primarily addresses posterior collapse, not disentanglement. For better disentanglement, consider:
- Beta-TC-VAE (penalizes Total Correlation directly)
- FactorVAE
- Higher β with careful tuning

---

## References

1. Fu, H., et al. (2019). **Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing**. NAACL 2019.
   - [Paper](https://arxiv.org/abs/1903.10145)
   - [Code](https://github.com/haofuml/cyclical_annealing)

2. Bowman, S. R., et al. (2016). **Generating Sentences from a Continuous Space**. CoNLL 2016.
   - First identified KL vanishing problem in VAE for text

3. Higgins, I., et al. (2017). **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**. ICLR 2017.
   - Introduced β-VAE for disentanglement

