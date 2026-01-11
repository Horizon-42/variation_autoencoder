# Beta-TC-VAE: Isolating Sources of Disentanglement

This document explains the Beta-TC-VAE model, its loss decomposition, and addresses the common issue of negative MI/TC loss values.

## 1. Overview

Beta-TC-VAE (Chen et al., 2018) improves upon β-VAE by decomposing the KL divergence into three interpretable terms, allowing independent control over different aspects of the latent representation.

## 2. KL Divergence Decomposition

### 2.1 Standard VAE KL Term

In a standard VAE, the KL divergence is:

$$
D_{KL}(q(z|x) \| p(z))
$$

Beta-TC-VAE decomposes this into **three terms**:

$$
\mathbb{E}_{p(x)}[D_{KL}(q(z|x) \| p(z))] = \underbrace{I_q(x; z)}_{\text{Index-Code MI}} + \underbrace{D_{KL}(q(z) \| \prod_j q(z_j))}_{\text{Total Correlation}} + \underbrace{\sum_j D_{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}
$$

### 2.2 Three Components Explained

| Term | Name | Meaning | Effect of Penalizing |
|------|------|---------|---------------------|
| \( I_q(x; z) \) | **Index-Code MI** | Mutual information between data and latent code | Reduces information in z about x (hurts reconstruction) |
| \( TC(z) \) | **Total Correlation** | Measures statistical dependence among latent dimensions | Encourages **disentanglement** |
| \( \sum_j D_{KL}(q(z_j) \| p(z_j)) \) | **Dimension-wise KL** | How much each marginal deviates from prior | Prevents individual dimensions from deviating too far |

### 2.3 Beta-TC-VAE Loss Function

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \alpha \cdot I_q(x;z) - \beta \cdot TC(z) - \gamma \cdot \sum_j D_{KL}(q(z_j) \| p(z_j))
$$

**Key insight**: By setting \( \beta > 1 \) while keeping \( \alpha = \gamma = 1 \), we specifically penalize the Total Correlation without unnecessarily hurting reconstruction quality.

## 3. Estimating the Decomposition

### 3.1 The Challenge

The decomposition requires computing:
- \( q(z) = \mathbb{E}_{p(x)}[q(z|x)] \) — the aggregated posterior (intractable)
- \( \prod_j q(z_j) \) — product of marginals

### 3.2 Minibatch Weighted Sampling (MWS) Estimator

The paper proposes using importance-weighted sampling with a minibatch:

$$
q(z) \approx \frac{1}{NM} \sum_{i=1}^{N} \sum_{j=1}^{M} q(z|x_j)
$$

Where \( N \) is dataset size and \( M \) is batch size.

#### Implementation in Code

```python
# log q(z|x) - posterior density at sampled z
log_q_zx = log_density_gaussian(z, mu, log_var).sum(dim=1)

# log p(z) - prior density
log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim=1)

# Estimate log q(z) using minibatch weighted sampling
mat_log_q_z = log_density_gaussian(
    z.view(batch_size, 1, latent_dim),
    mu.view(1, batch_size, latent_dim),
    log_var.view(1, batch_size, latent_dim)
)

# Apply stratified importance weights
log_q_z = logsumexp(mat_log_q_z.sum(2) + log_importance_weights, dim=1)
log_prod_q_z = logsumexp(mat_log_q_z + log_importance_weights, dim=1).sum(1)

# Final decomposition
mi_loss = (log_q_zx - log_q_z).mean()           # I(x;z)
tc_loss = (log_q_z - log_prod_q_z).mean()       # TC(z)
kld_loss = (log_prod_q_z - log_p_z).mean()      # Σ KL(q(z_j)||p(z_j))
```

## 4. Why MI and TC Loss Can Be Negative

### 4.1 Theoretical Bounds

In theory:
- \( I_q(x; z) \geq 0 \) (Mutual Information is non-negative)
- \( TC(z) \geq 0 \) (KL divergence is non-negative)
- \( D_{KL}(q(z_j) \| p(z_j)) \geq 0 \)

**However**, the MWS estimator produces **biased estimates** that can violate these bounds.

### 4.2 Mathematical Analysis of Estimator Bias

#### The Log-Sum-Exp Approximation

The true marginal is:

$$
q(z) = \frac{1}{N} \sum_{i=1}^{N} q(z|x_i)
$$

But we estimate it using only the minibatch:

$$
\hat{q}(z) = \frac{1}{M} \sum_{j=1}^{M} q(z|x_j) \cdot w_j
$$

Where \( w_j \) are importance weights attempting to correct for the sampling bias.

#### Why Bias Occurs

1. **Finite Sample Effect**: With \( M \ll N \), we're using a tiny fraction of the data
   
2. **Log-Space Computation**: We compute in log-space for numerical stability:
   $$
   \log \hat{q}(z) = \text{logsumexp}(\log q(z|x_j) + \log w_j) - \log M
   $$
   
   The logsumexp is dominated by the largest terms, introducing bias.

3. **High-Dimensional Latent Space**: For \( d \)-dimensional \( z \):
   $$
   \log q(z|x) = \sum_{k=1}^{d} \log q(z_k|x)
   $$
   
   Each term is negative (log of probability < 1), so the sum becomes **very negative** as \( d \) increases.

### 4.3 Detailed Example: Why TC Can Be Negative

Consider the TC estimate:

$$
\widehat{TC} = \log \hat{q}(z) - \sum_j \log \hat{q}(z_j)
$$

**Scenario causing negative TC:**

Let's say we have latent dim \( d = 128 \).

- \( \log \hat{q}(z) \approx -700 \) (joint density in 128D space is tiny)
- \( \sum_j \log \hat{q}(z_j) \approx -650 \) (sum of 128 individual estimates)

Then: \( \widehat{TC} = -700 - (-650) = -50 \)

This happens because:
1. The joint estimate \( \hat{q}(z) \) is computed by summing across all dimensions **first**, then applying logsumexp
2. The product estimate \( \prod_j \hat{q}(z_j) \) applies logsumexp to **each dimension separately**
3. These two approaches have different biases!

### 4.4 Formal Bias Analysis

The MWS estimator has the following properties:

$$
\mathbb{E}[\widehat{MI}] = I_q(x;z) + \mathcal{O}\left(\frac{1}{M}\right) + \text{bias from importance weights}
$$

$$
\mathbb{E}[\widehat{TC}] = TC(z) + \mathcal{O}\left(\frac{d}{M}\right) + \text{higher-order terms}
$$

Key observations:
- **Bias increases with latent dimension \( d \)**
- **Bias decreases with batch size \( M \)**
- The bias terms can be **positive or negative**

### 4.5 Impact of Latent Dimension

| Latent Dim | Typical MI Value | Typical TC Value | Notes |
|------------|------------------|------------------|-------|
| 10 | 50-100 | 10-50 | Estimates usually correct sign |
| 32 | 100-300 | -50 to 50 | TC may fluctuate around 0 |
| 64 | 200-500 | -200 to -50 | TC often negative |
| 128 | 500-800 | -700 to -600 | Severe bias, TC consistently negative |

## 5. Does Negative TC/MI Hurt Training?

### 5.1 Short Answer: Usually No

The **relative ordering** of losses matters more than absolute values. Even with biased estimates:
- Reducing TC still encourages disentanglement
- The gradient direction is approximately correct

### 5.2 What to Monitor Instead

1. **Reconstruction Quality**: Visual inspection of reconstructions
2. **KLD per Dimension**: Should see some active dims (high KLD) and some collapsed (low KLD)
3. **Disentanglement Metrics**: FVM, DCI, MIG scores on labeled data
4. **Latent Traversals**: Do individual dimensions control single factors?

### 5.3 When It Becomes a Problem

Negative TC can hurt when:
- The bias is so large it **reverses gradients** (rare)
- Training becomes unstable with loss oscillation
- All latent dimensions collapse (extreme posterior collapse)

## 6. Practical Recommendations

### 6.1 Hyperparameter Guidelines

| Parameter | Recommended | Your Value | Assessment |
|-----------|-------------|------------|------------|
| \( \alpha \) | 1 | 1 | ✅ |
| \( \beta \) | 1-10 | 5 | ✅ |
| \( \gamma \) | 1 | 1 | ✅ |
| latent_dim | 10-64 | 128 | ⚠️ Too high |
| batch_size | 256-512 | 256 | ✅ |
| anneal_steps | 100-1000 | 200 | ✅ |

### 6.2 Reducing Estimator Bias

1. **Lower latent dimension**: 32-64 is usually sufficient for CelebA
2. **Increase batch size**: Larger batches reduce variance
3. **Use alternative estimators**: 
   - TCVAE with different importance weighting
   - FactorVAE (uses discriminator instead)

### 6.3 Alternative: FactorVAE

If MWS estimator bias is problematic, consider FactorVAE which uses a discriminator to estimate TC:

$$
TC(z) \approx \mathbb{E}_{q(z)}[\log D(z)] + \mathbb{E}_{\bar{q}(z)}[\log(1 - D(z))]
$$

Where \( \bar{q}(z) = \prod_j q(z_j) \) is approximated by permuting dimensions across the batch.

## 7. Summary

| Issue | Cause | Solution |
|-------|-------|----------|
| Negative TC | High-dim latent space + MWS estimator bias | Reduce latent_dim to 32-64 |
| Very large MI | Same as above | Reduce latent_dim |
| Training instability | Biased gradients | Increase batch_size, reduce β |
| Poor disentanglement | TC penalty ineffective | Try FactorVAE or lower β initially |

## 8. References

1. Chen, R.T.Q., et al. (2018). "Isolating Sources of Disentanglement in Variational Autoencoders". NeurIPS.
2. Kim, H., & Mnih, A. (2018). "Disentangling by Factorising". ICML.
3. Locatello, F., et al. (2019). "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations". ICML.

## Appendix: Log-Density Gaussian

The log probability density of a Gaussian:

$$
\log \mathcal{N}(x | \mu, \sigma^2) = -\frac{1}{2}\left(\log(2\pi) + \log(\sigma^2) + \frac{(x - \mu)^2}{\sigma^2}\right)
$$

In code:
```python
def log_density_gaussian(x, mu, logvar):
    norm = -0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density
```

For a 128-dimensional latent with standard normal prior (\( \mu=0, \sigma=1 \)):
$$
\log p(z) = \sum_{k=1}^{128} \log \mathcal{N}(z_k | 0, 1) \approx 128 \times (-0.9189 - 0.5 \cdot z_k^2) \approx -120 \text{ to } -200
$$

This explains why the raw log-probability values are so negative in high dimensions.

