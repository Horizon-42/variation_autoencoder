# Latent Traversal in Variational Autoencoders

This document explains **latent traversal** (also known as latent space interpolation or disentanglement visualization), a technique used to understand and visualize what a VAE has learned in its latent space.

## 1. What is Latent Traversal?

Latent traversal is a technique where we:
1. Encode an image into the latent space to get a latent vector \( z \)
2. Systematically vary **one dimension** of \( z \) while keeping all others fixed
3. Decode the modified \( z \) back to image space
4. Observe how the generated image changes

This reveals what each latent dimension has learned to represent (e.g., rotation, color, expression, etc.).

## 2. Mathematical Foundation

### 2.1 VAE Latent Space Recap

In a VAE, an input image \( x \) is encoded into a latent distribution:

$$
q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))
$$

Where:
- \( \mu(x) \): Mean vector predicted by the encoder
- \( \sigma^2(x) \): Variance vector predicted by the encoder
- \( z \in \mathbb{R}^d \): Latent vector with \( d \) dimensions

### 2.2 Traversal Process

Given a base image \( x_0 \), we first encode it to get the mean vector:

$$
\mu_0 = \text{Encoder}(x_0)
$$

We use \( \mu_0 \) (not a sampled \( z \)) to eliminate randomness and ensure reproducibility.

For a specific dimension \( k \) (where \( k \in \{1, 2, ..., d\} \)), we create a modified latent vector:

$$
z^{(k, v)} = \mu_0 + v \cdot e_k
$$

Where:
- \( e_k \) is the one-hot unit vector with 1 at position \( k \) and 0 elsewhere
- \( v \in [v_{\min}, v_{\max}] \) is the traversal value (typically \( [-3, 3] \))

More explicitly, the modified latent vector is:

$$
z^{(k, v)}_i = 
\begin{cases} 
v & \text{if } i = k \\
(\mu_0)_i & \text{otherwise}
\end{cases}
$$

The traversed image is then:

$$
\hat{x}^{(k, v)} = \text{Decoder}(z^{(k, v)})
$$

### 2.3 Traversal Grid

For visualization, we typically:
- Select \( K \) dimensions to traverse (usually the most "active" ones)
- For each dimension, generate \( N \) images across the traversal range

This produces a grid of \( K \times N \) images:

$$
\text{Grid} = \left\{ \hat{x}^{(k, v_j)} \mid k \in \{1, ..., K\}, \; v_j \in \text{linspace}(v_{\min}, v_{\max}, N) \right\}
$$

## 3. Identifying Active Dimensions

Not all latent dimensions learn meaningful features. In a well-regularized VAE, many dimensions may collapse to the prior \( \mathcal{N}(0, 1) \) — these are "dead" dimensions.

### 3.1 KL Divergence per Dimension

To find **active dimensions**, we compute the KL divergence for each dimension separately:

$$
\text{KLD}_k = -\frac{1}{2} \mathbb{E}_{x \sim \mathcal{D}} \left[ 1 + \log(\sigma_k^2(x)) - \mu_k^2(x) - \sigma_k^2(x) \right]
$$

Where:
- \( \mu_k(x) \): The \( k \)-th component of the mean vector
- \( \sigma_k^2(x) \): The \( k \)-th component of the variance

### 3.2 Interpretation

| KLD Value | Meaning |
|-----------|---------|
| **High** (\( > 0.1 \)) | Dimension carries information, deviates from prior |
| **Low** (\( \approx 0 \)) | Dimension collapsed to prior, not useful |

We select the **top-K dimensions** with highest average KLD across the dataset:

$$
\text{Active Dims} = \text{argtop}_K \left( \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \text{KLD}_k(x) \right)
$$

## 4. Algorithm Summary

```
Algorithm: Latent Traversal Visualization

Input: 
  - Trained VAE model (Encoder, Decoder)
  - Base image x₀
  - Number of dimensions K
  - Traversal range [v_min, v_max]
  - Number of steps N

Step 1: Find Active Dimensions
  For each batch in dataset:
    μ, log_var = Encoder(batch)
    KLD_per_dim += mean(-0.5 * (1 + log_var - μ² - exp(log_var)), dim=0)
  active_dims = top_K_indices(KLD_per_dim)

Step 2: Encode Base Image
  μ₀, _ = Encoder(x₀)

Step 3: Generate Traversal Grid
  For each k in active_dims:
    For each v in linspace(v_min, v_max, N):
      z = μ₀.copy()
      z[k] = v
      x_hat = Decoder(z)
      Add x_hat to grid row k

Output: Grid of K × N images
```

## 5. Disentanglement

A key goal of models like β-VAE and β-TCVAE is **disentanglement** — ensuring each latent dimension captures a single, independent factor of variation.

### 5.1 Ideal Disentanglement

In a perfectly disentangled latent space:
- Dimension 1 might control **hair color**
- Dimension 2 might control **smile**
- Dimension 3 might control **age**
- etc.

Traversing one dimension should change **only one** visual attribute.

### 5.2 Mathematical Condition

For disentanglement, we want the latent distribution to factorize:

$$
q(z|x) = \prod_{k=1}^{d} q(z_k|x)
$$

And ideally, each \( z_k \) corresponds to one generative factor in the data.

## 6. Practical Considerations

### 6.1 Traversal Range

The typical range \( [-3, 3] \) covers ~99.7% of a standard normal distribution (3-sigma rule). Going beyond may produce artifacts.

### 6.2 Number of Steps

- **10 steps**: Good balance between smoothness and computation
- **More steps**: Smoother transitions but slower

### 6.3 Selecting Base Image

Choose a "typical" image (close to the data mean) for clearer traversal effects. Unusual images may produce unexpected results.

## 7. Code Reference

See `latent_traversal.py` for implementation:

```python
# Key functions:
get_active_dim_indices(model, data_loader, top_k=10)  # Find active dimensions
visualize_traversal(model, base_image, active_dims)    # Generate traversal grid
```

## 8. References

- Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- Chen et al. (2018). "Isolating Sources of Disentanglement in Variational Autoencoders"
- Kingma & Welling (2014). "Auto-Encoding Variational Bayes"

