# VAE Model and Criterion Function Explanation

This document explains the Variational Autoencoder (VAE) architecture and the loss function defined in `variation_autoencoder_excersize.ipynb`.

## 1. Model Building

The `VAE` class defines the neural network architecture, which consists of three main components: the Encoder, the Reparameterization (Sampling) mechanism, and the Decoder.

### Encoder
The encoder is responsible for compressing the high-dimensional input data into a lower-dimensional latent space representation.
-   **Input Layer**: Takes a flattened MNIST image vector (28x28 = 784, `x_in`).
-   **Hidden Layer (`fc_e`)**: A fully connected linear layer maps the 784 inputs to a hidden dimension (default 500). A ReLU activation function is applied.
-   **Latent Projections**: The network splits into two separate paths to parameterize the latent distribution $q(z|x)$:
    -   `fc_mean`: Maps the hidden representation to the latent dimension (default 20) to predict the mean vector ($\mu$).
    -   `fc_logvar`: Maps the hidden representation to the latent dimension to predict the log-variance vector ($\log(\sigma^2)$).

### Reparameterization (`sample_normal`)
To perform backpropagation through the stochastic sampling of the latent variable $z$, the "reparameterization trick" is used.
-   **Standard Deviation**: calculated as $\sigma = \exp(0.5 \times \text{logvar})$.
-   **Noise Sampling**: A noise vector $\epsilon$ is sampled from a standard normal distribution $\mathcal{N}(0, I)$.
-   **Transformation**: The latent vector $z$ is computed as $z = \mu + \epsilon \cdot \sigma$.
This allows gradients to flow through $\mu$ and $\sigma$ while treating $\epsilon$ as a constant during optimization.

### Decoder
The decoder reconstructs the original input data from the latent vector $z$.
-   **Input**: Takes the latent vector $z$ (dimension 20).
-   **Hidden Layer (`fc_d1`)**: Maps $z$ back to the hidden dimension (500) with a ReLU activation.
-   **Output Layer (`fc_d2`)**: Maps the hidden representation back to the original input dimension (784).
-   **Activation**: A Sigmoid activation is applied to constrain the output values between 0 and 1, representing pixel intensities (or probabilities in the context of BCE).
-   **Output Shape**: The result is reshaped to `(batch_size, 1, 28, 28)`.

## 2. Criterion Function

The criterion function (loss function) aims to maximize the Evidence Lower Bound (ELBO), or equivalently minimize the negative ELBO. It consists of two terms:

### Binary Cross Entropy (BCE) Loss
This term measures the **reconstruction error** between the original input `x_in` and the decoded output `x_out`.
-   `F.binary_cross_entropy` compares the pixel values.
-   `size_average=False` implies the loss is summed over the batch.
-   It encourages the decoder to produce images that are as close as possible to the inputs.

### Kullback-Leibler Divergence (KLD) Loss
This term acts as a **regularizer** on the latent space.
-   It measures the difference between the learned latent distribution $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ and the prior distribution $p(z) = \mathcal{N}(0, I)$.
-   The formula used is the analytical solution for KLD between two Gaussians:
    $$ \mathcal{L}_{KLD} = -0.5 \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2) $$
-   In the code: `-0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))`.
-   This forces the latent variables to be distributed normally, preventing the model from "cheating" by spacing points too far apart or memorizing specific regions.

### Total Loss
The final loss is the sum of the reconstruction loss and the KLD loss, normalized by the batch size:
$$ \text{Loss} = \frac{\text{BCE} + \text{KLD}}{\text{batch\_size}} $$
