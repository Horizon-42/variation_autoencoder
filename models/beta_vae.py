import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from lpips import LPIPS
from .resize_conv2d import ResizeConv2d
from.PID import PIDControl
from .cyclical_annealer import CyclicalAnnealer

def tv_loss(img: Tensor) -> torch.Tensor:
    """
    计算图像的平滑度，惩罚高频噪点
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    return (w_variance + h_variance) / (img.size(0) * img.size(1) * img.size(2))


class BetaVAE(BaseVAE):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 exp_kld_loss: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 image_size: int = 64,
                 enable_perceptual_loss: bool = True,
                 lpips_weight: float = 0.5,
                 tvl_weight: float = 1e-3,

                 max_steps: int = 100000,
                 n_cycles: int = 5,
                 ratio: float = 0.6,
                 circular_mode: str = 'linear',
                 **kwargs) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        :param in_channels: Input channels
        :type in_channels: int
        :param latent_dim: Description
        :type latent_dim: int
        :param hidden_dims: Description
        :type hidden_dims: List
        :param beta: Description
        :type beta: int
        :param gamma: Description
        :type gamma: float
        :param max_capacity: Description
        :type max_capacity: int
        :param exp_kld_loss: Expected KL Divergence loss for PID controller
        :param Capacity_max_iter: Description
        :type Capacity_max_iter: int
        :param loss_type: Description
        :type loss_type: str, choices=['H', 'B', 'PID]
        :param image_size: Description
        :type image_size: int
        :param enable_perceptual_loss: Description
        :type enable_perceptual_loss: bool
        :param lpips_weight: Description
        :type lpips_weight: float
        :param tvl_weight: Description
        :type tvl_weight: float
        :param kwargs: Description
        """
        super(BetaVAE, self).__init__()

        self.enable_perceptual_loss = enable_perceptual_loss
        # lpips model for perceptual loss
        # self.lpips_model = LPIPS(net='alex', verbose=False).eval()
        self.lpips_model = LPIPS(net='vgg', verbose=False).eval()
        # set requires_grad to False
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        # lpips loss weight, intial 0.1 to 0.5
        self.lpips_weight = lpips_weight
        self.tvl_weight = tvl_weight

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        # for PID controller
        self.exp_kld_loss = exp_kld_loss
        self.pid_controller = PIDControl()

        # Cyclical Annealer for beta
        self.annealer = CyclicalAnnealer(total_steps=max_steps,
                                         n_cycles=n_cycles,
                                         max_beta=beta,
                                         ratio=ratio,
                                         mode=circular_mode)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        cur_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cur_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1,
                              padding_mode="reflect"),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            cur_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # get flat_size and encoder_output_shape
        input_height, input_width = image_size
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, in_channels, input_height, input_width)
            encoder_output = self.encoder(dummy_input)
            # Get the flat size automatically (e.g., 512 * 2 * 2 = 2048)
            self.flat_size = encoder_output.view(1, -1).size(1)
            # Use encoder_output.shape[1:] to save the spatial dims (512, 2, 2) for the decoder
            self.encoder_output_shape = encoder_output.shape[1:]

        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.flat_size, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flat_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                    # ResizeConv2d(hidden_dims[i],
                                 hidden_dims[i + 1],
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
            # ResizeConv2d(hidden_dims[-1],
                         hidden_dims[-1],
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_output_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        if kwargs["is_val"] == False:
            self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']

        mse_loss = F.mse_loss(recons, input)
        
        recons_loss = mse_loss
        
        if self.enable_perceptual_loss:
            # Total Variation (TV) Loss, Penalize high-frequency noise
            tvl = tv_loss(recons)
            recons_loss += self.tvl_weight * tvl  # TV loss weight can be adjusted

            # 计算感知损失
            # 确保 self.lpips_model 和 input 在同一个 device 上
            if self.lpips_model.parameters().__next__().device != input.device:
                self.lpips_model = self.lpips_model.to(input.device)

            perceptual_loss = self.lpips_model(recons, input)
            # pikle the recons input if perceptual_loss is negative
            # LPIPS 返回的是 [Batch, 1, 1, 1]，需要取平均
            perceptual_loss = perceptual_loss.mean()
            recons_loss += self.lpips_weight * perceptual_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter *
                            self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        elif self.loss_type == 'PID':  # PID controller for beta-VAE
            beta, _ = self.pid_controller.pid(self.exp_kld_loss, kld_loss.item())
            loss = recons_loss + beta * kld_loss
        elif self.loss_type == 'Cyclical':
            beta = self.annealer(self.num_iter)
            loss = recons_loss + beta * kld_weight * kld_loss
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD_Loss':kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
