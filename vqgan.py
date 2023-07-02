#
# Created on Sun Jul 02 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    """
    Implementation of the VQ-GAN model, see:
    Esser, Patrick, Robin Rombach, and Bjorn Ommer.
    "Taming transformers for high-resolution image synthesis."
    Proceedings of the IEEE/CVF conference on computer vision and
    pattern recognition. 2021.
    """

    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(
            args.latent_dim, args.latent_dim, kernel_size=1, stride=1
        ).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(
            device=args.device
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quant_conv_encoded = self.quant_conv(encoded)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded = self.decoder(post_quant_conv_mapping)

        return decoded, codebook_indices, q_loss

    def encode(self, x):
        encoded = self.encoder(x)
        quant_conv_encoded = self.quant_conv(encoded)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded)

        return codebook_mapping, codebook_indices, q_loss

    def decode(self, x):
        post_quant_conv_mapping = self.post_quant_conv(x)
        decoded = self.decoder(post_quant_conv_mapping)

        return decoded

    def compute_lambda(self, perceptual_loss, gan_loss):
        """
        Compute the lambda parameter for the loss function.
        """
        decoder_last_layer = self.decoder.model[-1]
        last_layer_weight = decoder_last_layer.weight
        # retain_graph option is on because we need to compute the gradients for backward pass
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lambda_ = torch.mean(torch.abs(perceptual_loss_grads)) / (
            torch.mean(torch.abs(gan_loss_grads)) + 1e-8
        )
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()

        return 0.8 * lambda_

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.0):
        """Delay the discriminator training by a certain number of iterations."""
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
