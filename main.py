#
# Created on Thu Jul 13 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#

# required imports
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as torchvisionutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, init_weights

def configure_optimizers(opt_args):
    optimizer_quantization = torch.optim.Adam(
        params=list(vqgan.encoder.parameters())
        + list(vqgan.decoder.parameters())
        + list(vqgan.codebook.parameters())
        + list(vqgan.quant_conv.parameters())
        + list(vqgan.post_quant_conv.parameters()),
        lr=opt_args.learning_rate,
        eps=1e-8,
        betas=(opt_args.beta1, opt_args.beta2),
    )
    optimizer_discriminator = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=opt_args.learning_rate,
        eps=1e-8,
        betas=(opt_args.beta1, opt_args.beta2),
    )
    return optimizer_quantization, optimizer_discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                        help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='data', help='Path to data (default: data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = 'data/flowers'

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    vqgan = VQGAN(args).to(device=args.device)
    discriminator = Discriminator(args).to(device=args.device)
    discriminator.apply(init_weights)
    perceptual_loss = LPIPS().eval().to(device=args.device)

    opt_vq, opt_disc = configure_optimizers(args)

    train_dataloader = load_data(args)
    steps_per_epoch = len(train_dataloader)

    for epoch in range(args.epochs):
        with tqdm(range(len(train_dataloader))) as progress_bar:
            for i, imgs in zip(progress_bar, train_dataloader):
                imgs = imgs.to(device=args.device)

                decoded_imgs, codebook_indices, q_loss = vqgan(imgs)

                disc_real = discriminator(imgs)
                disc_fake = discriminator(decoded_imgs)

                disc_factor = vqgan.adopt_weight(
                    args.disc_factor,
                    epoch * steps_per_epoch + i,
                    threshold=args.disc_start,
                )

                ploss = perceptual_loss(
                    imgs, decoded_imgs
                )  # LPIPS(real, fake)
                reconstruction_loss = torch.abs(imgs - decoded_imgs)  # L1(real, fake)
                perceptual_reconstruction_loss = (
                        args.perceptual_loss_factor * ploss
                        + args.rec_loss_factor * reconstruction_loss
                )
                perceptual_reconstruction_loss = (
                    perceptual_reconstruction_loss.mean()
                )  # mean over batch
                generator_loss = - torch.mean(disc_fake)

                # compute lambda
                lamda = vqgan.compute_lambda(perceptual_reconstruction_loss, generator_loss)

                # compute vector quantization loss
                vq_loss = perceptual_reconstruction_loss + q_loss + disc_factor * lamda * generator_loss

                disc_loss_real = torch.mean(F.relu(1. - disc_real))
                disc_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5 * (disc_loss_real + disc_loss_fake)

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                opt_disc.zero_grad()
                gan_loss.backward()

                opt_vq.step()
                opt_disc.step()

                if i % 10 == 0:
                    with torch.no_grad():
                        real_fake_images = torch.cat((imgs[:4], decoded_imgs.add(1).mul(0.5)[:4]))
                        torchvisionutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.png"),
                                                    nrow=4)

                progress_bar.set_postfix(
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                )
                progress_bar.update(0)

            torch.save(vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_{epoch}.pth"))
                
