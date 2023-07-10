#
# File   : discriminator.py
#
# Implementation based on the PatchGAN discriminator.
# See:
# - https://arxiv.org/abs/1611.07004
# - https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/GANs/Pix2Pix/discriminator_model.py#L20C1-L21C1
# - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538
#
# Created on Mon Jul 10 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#

import torch.nn as nn


class CNNBlock(nn.Module):
    """CNN block for the discriminator."""

    def __init__(self, in_channels, out_channels, stride):
        """Constructor for the class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
        """
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            # see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            # see https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(num_features=out_channels),
            # see https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
            # also see https://stackoverflow.com/questions/69913781/is-it-true-that-inplace-true-activations-in-pytorch-make-sense-only-for-infere
            # for using inplace=True in inference
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_dim_last=64, num_layers=3) -> None:
        super(Discriminator).__init__()
        # create a list of feature dimensions for each layer
        feature_dims = [min(feature_dim_last * 2**i, 8) for i in range(num_layers)]

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=feature_dims[0],
                kernel_size=4,
                stried=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]  # layer initialization

        for feature_dim in feature_dims[1:]:  # skip the first layer
            layers.append(
                CNNBlock(
                    in_channels=in_channels,
                    out_channels=feature_dim,
                    stride=1 if feature_dim == feature_dims[-1] else 2,
                )
            )
            in_channels = feature_dim  # update the number of input channels

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )  # add the last layer

        self.model = nn.Sequential(*layers)  # create the model

    def forward(self, x):
        """forward pass of the discriminator.

        Args:
            x (torch.Tensor): input tensor.
        """
        return self.model(x)
