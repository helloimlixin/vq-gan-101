#
# Created on Sun Jul 02 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#
import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args) -> None:
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        # the embedding matrix is initialized uniformly
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(
            -1. / self.num_codebook_vectors, 1. / self.num_codebook_vectors
        )

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        # compute the distances between the latent vectors and the embedding vectors
        dist = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # codebook loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z - z_q.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()  # stop gradient for the backward pass

        z_q = z_q.permute(0, 3, 1, 2)  # convert to the original shape

        return z_q, min_encoding_indices, loss
