#
# Created on Fri Jun 16 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#
# This file contains modules that are used in both the encoder and the decoder
# of the model.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    '''
    A nice alternative to batch normalization. It is more suitable for small
    batch size, and it is also more stable than batch normalization. Generally
    good for vision problems with intensive memory usage.
    '''
    def __init__(self, channels) -> None:
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    """
    An activation function attained by Neural Architecture Search (NAS) with
    a little bit of modification. It is a smooth approximation of ReLU. Its
    most distinctive property is that has a non-monotonic "bump" and has the
    following properties:
    - Non-monotonicity
    - Unboundedness
    - Smoothness
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        if self.in_channels != self.out_channels: # handle dimension mismatch for skip connections
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x) # skip connection

class UpSampleBlock(nn.Module):
    '''Up-Sample Block, consists of a single convolutional layer and an additional
    prescale of the input features, resulting in a 2x up-sampling of the input.
    '''
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2) # up-sample by nearest-neighbor interpolation
        return self.conv(x)

class DownSampleBlock(nn.Module):
    '''Just a reverse of UpSampleBlock, consists of a single convolutional layer and
    a padding operation before the downsampling operation, resulting in a 2x down-sampling
    of the input, off by 1 pixel.
    '''
    def __init__(self, channels) -> None:
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        padding = (0, 1, 0, 1)
        x = F.pad(x, padding, mode='constant', value=0)
        
        return self.conv(x)

class NonLocalBlock(nn.Module):
    '''
    Non-local block, used for long-range dependencies. It is a generalization of
    the self-attention mechanism. See,
    Wang, Xiaolong, et al. "Non-local neural networks."
    Proceedings of the IEEE conference on computer vision and
    pattern recognition. 2018. 
    '''
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        
        self.group_norm = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        hidden = self.group_norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        
        # some reshaping for matrix multiplication
        b, c, h, w = q.shape
        
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        
        attn = torch.bmm(q, k) # batch matrix multiplication for attention
        attn = attn * (int(c) ** (-0.5)) # scaling factor
        attn = F.softmax(attn, dim=2) # softmax along the last dimension to get the probabilies as attention weights
        attn = attn.permute(0, 2, 1) # transpose for matrix multiplication
        
        a = torch.bmm(v, attn)
        a = a.reshape(b, c, h, w)
        
        return x + a # residual connection for baseline performance guarantee

        
