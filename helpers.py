import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, channels) -> None:
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
    
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
        pass