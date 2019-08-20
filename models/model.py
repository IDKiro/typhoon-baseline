import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Network, self).__init__()

        num_feat = 32
        
        self.inconv = nn.Conv3d(in_channels, num_feat, 3, padding=1)
        self.mainconv = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        self.outconv = nn.Conv3d(num_feat, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        x = self.mainconv(x)
        x = self.outconv(x)

        return x