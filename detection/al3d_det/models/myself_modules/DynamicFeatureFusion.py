import torch
import torch.nn as nn


# 一个很简单的模块
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_atten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(output)
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output