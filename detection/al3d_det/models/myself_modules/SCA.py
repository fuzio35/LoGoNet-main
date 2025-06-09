import torch
import torch.nn as nn

import torch
import torch.nn as nn

# 效果不大
class LayerNormFunction3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, D, H, W = x.size()
        # 求均值、方差然后归一化
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1, 1) * y + bias.view(1, C, 1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, D, H, W = grad_output.size()
        # 计算梯度
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0, 3, 4, 2]), grad_output.sum(dim=[0, 3, 4, 2]), None


class LayerNorm3d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction3D.apply(x, self.weight, self.bias, self.eps)


# 门控前先通道*2，再进入门控就行
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock3D(nn.Module):
    # DW--通道扩展倍率
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(c, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv3d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv3d(dw_channel // 2, c, kernel_size=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channel // 2, dw_channel // 2, kernel_size=1)
        )

        self.sg = SimpleGate()

        # 网格扩展倍率
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv3d(ffn_channel // 2, c, kernel_size=1)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma