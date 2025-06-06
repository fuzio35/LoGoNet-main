import torch
import torch.nn as nn
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # k 卷积核大小
    # p 膨胀大小
    # d 膨胀率 保证输入域输出维度一致
    if d > 1:
        # 如果使用膨胀卷积
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

#
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    # c1 c2是输入输出通道 k-卷积核，s-步长，p-填充，d-膨胀率，g-分组数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class LAE(nn.Module):
    # Light-weight Adaptive Extraction
    # 轻量级自适应
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )
        # 下采样 输出通道是输入通道4倍 步长为2，以保证空间维度减半
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x

class MatchNeck_Inner(nn.Module):
    def __init__(self, channels):
        super(MatchNeck_Inner,self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),

        )
