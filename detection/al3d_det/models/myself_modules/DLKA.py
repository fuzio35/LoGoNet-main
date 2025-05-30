import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.utils import _triple

class DeformConv(nn.Module):

    def __init__(self,in_channels,output_channels,kernel_size=(3,3),stride=1,padding=1,
                 dilation=1,groups=1,deformable_groups=1,bias=True):
        super.__init__()
        # 对核大小产生对应的偏移量 所以输出通道大小是 kernel_size
        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1], # 每个位置有两个偏移量
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias
        )

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels= in_channels, # 每个位置有两个偏移量
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False
        )
        
    def forward(self,x):
        offset = self.offset_net(x)
        out = self.deform_conv(x,offset)
        return out

# 结合大核和可变形卷积的优势   
# 大小可调节
# 对应着中间的部分
class deformable_LKA(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv0 = DeformConv(dim,groups=dim,padding=2,kernel_size=(5,5))
        self.conv_spatial = DeformConv(dim,kernel_size=(7,7),stride=1,padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        # 残差连接
        x = x + shorcut
        return x
    

class DeformConv3D(nn.Module):

    def __init__(self,in_channels,output_channels,kernel_size=,stride=1,padding=1,
                 dilation=1,groups=1,deformable_groups=1,bias=True):
        super.__init__()

        # 分组 判断是否能分成功
        if(in_channels % groups != 0):
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if output_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.kernel_size = (kernel_size)
