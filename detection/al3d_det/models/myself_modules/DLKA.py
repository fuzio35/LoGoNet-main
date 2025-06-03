import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable
import D3D
import math
from torch.nn import init

class DeformConv(nn.Module):

    def __init__(self,in_channels,groups,kernel_size=(3,3),padding=1,stride=1,dilation=1,bias=True):
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

# 3d  deform定义
class DeformConvFunction(Function):
    @staticmethod
    # CTX代表上下文对象 前向传播里面保存参数，反向传播通过这些参数保存梯度
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        #stride padding 核大小都是三个方向
        #input - B C D H W
        #OFFSET B   3 * dg * k_D * k_H * k_W  D_o h_o w_o
        # weight c_o c_in/g k_D k_H k_w
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = D3D.deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                         ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                         ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                         ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            D3D.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None

# 结合大核和可变形卷积的优势

# 大小可调节
# 对应着中间的部分


class DeformConv3D(nn.Module):

    def __init__(self, in_channels, output_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv3D,self).__init__()

        # 分组 判断是否能分成功
        if (in_channels % groups != 0):
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if output_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(output_channels, groups))
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.use_bias = bias
        self.im2col_step = im2col_step
        # out * in_per_group * kD * KH * KW
        self.weight = nn.Parameter(torch.Tensor(output_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(output_channels))
        self.reset_parameters()
        if not bias:
            self.bias.requires_grad = False

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
               offset.shape[1]
        return DeformConvFunction.apply(input,offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                        self.groups,self.deformable_groups,self.im2col_step)

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

# 一个新库 继承自上述类；上述类不能offset
class DeformConvPack(DeformConv3D):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)


        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset,
                                          self.weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)
