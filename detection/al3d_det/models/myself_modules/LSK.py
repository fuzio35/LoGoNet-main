import torch
import torch.nn as nn

class LSKBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        # 两个可分离卷积核--逐通道卷积
        self.conv0 = nn.Conv3d(dim,dim,kernel_size=5,padding=2,groups=dim)
        self.conv_spatial = nn.Conv3d(dim,dim,kernel_size=7,stride=1,
                                      padding=9,groups=dim,dilation=3)
        self.conv1 = nn.Conv3d(dim,dim//2,kernel_size=1)
        self.conv2 = nn.Conv3d(dim,dim//2,kernel_size=1)
        # 生成注意力图
        self.conv_squeeze = nn.Conv3d(2,2,kernel_size=7,padding=3)
        self.conv3 = nn.Conv3d(dim//2,dim,kernel_size=1)
    
    def forward(self, x):
        # 两个逐通道卷积得到结果
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        # 逐点卷积 最终得到两个U--即得到两个感受野不同的特征
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        # 拼接
        attn = torch.cat([attn1, attn2], dim=1)
        # 两个池化
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # B 2 H W 
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 得到2个通道的注意力掩码--B 2 H W
        sig = self.conv_squeeze(agg).sigmoid()

        # 生成注意力图后进行元素级乘法
        attn = attn1 * sig[:, 0, :, :,:].unsqueeze(1) + attn2 * sig[:, 1, :, :,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
