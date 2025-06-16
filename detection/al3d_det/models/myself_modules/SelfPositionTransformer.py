import torch
import torch.nn as nn
from al3d_det.models.myself_modules.SPoTr.openpoints.models.backbone import  spotr as SPoTr

import torch
import torch.nn as nn

class SPoTrRoIEncoder(nn.Module):
    def __init__(self,
                 in_channels=4,           # 包含 xyz + intensity/feature 等
                 width=32,blocks=[1,2],strides=[1,4],
                 block='LPAMLP',nsample=16, #
                 radius=0.1,aggr_args={'feature_type': 'dp_fj', 'reduction': 'max'},
                 group_args={'NAME': 'ballquery',},num_layers=1,
                 num_heads=8,query_dim=128,
                 **kwargs):
        super().__init__()
          # 假设你之前的 encoder 类名

        # SPoTrEncoder 实际支持 B*N x N x C 形式输入
        self.encoder = SPoTr.SPoTrEncoder(
            in_channels=in_channels,
            width=width,
            blocks=blocks,
            strides=strides,
            block=block,
            nsample=nsample,
            radius=radius,
            aggr_args=aggr_args,
            group_args=group_args,
            num_layers=num_layers,
            is_cls=True,
            expansion=4,gamma=16,num_gp=16,tau=0.5, # 尽管在 SPoTrEncoder 内部可能不直接使用 tau，但可以传递
            task='cls',conv_args={'order': 'conv-norm-act'},act_args={'act': 'relu'},
            norm_args={'norm': 'bn'},
            radius_scaling=2,nsample_scaling=1,
        )

        self.linear = nn.Linear(64, query_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim,num_heads=num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim,query_dim*2),
            nn.ReLU(),
            nn.Linear(query_dim*2,query_dim),
        )

    def forward(self, roi_points,GDF_feat):
        """
        roi_points: Tensor [B*N_ROI, N_POINTS, C]  (C: xyz + features)
        """
        xyz = roi_points[..., :3].contiguous()                  # [BN, N, 3]
        features = roi_points[..., ].permute(0, 2, 1).contiguous()  # [BN, C, N]


        # 调用 SPoTrEncoder 提取最终向量 [BN, D,N]
        features_per_roi = self.encoder.forward_cls_feat(xyz, features)
        features_per_roi = features_per_roi.permute(0,2,1).contiguous()
        GDF_feat = GDF_feat.permute(0,2,1).contiguous()
        # 开始处理
        features_per_roi = self.linear(features_per_roi)
        attn_out,_ = self.cross_attn(GDF_feat,features_per_roi,features_per_roi)
        out = self.norm1(attn_out+GDF_feat)
        out = self.norm2(self.ffn(out)+out)
        out = out.permute(0,2,1).contiguous()
        return out
