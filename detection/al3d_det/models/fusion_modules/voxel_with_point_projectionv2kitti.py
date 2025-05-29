import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from .deform_fusion import DeformTransLayer
from .point_to_image_projectionv2 import Point2ImageProjectionV2
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock1D

# 交叉部分融合
class VoxelWithPointProjectionV2KITTI(nn.Module):
    def __init__(self, 
                fuse_mode,  # 'crossattention_deform' 融合模式
                interpolate, # 是否插值 true
                voxel_size, 
                pc_range,  # 点云范围
                image_list, # ['camera_0', 'camera_1', 'camera_2', 'camera_3', 'camera_4']
                image_scale=1,  # 缩放比例
                depth_thres=0, # 深度阈值
                mid_channels = 16,
                double_flip=False, # 双重翻转
                dropout_ratio=0, #
                layer_channel=None, # 每层通道
                activate_out=True, # 是否激活
                fuse_out=False): # 是否融合后输出 true
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        # 点投影模块
        self.point_projector = Point2ImageProjectionV2(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        # 融合模式 crossattention_deform
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        self.mid_channels = mid_channels
        self.dropout_ratio = dropout_ratio
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        # 融合模式有直接连接和注意力
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            #  {'layer1': 64}
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)
        elif self.fuse_mode == 'crossattention_deform':
            # 注意力机制
            # 点特征投影
            self.pts_key_proj = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            # 点特征变换
            self.pts_transform = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            # 变形注意力层
            self.fuse_blocks = DeformTransLayer(d_model=self.mid_channels, \
                    n_levels=1, n_heads=4, n_points=4)
            # 如果融合输出，就两个融合一下
            if self.fuse_out:
                self.fuse_conv = nn.Sequential(
                    nn.Linear(self.mid_channels*2, self.mid_channels),
                    # For pts the BN is initialized differently by default
                    # TODO: check whether this is necessary
                    nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU())
    # 特征融合
    def fusion_back(self, voxel_feat, layer_name):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        fuse_feat = torch.zeros(voxel_feat.shape).to(voxel_feat.device)
        # 图像与体素拼接一下 就处理了 但是这里不需要图像 全0就可以
        concat_feat = torch.cat([fuse_feat.permute(1,0).contiguous(), voxel_feat.permute(1,0).contiguous()], dim=0)
        voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
        voxel_feat = voxel_feat.permute(1,0).contiguous()
        return voxel_feat

    # 特征融合
    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if self.fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            # 相加
            voxel_feat = voxel_feat + fuse_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            # 求均值
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0).contiguous()) / 2
        elif self.fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            # 拼接
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0).contiguous()], dim=0)
            # 处理
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'crossattention':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]].permute(1,0).contiguous()
            voxel_feat = self.fuse_blocks(fuse_feat.unsqueeze(0), voxel_feat.unsqueeze(0))
        else:
            raise NotImplementedError
        
        return voxel_feat
    # 融合 
    def fusion_withdeform(self, img_pre_fuse, voxel_feat):
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feat)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([pts_pre_fuse, img_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def forward(self, batch_dict, point_features, point_coords, layer_name=None, img_conv_func=None):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
            encoded_voxel: (N, C), Sparse Voxel featuress
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
            voxel_features: (N, C), Sparse Image voxel features
        """
        # 初始化列表
        voxel_fusefeatlist = []
        final_img_voxels = point_features.new_zeros((point_features.shape[0], self.mid_channels))
        # 预处理
        pts_feats_org = self.pts_key_proj(point_features)

        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]
        image_feat = batch_dict['image_features'][layer_name+'_feat2d']
        # 有插值就插值
        if self.image_interp:
            image_feat = nn.functional.interpolate(image_feat, (h, w), mode='bilinear')
        image_with_voxelfeatures = []
        filter_idx_list = []
        for b in range(batch_size):
            # 获取图像特征与对应的标定矩阵
            image_feat_batch = image_feat[b]

            calib = calibs[b]
            index_mask = point_coords[:,0]==b
            # 选择坐标
            point_grid_batch = point_coords[index_mask][:, 1:]
            # 选择特征
            voxel_features_sparse = pts_feats_org[index_mask]
            # 去掉增强
            if 'aug_matrix_inv' in batch_dict.keys():
                aug_matrix_inv = batch_dict['aug_matrix_inv'][b]
                for aug_type in ['translate', 'rescale', 'rotate', 'flip']:
                    if aug_type in aug_matrix_inv:
                        if aug_type == 'translate':
                            point_grid_batch = point_grid_batch + torch.Tensor(aug_matrix_inv[aug_type]).to(point_grid_batch.device)
                        else:
                            point_grid_batch = point_grid_batch @ torch.Tensor(aug_matrix_inv[aug_type]).to(point_grid_batch.device)
            # 投影到图像
            voxels_2d, _ = calib.lidar_to_img(point_grid_batch[:, :].cpu().numpy())

            voxels_2d_int = torch.Tensor(voxels_2d).to(image_feat_batch.device).long()
            # 选择在范围内的点
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            # 放入list
            filter_idx_list.append(filter_idx)
            # 筛选有效坐标
            image_grid = voxels_2d_int[filter_idx]
            image_features_batch = image_feat_batch.unsqueeze(0)
            
            _, channel_num, f_h, f_w = image_features_batch.shape
            # K V
            flatten_img_feat = image_features_batch.permute(0, 2, 3, 1).reshape(1, f_h*f_w, channel_num)
            # 如果不插值 就调整坐标大小
            if not self.image_interp:
                raw_shape = tuple(batch_dict['image_shape'][b].cpu().numpy())
                image_grid = image_grid.float()
                image_grid[:,0] *= (f_w/raw_shape[1])
                image_grid[:,1] *= (f_h/raw_shape[0])
                image_grid = image_grid.long()
            # 归一化
            ref_points = image_grid.float()
            ref_points[:, 0] /= f_w
            ref_points[:, 1] /= f_h
            ref_points = ref_points.reshape(1, -1, 1, 2)

            # 处理体素特征
            N, Len_in, _ = flatten_img_feat.shape
            # Q
            pts_feats = voxel_features_sparse[filter_idx].reshape(1, -1, self.mid_channels)
            # 空间形状
            level_spatial_shapes = pts_feats.new_tensor([(f_h, f_w)], dtype=torch.long)
            level_start_index = pts_feats.new_tensor([0], dtype=torch.long)
            # 融合
            voxel_features_sparse[filter_idx] = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index).squeeze(0)
            # 最终溶解结果
            image_with_voxelfeatures.append(voxel_features_sparse)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures, dim= 0)
        # 
        final_voxelimg_feat = self.fusion_withdeform(image_with_voxelfeatures, point_features)

        return final_voxelimg_feat
        