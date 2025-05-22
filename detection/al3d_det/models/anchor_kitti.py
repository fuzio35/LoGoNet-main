import os

import torch
import torch.nn as nn
import numpy as np

from al3d_utils import common_utils
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils

from al3d_det.utils import nms_utils
from al3d_det.models import modules as cp_modules
from al3d_det.models.modules import backbone_3d, backbone_2d, dense_heads, roi_heads

# 基类的锚点生成
class ANCHORKITTI(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        # 模型配置与对应的类别、数据集
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        # 全局缓冲区
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        # 是否二阶段检测（RPN
        self.second_stage = model_cfg.SECOND_STAGE
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # 依次使用每个模块处理
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # 训练就得到损失
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # 否则得到后处理结果
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    # 得到训练损失
    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        # RPN头损失
        loss, tb_dict = self.dense_head.get_loss()
        # ROI损失 两个损失相加
        if self.second_stage:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn

        return loss, tb_dict, disp_dict

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
    # 创建网络
    def build_networks(self):
        model_info_dict = {
            # 对应的网格
            'module_list': [],
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
        }
        # 选取合适对应的模块进行初始化 这里是MeanVFE
        vfe = backbone_3d.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )
        # KITTI= 4
        model_info_dict['num_point_features'] = vfe.get_output_feature_dim()
        # Backbone3D
        backbone3d = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['num_point_features'] = backbone3d.num_point_features
        # MAPTOBEV -- HeightCompression
        map_to_bev = backbone_2d.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['num_bev_features'] = map_to_bev.num_bev_features
        # Backbone2D = Backbone2D
        backbone2d = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['num_bev_features'] = backbone2d.num_bev_features
        # head = AnchorHeadSingle
        dense_head = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.second_stage
        )

        self.add_module('vfe', vfe)
        self.add_module('backbone3d', backbone3d)
        self.add_module('map_to_bev', map_to_bev)
        self.add_module('backbone2d', backbone2d)
        self.add_module('dense_head', dense_head)
        # 依次添加模块
        module_list = [vfe, backbone3d, map_to_bev, backbone2d, dense_head]
        # RPN阶段
        # LoGoHeadKITTI
        if self.second_stage:
            roi_head = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
            )
            self.add_module('roi_head', roi_head)
            module_list.append(roi_head)

        return module_list
    # 后处理模块
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {} # 存贮召回信息
        pred_dicts = [] # 存储预测信息
        # 对每一个batch 
        for index in range(batch_size):
            # 如果提供了batch对应的索引
            if batch_dict.get('batch_index', None) is not None:
                # 此batch的提取预测框掩码
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                # 否则整个拿出来
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            # 提取预测框
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            # 提取分类结果
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    # 如果没有归一化要先归一化
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            # 已经提取了框和对应的分类类别了
            # false 就是说每个类别有自己的NMS
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                # 如果不是列表就转成列表
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    # 创建标签映射
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                # 初始化空列表 预测分数、预测标签与预测框
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    # 提取预测框
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    # NMS
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    # 标签
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    # 放上结果
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                # 否则公用一个nms阈值
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                # 获取每个预测的最大预测分数和标签
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                # 类别无关NMS进行处理
                selected, selected_scores = nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                # 代表输出原始分数
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            # 记录召回信息 下面的函数
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            # 放结果
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict
    @staticmethod
    # 传递盒子预测 
    # 预测 召回字典在参数传递时为空 batch索引，data_dict是batch的数据 阈值是列表
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        # 没有GT直接返回
        if 'gt_boxes' not in data_dict:
            return recall_dict
        # 取出ROI与GT
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        # 召回字典为空，先初始化之
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0
        # 对预测框清理全0的框
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        # 判断是否有有效的预测框
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                # 计算预测框与GT的IOU
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
            # 计算ROI与GT的IOU
            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
            # 对每一个阈值
            for cur_thresh in thresh_list:
                # 检测是否召回
                # 如果没有预测框
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    # 大于阈值的总数统计
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
