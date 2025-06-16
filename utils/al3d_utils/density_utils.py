from collections import defaultdict

import torch

from . import common_utils, voxel_aggregation_utils
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
from al3d_utils.ops.pointnet2.pointnet2_stack.pointnet2_modules import pointnet2_utils



# 计算每个部分中点的数量
def find_num_points_per_part(batch_points, batch_boxes, grid_size):
    """
    Args:
        batch_points: (N, 4)
        batch_boxes: (B, O, 7)
        grid_size: G
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    # 就是计算每个ROI的每个网格
    assert grid_size > 0

    # 取出当前索引批次和当前点坐标
    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    # 存储每部分的点数
    points_per_parts = []

    # 遍历每个batch
    for i in range(batch_boxes.shape[0]):
        # box数量 ---- O * 7
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        # 取出当前batch的点特征  N * 4
        points = batch_points[bs_mask]
        # 使用GPU加速的方法判断哪些点在当前框内
        # N' * 1;代表每个点对应的框，-1表示不在
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)
        # 标记在框里的点
        points_in_boxes_mask = box_idxs_of_pts != -1
        # 有效点数量 * 7 ；获取的是每个有效的点对应的框
        box_for_each_point = boxes[box_idxs_of_pts.long()][points_in_boxes_mask]
        # 点坐标转化为相对框的局部坐标
        xyz_local = points[points_in_boxes_mask] - box_for_each_point[:, 0:3]
        # 角度旋转--这样可以保证每个框的处理都是一致的
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local[:, None, :], -box_for_each_point[:, 6]
        ).squeeze(dim=1)
        # Change coordinate frame to corner instead of center of box

        # 这里插入一些额外的特征提取器
        # 对原始点云特征进行池化，以填充


        # 转换到角落
        xyz_local += box_for_each_point[:, 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        # 防止坐标越界
        xyz_local = torch.min(xyz_local, box_for_each_point[:, 3:6] - 1e-5)
        # 局部坐标转化为网格坐标
        xyz_local_grid = (xyz_local // (box_for_each_point[:, 3:6] / grid_size))
        # 每个点的框索引与网格坐标相连接  N * 4
        xyz_local_grid = torch.cat((box_idxs_of_pts[points_in_boxes_mask].unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        # 获取网格坐标和对应的网格点的数量--去除重复的框坐标;得到每个重复坐标的计数
        # part-idx-- 4 * X  ; per_part - X 
        part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
        # 计算每个部分的点数--每个框在每个网格的点数
        points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()
        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)


# 进行修改，寻找地方插入对PIE的优化
# 在每个 3D RoI 内进行点的体素划分 并统计每个体素内的点数或质心等信息
def find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=False):
    """
    Args:
        batch_points: (N, 4) N是输入点数
        batch_boxes: (B, O, 7) O是最大检测数量,即128
        grid_size: G
        max_num_boxes: M 最多20个框
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    assert grid_size > 0
    # 取出batch
    batch_idx = batch_points[:, 0]
    # 取出点云坐标
    batch_points = batch_points[:, 1:4]



    points_per_parts = []
    for i in range(batch_boxes.shape[0]):
        # 对每一个盒子 取出当前掩码的
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        # 取出batch对应点
        points = batch_points[bs_mask]
        # 判断每个点是否在框内,得到每个点对应的框id
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_multi_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0), max_num_boxes).squeeze(0)
        # 取出每个点对应的框
        box_for_each_point = boxes[box_idxs_of_pts.long()]

        # 局部坐标系的转换，点对每个框的相对坐标全取出来，此时剩下20个框
        # 转换至ROI内部坐标系，ROI内部划分网格
        xyz_local = points.unsqueeze(1) - box_for_each_point[..., 0:3]
        xyz_local_original_shape = xyz_local.shape
        xyz_local = xyz_local.reshape(-1, 1, 3) # [445080, 1, 3]
        # Flatten for rotating points [445080, 1, 3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -box_for_each_point.reshape(-1, 7)[:, 6]
        )
        # Change coordinate frame to corner instead of center of box
        # 从框中心转移至角落--坐标原点的重新设定

        xyz_local_corner = xyz_local.reshape(xyz_local_original_shape) + box_for_each_point[..., 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        # 将坐标约束在网格大小内---
        # xyz_local--原始点云在坐标，[445080, 1, 3]
        # xyz_local_corner--原点移动以后的坐标，([22254, 20, 3])
        # xyz_local_grid -- 获取对应的网格坐标


        # 最终---点映射至ROI的体素网格坐标系内，转化为网格坐标
        xyz_local_grid = (xyz_local_corner / (box_for_each_point[..., 3:6] / grid_size))
        # 筛选不正常坐标
        points_out_of_range = ((xyz_local_grid < 0) | (xyz_local_grid >= grid_size) | (xyz_local_grid.isnan())).any(-1).flatten()
        # 点的局部坐标 + 所在框
        # 加上了盒子的id
        xyz_local_grid = torch.cat((box_idxs_of_pts.unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        xyz_local_grid = xyz_local_grid.reshape(-1, xyz_local_grid.shape[-1])
        # Filter based on valid box_idxs
        valid_points_mask = (xyz_local_grid[:, 0] != -1) & (~points_out_of_range)
        # 取出最终的有效的点
        # 这里已经完成了体素划分，特征为所在体素坐标与对应的所在网格
        xyz_local_grid = xyz_local_grid[valid_points_mask]
        # 也是先取出有效的点，落入网格的点
        xyz_local = xyz_local[valid_points_mask].squeeze(1)

        # 这里对点的特征进行提取


        if return_centroid:

            # 对每个点求质心，得到每个点的质心
            # part_idxs是去重以后的所有点，points_per_part是每个点的出现次数
            centroids, part_idxs, points_per_part = voxel_aggregation_utils.get_centroid_per_voxel(xyz_local, xyz_local_grid)
            # 质心添加到特征,此时具有的为---质心的坐标+每个点的重复次数
            points_per_part = torch.cat((points_per_part.unsqueeze(-1), centroids), dim=-1)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1]), dtype=points_per_part.dtype, device=points.device)
            else:
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1])).to_dense()
        else:
            part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size), dtype=points_per_part.dtype, device=points.device)
            else:
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()

        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)

# get_fixed_length_roi_points(afasd, batch_boxes, 256, max_num_boxes,1)
# 得到每个ROI对应的特征与点
def get_fixed_length_roi_points(batch_points, batch_boxes,
                                max_points_per_boxes,max_nums_roi_boxes,
                                other_feature_dims):

    # BATCH 与 ROI总数
    batch_size = batch_boxes.shape[0]
    total_rois = batch_boxes.shape[0] * batch_boxes.shape[1]

    #初始化存储 存储每个ROI的点
    points_per_roi = torch.zeros(
        (total_rois, max_points_per_boxes, other_feature_dims+3),
        dtype=batch_points.dtype,
        device=batch_points.device
    )
    # 掩码 代表说当前内部多少点是有效的
    points_per_roi_mask = torch.zeros(
        (total_rois, max_points_per_boxes),
        dtype=torch.bool,
        device=batch_points.device
    )
    #提取所有点的batch与特征
    batch_points_idx_all = batch_points[:,0]
    all_points_xyz_feat = batch_points[:, 1:]


    # 记录当前ROI在展平的位置
    current_roi_idx_in_flat_batch = 0
    # 开始遍历每个batch
    for i in range(batch_size):
        # 场景掩码与当前ROI和当前点
        current_scenes_mask = (batch_points_idx_all == i)
        current_scenes_points_feat = all_points_xyz_feat[current_scenes_mask]
        current_scenes_boxes = batch_boxes[i]
        # 取出点坐标
        current_scenes_points_xyz = current_scenes_points_feat[:, 0:3]
        # 如果没有点或者没有ROI 跳过
        if current_scenes_points_xyz.shape[0] == 0 or current_scenes_boxes.size(0) == 0 :
            current_roi_idx_in_flat_batch += current_scenes_boxes.size(0)
            continue

        # 获取点所在的ROI索引
        roi_indices_for_each_point = roiaware_pool3d_utils.points_in_multi_boxes_gpu \
            (current_scenes_points_xyz.unsqueeze(0), current_scenes_boxes.unsqueeze(0), max_nums_roi_boxes).squeeze(0)

        #遍历每个ROI
        for roi_idx in range(current_scenes_boxes.shape[0]):
            # 找到ROI内的点
            mask_points_in_now_roi = (roi_indices_for_each_point == roi_idx).any(dim=1)
            points_in_this_roi_data_xyz = current_scenes_points_xyz[mask_points_in_now_roi]
            points_in_this_roi_data_feature = current_scenes_points_feat[mask_points_in_now_roi]
            # 判断是否采样
            num_points_in_this_roi = points_in_this_roi_data_xyz.shape[0]
            if num_points_in_this_roi > 0 :
                #要么填充要么采样
                if num_points_in_this_roi > max_points_per_boxes:
                    points_in_this_roi_data_xyz = points_in_this_roi_data_xyz.unsqueeze(0)
                    sample_idx = pointnet2_utils.furthest_point_sample(points_in_this_roi_data_xyz, max_points_per_boxes)
                    sample_idx = sample_idx.squeeze(0)
                    sampled_points = points_in_this_roi_data_feature[sample_idx.long()]
                    points_per_roi[current_roi_idx_in_flat_batch,:max_points_per_boxes] = sampled_points
                    points_per_roi_mask[current_roi_idx_in_flat_batch,:max_points_per_boxes] = True
                else:
                    points_per_roi[current_roi_idx_in_flat_batch,:num_points_in_this_roi] = points_in_this_roi_data_feature
                    points_per_roi_mask[current_roi_idx_in_flat_batch,:num_points_in_this_roi] = True
            current_roi_idx_in_flat_batch += 1
    points_per_roi = points_per_roi.contiguous()
    return points_per_roi,points_per_roi_mask